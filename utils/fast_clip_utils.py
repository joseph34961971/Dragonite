'''
for continuous drag
2024-02-26
'''

import torch
import copy
from tqdm import tqdm
import numpy as np
pdist = torch.nn.PairwiseDistance(p=2)
import torch.nn.functional as F
from utils.augmentations import ImageAugmentations
import clip
from torchvision import transforms
import torchvision

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
# from .temp import EditStableDiffusion, unsupervised_loco_edit


# from .edit import EditStableDiffusion
# from utils.define_argparser import parse_args, preset


def judge_edge(point_tuple,invert_code_d):
    y,x = point_tuple[0],point_tuple[1]
    max_y,max_x = invert_code_d.shape[2],invert_code_d.shape[3]
    y = 0 if y<0 else y
    x = 0 if x<0 else x
    y = int(max_y-1) if y>max_y-1 else y
    x = int(max_x-1) if x>max_x-1 else x
    new_point_tuple = (y,x)
    return new_point_tuple


def draw_arrow_graph_with_bg(points,vectors,img_path=0):
    import matplotlib.pyplot as plt
    import numpy as np
    
    # background = plt.imread(img_path)
    fig, ax = plt.subplots()    
    ax.scatter(points[:, 0], points[:, 1], color='red') 
    for point, vector in zip(points, vectors):
        ax.arrow(point[0], point[1], vector[0], vector[1], head_width=1, head_length=2, fc='blue', ec='black')
    ax.axis('off')
    plt.grid(False)  
    plt.savefig('./arrow_chart_v2.png') 

def get_rectangle(mask: torch.Tensor):
    # get the rect of 1 in mask
    N,X,H,W = mask.shape  # eg:torch.Size([1, 1, 64, 64])   mask_cp_handle
    index_1 = torch.nonzero(mask)   
    min_y,min_x = torch.min(index_1,dim=0)[0][-2:]
    max_y,max_x = torch.max(index_1,dim=0)[0][-2:]
    left_top = torch.Tensor((min_y, min_x)).to(device=mask.device)
    left_bottom = torch.Tensor((min_y, max_x)).to(device=mask.device)
    right_top = torch.Tensor((max_y, min_x)).to(device=mask.device)
    right_bottom = torch.Tensor((max_y, max_x)).to(device=mask.device)
    rect = torch.stack((left_top, left_bottom, right_top, right_bottom),dim=0).to(device=mask.device)
    return rect, left_top, left_bottom, right_top, right_bottom


   
def interpolation(x):
    assert x.dim() == 4, "Input tensor x should have shape (1, C, N, M)"
    batch_size, channels, N, M = x.shape 

    for b in range(batch_size):
        zero_positions = (x[b, 0] == 0)

        for i in range(N):
            for j in range(M):
                if zero_positions[i, j]:
                    values = []  
                    weights = [] 

                    for k in range(1, j + 1):
                        if j - k >= 0 and x[b, 0, i, j - k] != 0:
                            values.append(x[b, :, i, j - k])
                            weights.append(1 / k)
                            break

                    for k in range(1, M - j):
                        if j + k < M and x[b, 0, i, j + k] != 0:
                            values.append(x[b, :, i, j + k])
                            weights.append(1 / k)
                            break

                    for k in range(1, i + 1):
                        if i - k >= 0 and x[b, 0, i - k, j] != 0:
                            values.append(x[b, :, i - k, j])
                            weights.append(1 / k)
                            break

                    for k in range(1, N - i):
                        if i + k < N and x[b, 0, i + k, j] != 0:
                            values.append(x[b, :, i + k, j])
                            weights.append(1 / k)
                            break

                    if weights:
                        total_weight = sum(weights)
                        interpolated_value = sum(w * v for w, v in zip(weights, values)) / total_weight
                        x[b, :, i, j] = interpolated_value

    return x


def interpolation_mean_adjusted(x):
    """
    Mean-adjusted interpolation using NIN (Normalization to Interpolated Norms)
    and channel-wise mean adjustment (nin/chm) for diffusion latents.

    Args:
        x (torch.Tensor): Input tensor of shape (1, C, N, M), where nulls (holes)
                          are marked by zeros in channel 0.

    Returns:
        torch.Tensor: Filled tensor with consistent norm and mean-adjusted values.
    """
    assert x.dim() == 4, "Input tensor x should have shape (1, C, N, M)"
    batch_size, channels, N, M = x.shape 
    x = x.clone().float()
    eps = 1e-8

    for b in range(batch_size):
        zero_positions = (x[b, 0] == 0)  # (N, M)

        # --- Step 1: Compute channel-wise mean over non-zero positions ---
        channel_means = []
        for c in range(channels):
            known_vals = x[b, c][~zero_positions]
            mean_c = known_vals.mean() if known_vals.numel() > 0 else 0.0
            channel_means.append(mean_c)
            x[b, c][~zero_positions] -= mean_c
        channel_means = torch.tensor(channel_means, device=x.device).view(channels, 1, 1)

        # --- Step 2: Interpolate each zero-position using NIN with channel-wise mean adjustment ---
        for i in range(N):
            for j in range(M):
                if zero_positions[i, j]:
                    values = []
                    weights = []

                    # search LEFT
                    for k in range(1, j + 1):
                        jj = j - k
                        if x[b, 0, i, jj] != 0:
                            values.append(x[b, :, i, jj])
                            weights.append(1 / k)
                            break

                    # search RIGHT
                    for k in range(1, M - j):
                        jj = j + k
                        if x[b, 0, i, jj] != 0:
                            values.append(x[b, :, i, jj])
                            weights.append(1 / k)
                            break

                    # search UP
                    for k in range(1, i + 1):
                        ii = i - k
                        if x[b, 0, ii, j] != 0:
                            values.append(x[b, :, ii, j])
                            weights.append(1 / k)
                            break

                    # search DOWN
                    for k in range(1, N - i):
                        ii = i + k
                        if x[b, 0, ii, j] != 0:
                            values.append(x[b, :, ii, j])
                            weights.append(1 / k)
                            break

                    if weights:
                        total_weight = sum(weights)
                        weights_tensor = torch.tensor(weights, dtype=x.dtype, device=x.device)
                        values_tensor = torch.stack(values, dim=0)

                        # Linear interpolation  
                        interpolated_value = (weights_tensor.view(-1, 1) * values_tensor).sum(dim=0) / total_weight

                        # --- NIN scaling ---
                        neighbor_norms = values_tensor.norm(p=2, dim=1)
                        target_norm = (weights_tensor * neighbor_norms).sum() / total_weight
                        current_norm = interpolated_value.norm(p=2)

                        if current_norm > eps:
                            interpolated_value *= (target_norm / (current_norm + eps))

                        x[b, :, i, j] = interpolated_value

        # --- Step 3: Add back channel-wise means ---
        x[b] += channel_means

    return x


def get_circle(mask: torch.Tensor):
    rect, left_top, left_bottom, right_top, right_bottom = get_rectangle(mask=mask)
    center = torch.Tensor(((left_top[0] + right_bottom[0]) / 2, (left_top[1] + right_bottom[1]) / 2)).to(device=mask.device)  # y,x
    radius = pdist(center, left_top) 
    return center,radius


def get_scale_factor(C, A, OA, d_OA, R, O):
    '''
    xA, yA = A  xB, yB = B  xC, yC = C
    '''
    # print("\n=============================")
    AC =  C-A  
    d_AC = torch.norm(AC)
    e_AC = AC/d_AC
    # print(f"O:{O}   \nA:{A} \nC:{C}  \nAC:{AC} \nd_AC:{d_AC} \ne_AC:{e_AC}  \nOA:{OA} \ntorch.dot(AC, OA):{torch.dot(AC, OA)}")
    L0 = torch.dot(AC, OA) / d_AC             #  |G1 A|    θ>90，L0<0
    L1 = torch.sqrt(R**2 - d_OA**2 + L0**2)   #  |G1 P|
    AP = (L1-L0)*e_AC  # $L1-L0=|G_1P|-|G_0A| =|G_1P| |G_1A| = |AP|$
    PC = AC-AP
    # print(f"L0:{L0} \nL1:{L1}  \nAP:{AP} \nPC:{PC}")
    scale_factor = torch.norm(PC)/torch.norm(AP)  # |PC|/|AP| == |CD|/|AB|
    return scale_factor

def transform_point(point, shift_yx, scale_factor):
    shift_yx = shift_yx*scale_factor
    point_new = torch.round(point+shift_yx)
    return point_new


def drag_stretch_with_clip_grad(model,invert_code,text_embeddings,t,handle_points,target_points,mask_cp_handle,args,shift_yx=None,fill_mode='interpolation',projection_mode="Naive"):
    print("Running drag_stretch_with_clip_grad")
    print(args)
    use_naive = False
    use_jacobian = False
    use_locoedit = False
    grad_global = None
    clip_semantic_direction = None

    if projection_mode == "Naive":
        print(f"use naive projection")
        use_naive = True
    elif projection_mode == "Jacobian":
        print(f"use jacobian projection")
        use_jacobian = True
    elif projection_mode == "LOCOEdit":
        print(f"use locoedit projection")
        use_locoedit = True
    else:
        raise ValueError(f"projection method {projection_mode} not supported")

    assert len(handle_points) == len(target_points), \
        "number of handle point must equals target points"
    if text_embeddings is None:
        text_embeddings = model.get_text_embeddings(args.prompt)
    for param in model.unet.parameters():
        param.requires_grad = False
    for param in model.vae.parameters():
        param.requires_grad = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = clip.load("ViT-B/16", device=device)[0].eval().requires_grad_(False)

    invert_code_d = copy.deepcopy(invert_code)
    if fill_mode == 'ori':
        pass
    if fill_mode == '0':
        invert_code_d[(mask_cp_handle>0).repeat(1,4,1,1)] = 0
    if fill_mode == "interpolation":
        invert_code_d[(mask_cp_handle>0).repeat(1,4,1,1)] = 0 
    if fill_mode == "random":
        invert_code_d[(mask_cp_handle>0).repeat(1,4,1,1)] = torch.rand_like(invert_code_d)[(mask_cp_handle>0).repeat(1,4,1,1)].to(device=invert_code_d.device)

    ### Naive approach: extract CLIP global gradient from the latent (invert_code = z_t)
    def cal_global_grad(invert_code, text_embeddings, t, model, args):
        device = model.device  # Ensure all tensors are on the same device
        z = invert_code.detach().to(device).requires_grad_(True)
        image_aug = ImageAugmentations(224, 1).to(device)
        text_embeddings = text_embeddings.to(device)
        t = t.to(device)

        unet_output = model.unet(z, t, encoder_hidden_states=text_embeddings)
        x_prev_0, pred_x0 = model.step(unet_output, t, z)
        pred_image = 1 / 0.18215 * pred_x0.to(dtype=z.dtype)  # Match dtype with z
        pred_image = model.vae.decode(pred_image)['sample'].to(device)
        pred_image = (pred_image / 2 + 0.5).clamp(0, 1)

        clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )

        aug_img = image_aug(pred_image).add(1).div(2)
        clip_img = clip_normalize(aug_img).to(device)  # Ensure clip_img is on the same device

        drag_prompt = clip.tokenize(args.drag_prompt).to(device)
        image_features = clip_model.encode_image(clip_img).to(device)
        text_features = clip_model.encode_text(drag_prompt).to(device)
        x = F.normalize(image_features, dim=-1)
        y = F.normalize(text_features, dim=-1)
        clip_loss = 1 - (x @ y.t()).squeeze()
        clip_loss = args.clip_loss_coef * clip_loss
        print(f'clip loss:{clip_loss}')

        # Ensure clip_loss is on the same device as z
        clip_loss = clip_loss.to(device)

        grad_global = torch.autograd.grad(clip_loss, z)[0]

        # visualize_grad_global(grad_global, z)

        del pred_image, clip_img, aug_img, image_features, text_features
        return grad_global
    
    def get_clip_semantic_direction(invert_code, t, model, args, clip_model):
        """
        Returns LOCOEdit-style semantic direction for the CLIP loss (w.r.t. the latent code)
        """
        print("calculate LOCOEdit-style semantic direction")
        device = model.device
        z = invert_code.detach().to(device).requires_grad_(True)
        t = t.to(device)
        image_aug = ImageAugmentations(224, 1).to(device)
        drag_prompt = clip.tokenize(args.drag_prompt).to(device)

        def clip_loss_fn(z_):
            # 1. UNet forward & get denoised pred_x0
            # Ensure all inputs to UNet are in half precision
            # Temporarily cast UNet to half precision for forward pass
            unet_output = model.unet.half()(z_.half(), t.half(), encoder_hidden_states=model.get_text_embeddings(args.prompt).to(device).half())
            # Cast UNet output to float for subsequent calculations
            unet_output = unet_output.float()
            x_prev_0, pred_x0 = model.step(unet_output, t, z_)
            pred_image = 1 / 0.18215 * pred_x0.to(dtype=z_.dtype)
            # Temporarily cast VAE to half precision for decode pass
            pred_image = model.vae.half().decode(pred_image.half())['sample'].to(device).float() # Cast result to float
            pred_image = (pred_image / 2 + 0.5).clamp(0, 1)

            # 2. CLIP preprocess
            clip_normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            )
            aug_img = image_aug(pred_image).add(1).div(2)
            clip_img = clip_normalize(aug_img).to(device)
            image_features = clip_model.encode_image(clip_img)
            text_features = clip_model.encode_text(drag_prompt)
            x = F.normalize(image_features, dim=-1)
            y = F.normalize(text_features, dim=-1)
            clip_loss = 1 - (x @ y.t()).squeeze()
            return clip_loss

        # Use autograd.functional.jacobian to get the Jacobian wrt z
        # Note: If memory is an issue, you can use torch.autograd.grad with a random vector direction
        clip_jacobian = torch.autograd.functional.jacobian(clip_loss_fn, z)
        # clip_jacobian shape: [1, 4, H, W] (same as z)
        # Normalize if desired
        clip_direction = clip_jacobian / (clip_jacobian.norm() + 1e-8)
        return clip_direction


    def get_locoedit_semantic_direction(invert_code, t, model, args, clip_model, n_jacobian_samples=10, n_null=5, jacobian_eps=1e-3, eps=1e-8):
        """
        Returns LOCOEdit-style semantic direction for the CLIP loss (w.r.t. the latent code),
        with nullspace projection as in LOCOEdit.
        """
        print("calculate LOCOEdit-style semantic direction (with nullspace projection)")
        device = model.device
        z = invert_code.detach().to(device).float().requires_grad_(True)
        t = t.to(device)
        image_aug = ImageAugmentations(224, 1).to(device)
        drag_prompt = clip.tokenize(args.drag_prompt).to(device)

        def clip_loss_fn(z_):
            # 1. UNet forward & get denoised pred_x0
            # Ensure all inputs to UNet are in half precision
            # Temporarily cast UNet to half precision for forward pass
            unet_output = model.unet.half()(z_.half(), t.half(), encoder_hidden_states=model.get_text_embeddings(args.prompt).to(device).half())
            # Cast UNet output to float for subsequent calculations
            unet_output = unet_output.float()
            x_prev_0, pred_x0 = model.step(unet_output, t, z_)
            pred_image = 1 / 0.18215 * pred_x0.to(dtype=z_.dtype)
            # Temporarily cast VAE to half precision for decode pass
            pred_image = model.vae.half().decode(pred_image.half())['sample'].to(device).float() # Cast result to float
            pred_image = (pred_image / 2 + 0.5).clamp(0, 1)

            # 2. CLIP preprocess
            clip_normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            )
            aug_img = image_aug(pred_image).add(1).div(2)
            clip_img = clip_normalize(aug_img).to(device)
            image_features = clip_model.encode_image(clip_img)
            text_features = clip_model.encode_text(drag_prompt)
            x = F.normalize(image_features, dim=-1)
            y = F.normalize(text_features, dim=-1)
            clip_loss = 1 - (x @ y.t()).squeeze()
            return clip_loss

        # 1. Get the main direction (CLIP gradient/Jacobian)
        clip_jacobian = torch.autograd.functional.jacobian(clip_loss_fn, z).float()
        clip_direction = clip_jacobian / (clip_jacobian.norm() + eps)

        # 2. Build the local Jacobian matrix (nullspace)
        grads = []
        z_flat = z.detach().clone().view(-1)
        for _ in range(n_jacobian_samples):
            noise = torch.randn_like(z_flat)
            noise = noise / (noise.norm() + eps)
            z_perturbed = (z_flat + jacobian_eps * noise).view(z.shape).detach().clone().requires_grad_(True)
            loss = clip_loss_fn(z_perturbed)
            grad = torch.autograd.grad(loss, z_perturbed)[0].detach().view(-1).float()
            grads.append(grad)
        jacobian_matrix = torch.stack(grads, dim=0)  # [n_jacobian_samples, D]

        # 3. SVD for nullspace basis
        U, S, Vh = torch.linalg.svd(jacobian_matrix, full_matrices=False)
        null_basis = Vh[:n_null, :].float()  # [n_null, D]

        # 4. Nullspace projection of the direction
        original_shape = clip_direction.shape
        direction_flat = clip_direction.view(-1).float()
        # Project out the nullspace components
        # Ensure null_basis and direction_flat are float for the first matmul
        proj_coeffs = torch.matmul(null_basis.float(), direction_flat.float()).float()    # [n_null]
        # Ensure proj_coeffs and null_basis are float for the second matmul
        null_proj = torch.matmul(proj_coeffs.float(), null_basis.float())         # [D]
        # Ensure direction_flat and null_proj are float for subtraction
        direction_proj = direction_flat.float() - null_proj.float()
        # Re-normalize
        direction_proj = direction_proj / (direction_proj.norm() + eps)
        direction_proj = direction_proj.view(original_shape)
        return direction_proj
    
    if args.drag_prompt != "":
        if use_naive:
            grad_global = cal_global_grad(invert_code, text_embeddings, t, model, args)  # shape: (1, 4, H, W) 
        elif use_jacobian:
            print(f"run clip loss jacobian")
            clip_semantic_direction = get_clip_semantic_direction(invert_code, t, model, args, clip_model)
        elif use_locoedit:
            print(f"run clip loss locoedit")
            grad_global = cal_global_grad(invert_code, text_embeddings, t, model, args)  # shape: (1, 4, H, W) 
            clip_semantic_direction = get_locoedit_semantic_direction(invert_code, t, model, args, clip_model, n_jacobian_samples=10, n_null=5, jacobian_eps=1e-3, eps=1e-8)


    index_1 = torch.nonzero(mask_cp_handle) 
    O,R = get_circle(mask_cp_handle)   
    move_vectors = []       
    move_vectors_radio = [] 
    for point_i in range(len(handle_points)):
        A = handle_points[point_i].to(device=mask_cp_handle.device)/4    # y,x
        B = target_points[point_i].to(device=mask_cp_handle.device)/4
        shift_yx = B-A
        OA =  A-O 
        d_OA = torch.norm(OA)  # &
        for j, index in enumerate(tqdm(index_1, desc="get factor")):
            C = index[-2:]      # y,x
            scale_factor = get_scale_factor(C, A, OA, d_OA, R, O)
            move_vector = scale_factor*shift_yx
            if len(move_vectors)<=j:
                move_vectors.append([move_vector,])
                move_vectors_radio.append([1/(torch.norm(C-A)+0.0001),])
            else:
                move_vectors[j].append(move_vector)
                move_vectors_radio[j].append(1/(torch.norm(C-A)+0.0001))
    
    for j, index in enumerate(index_1):
        move_vectors[j] = torch.cat([ts.unsqueeze(0) for ts in move_vectors[j]], dim=0)
        move_vectors_radio[j] = torch.cat([ts.unsqueeze(0) for ts in move_vectors_radio[j]], dim=0)
    move_mode = "not recover"
    point_new_l = []
    point_new_l_value = {}
    graph_points = []
    graph_vectors = []
    heatmap_value = np.zeros((64,64))+20
    heatmap_value_target = np.zeros((64,64))+20
    flag = 0
    for j, index in enumerate(tqdm(index_1, desc="drag stretch")):
        C = index[-2:]
        radio_factor = move_vectors_radio[j]/move_vectors_radio[j].sum()
        move_vector = (radio_factor*move_vectors[j].T).T.sum(dim=0)

        if grad_global is not None:
            # C = [y, x], index[-2:] from mask
            y, x = int(C[0].item()), int(C[1].item())

            # Project CLIP gradient vector into 2D motion space (dy, dx)
            clip_vec = grad_global[0, :, y, x]  # vector of shape [4] (latent dim)

            # Simple projection into spatial motion: e.g. use 2D mean/std or linear layer
            dy = clip_vec.mean().item()
            dx = clip_vec.std().item()
            clip_motion_vector = torch.tensor([dy, dx], device=move_vector.device)

            # Fuse via cosine-aware weighting like CLIPDrag
            cos_sim = torch.nn.functional.cosine_similarity(
                move_vector.unsqueeze(0), clip_motion_vector.unsqueeze(0)
            )
            alpha = args.fuse_coef  # similar to pro_lambda

            # print(f"move vector before fusion: {move_vector}  clip_motion_vector: {clip_motion_vector}  cos_sim: {cos_sim} alpha: {alpha}")

            if cos_sim >= 0:
                move_vector = move_vector + alpha * (1 - cos_sim ** 2).sqrt() * clip_motion_vector
            else:
                move_vector = move_vector + alpha * cos_sim * clip_motion_vector

            # print(f"move vector after fusion: {move_vector}  clip_motion_vector: {clip_motion_vector}  cos_sim: {cos_sim} alpha: {alpha}")

        # print(f"move_vector: {move_vector} C: {C} radio_factor: {radio_factor}  move_vectors_radio[j]: {move_vectors_radio[j]}")

        if clip_semantic_direction is not None:
            y, x = int(C[0].item()), int(C[1].item())

            clip_vec = clip_semantic_direction[0, :, y, x]  # [4] per spatial location
            # Project this to 2D if desired (same way as your old code)
            dy = clip_vec.mean().item()
            dx = clip_vec.std().item()
            clip_motion_vector = torch.tensor([dy, dx], device=move_vector.device)

            # Fuse via cosine-aware weighting like CLIPDrag
            cos_sim = torch.nn.functional.cosine_similarity(
                move_vector.unsqueeze(0), clip_motion_vector.unsqueeze(0)
            )
            alpha = args.fuse_coef  # similar to pro_lambda

            # print(f"move vector before fusion: {move_vector}  clip_motion_vector: {clip_motion_vector}  cos_sim: {cos_sim} alpha: {alpha}")

            if cos_sim >= 0:
                move_vector = move_vector + alpha * (1 - cos_sim ** 2).sqrt() * clip_motion_vector
            else:
                move_vector = move_vector + alpha * cos_sim * clip_motion_vector

        point_new = torch.round(C+move_vector)

        # for draw arrow chart
        if flag%20 == 0:
            try:
                graph_points.append([int(torch.round(C[1])),int(torch.round(C[0]))])
                graph_vectors.append([int(torch.round(move_vector[1])),int(torch.round(move_vector[0]))])
                flag+=1
            except Exception as e:
                print(f"has a err: {e}")
                print(f"move_vector: {move_vector} C: {C} radio_factor: {radio_factor}  move_vectors_radio[j]: {move_vectors_radio[j]}")
        else:
            flag+=1
        
        try:
            point_tuple = (int(torch.round(point_new[0])),int(torch.round(point_new[1])))
            point_tuple = judge_edge(point_tuple,invert_code_d)
        except Exception as e:
            print(f"has a err: {e}")
            print(f"point_new: {point_new} C: {C} move_vector: {move_vector}")
            

        if move_mode == "not recover": # not recover point which has been cover
            if point_tuple in point_new_l:
                continue    
            point_new_l.append(point_tuple)
            invert_code_d[:,:,point_tuple[0],point_tuple[1]] = invert_code[:,:,int(torch.round(C[0])),int(torch.round(C[1]))]
        elif move_mode == "mean when recover": # when point is recovered, set the mean value to this point
            move_value = invert_code[:,:,int(torch.round(C[0])),int(torch.round(C[1]))]
            if point_tuple not in point_new_l_value.keys(): 
                point_new_l_value[point_tuple] = [move_value]
            else:
                point_new_l_value[point_tuple].append(move_value)

            if point_tuple in point_new_l:
                invert_code_d[:,:,point_tuple[0],point_tuple[1]] = sum(point_new_l_value[point_tuple])/len(point_new_l_value[point_tuple])
                continue
            point_new_l.append(point_tuple)
            invert_code_d[:,:,point_tuple[0],point_tuple[1]] = invert_code[:,:,int(torch.round(C[0])),int(torch.round(C[1]))]  

    # visualize_all_vectors(graph_points, plot_move_vector, plot_clip_motion_vector, save_path="all_vectors_visualization.png")

    draw_arrow_graph_with_bg(
        points=torch.tensor(graph_points),
        vectors=torch.tensor(graph_vectors)
    )

    if fill_mode == "interpolation":
        #invert_code_d = interpolation(invert_code_d)
        print(f"invert_code_d shape: {invert_code_d.shape}")
        print(f"invert_code_d dtype: {invert_code_d.dtype}")
        invert_code_d = interpolation_mean_adjusted(invert_code_d) ##### mean adjusted interpolation, remember to uncomment this
        #invert_code_d = interpolation(invert_code_d) ##### BNNI
        print(f"invert_code_d shape: {invert_code_d.shape}")
        print(f"invert_code_d dtype: {invert_code_d.dtype}")
        #print(f"grad_global shape: {grad_global.shape}")s

        # visualize_grad_global(grad_global, invert_code_d)

    return invert_code_d, grad_global
