import gc
import os
from tqdm import tqdm
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.utils as tvu

import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, einsum
from diffusers.utils import pt_to_pil
# from utils.utils import (
#     get_stable_diffusion_model,
#     get_stable_diffusion_scheduler,
#     extract,
# )
from copy import deepcopy
######################
# LDM ; use diffuser #
######################
from diffusers import (
    # DDIMInverseScheduler,
    DDIMScheduler, 
)

# from modules.mask_segmentation import SAM


class EditStableDiffusion(object):
    def __init__(self, args):
        # default setting
        self.seed = args.seed
        # self.pca_device     = args.pca_device
        # self.buffer_device  = args.buffer_device
        # self.memory_bound   = args.memory_bound


        # # get model
        # self.pipe = get_stable_diffusion_model(args)
        # self.vae  = self.pipe.vae
        # self.unet = self.pipe.unet
        # self.sam = SAM(args, log_dir = self.result_folder)

        # self.dtype  = args.dtype
        # self.device = self.pipe._execution_device

        # # args (diffusion schedule)
        # self.scheduler = get_stable_diffusion_scheduler(args, self.pipe.scheduler)
        # self.for_steps = args.for_steps
        # self.inv_steps = args.inv_steps
        # self.use_yh_custom_scheduler = args.use_yh_custom_scheduler
        
        # # args (guidance)
        # self.guidance_scale     = args.guidance_scale
        # self.guidance_scale_edit= args.guidance_scale_edit
        

        # # args (h space edit)        
        # self.edit_prompt        = args.edit_prompt 
        # self.edit_prompt_emb    = self._get_prompt_emb(args.edit_prompt)

        # # x-space guidance
        # self.x_edit_step_size = args.x_edit_step_size
        # self.x_space_guidance_edit_step         = args.x_space_guidance_edit_step
        # self.x_space_guidance_scale             = args.x_space_guidance_scale
        # self.x_space_guidance_num_step          = args.x_space_guidance_num_step
        # self.x_space_guidance_use_edit_prompt   = args.x_space_guidance_use_edit_prompt


        # self.scheduler.set_timesteps(self.for_steps, device=self.device)
        # self.edit_t         = args.edit_t
        # self.edit_t_idx     = (self.scheduler.timesteps - self.edit_t * 1000).abs().argmin()
        # self.sampling_mode  = args.sampling_mode
        # self.use_sega = args.use_sega
        # self.tilda_v_score_type = args.tilda_v_score_type


    @torch.no_grad()
    def run_DDIMforward(self, num_samples=5):
        print('start DDIMforward')
        self.EXP_NAME = f'DDIMforward-for_{self.for_prompt}'

        # get latent code
        zT = torch.randn(num_samples, 4, 64, 64).to(device=self.device, dtype=self.dtype)

        # simple DDIMforward
        self.DDIMforwardsteps(zT, t_start_idx=0, t_end_idx=-1)

    @torch.no_grad()
    def run_DDIMinversion(self, idx, guidance=None, vis_traj=False):
        '''
        Prompt
            (CFG)       pos : inv_prompt, neg : null_prompt
            (no CFG)    pos : inv_prompt
        '''
        print('start DDIMinversion')
        self.EXP_NAME = f'DDIMinversion-{self.dataset_name}-{idx}-for_{self.for_prompt}-inv_{self.inv_prompt}'

        # inversion scheduler

        # before start
        num_inference_steps = self.inv_steps
        do_classifier_free_guidance = (self.guidance_scale > 1.0) & (guidance is not None)

        # set timestep (we do not use default scheduler set timestep method)
        if self.use_yh_custom_scheduler:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device, is_inversion=True)
        else:
            raise ValueError('recommend to use yh custom scheduler')
            self.scheduler = DDIMInverseScheduler.from_config(self.scheduler.config)
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # get image
        x0 = self.dataset[idx]
        tvu.save_image((x0 / 2 + 0.5).clamp(0, 1), os.path.join(self.result_folder, f'original_x0.png'))

        # get latent
        z0 = self.vae.encode(x0).latent_dist
        z0 = z0.sample()
        z0 = z0 * 0.18215

        ##################
        # denoising loop #
        ##################
        latents = z0
        for i, t in enumerate(timesteps):
            if i == len(timesteps) - 1:
                break

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            if do_classifier_free_guidance:
                prompt_emb = torch.cat([self.null_prompt_emb.repeat(latents.size(0), 1, 1), self.inv_prompt_emb.repeat(latents.size(0), 1, 1)], dim=0)
            else:
                prompt_emb = self.inv_prompt_emb.repeat(latents.size(0), 1, 1)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t,
                encoder_hidden_states=prompt_emb,
                # cross_attention_kwargs=cross_attention_kwargs,
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, eta=0).prev_sample

        return latents


    def _classifer_free_guidance(self, latents, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode, do_classifier_free_guidance):
        assert mode in ["null+(for-null)+(edit-null)","null+(for-null)","null+(edit-null)","(for-edit)"]
        if do_classifier_free_guidance:
            if mode == "null+(for-null)":
                latent_model_input = torch.cat([latents] * 2, dim=0)
                prompt_emb = torch.cat([for_prompt_emb.repeat(latents.size(0), 1, 1), null_prompt_emb.repeat(latents.size(0), 1, 1)], dim=0)
            elif mode == "null+(for-null)+(edit-null)":
                latent_model_input = torch.cat([latents] * 3, dim=0)
                prompt_emb = torch.cat([for_prompt_emb.repeat(latents.size(0), 1, 1), edit_prompt_emb.repeat(latents.size(0), 1, 1), null_prompt_emb.repeat(latents.size(0), 1, 1)], dim=0)
            elif mode == "null+(edit-null)":
                latent_model_input = torch.cat([latents] * 2, dim=0)
                prompt_emb = torch.cat([edit_prompt_emb.repeat(latents.size(0), 1, 1), null_prompt_emb.repeat(latents.size(0), 1, 1)], dim=0)    
            elif mode == "(for-edit)":
                latent_model_input = torch.cat([latents] * 2, dim=0)
                prompt_emb = torch.cat([for_prompt_emb.repeat(latents.size(0), 1, 1), edit_prompt_emb.repeat(latents.size(0), 1, 1)], dim=0)                               
        else:
            latent_model_input = latents
            prompt_emb = for_prompt_emb.repeat(latents.size(0), 1, 1)
    
        noise_pred = self.unet(
            latent_model_input, t,
            encoder_hidden_states=prompt_emb,
        ).sample

        # perform guidance
        if do_classifier_free_guidance:
            if mode == "null+(for-null)+(edit-null)":
                noise_pred_for, noise_pred_edit, noise_pred_uncond = noise_pred.chunk(3)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_for - noise_pred_uncond) + self.guidance_scale_edit * (noise_pred_edit - noise_pred_uncond)
            elif mode == "null+(for-null)":
                noise_pred_for, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_for - noise_pred_uncond)
            elif mode == "null+(edit-null)":
                noise_pred_edit, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_edit - noise_pred_uncond)
            elif mode == "(for-edit)":
                noise_pred_for, noise_pred_edit = noise_pred.chunk(2)
                noise_pred = self.guidance_scale * (noise_pred_for - noise_pred_edit)                        
        return noise_pred

    @torch.no_grad()
    def DDIMforwardsteps(
            self, zt, t_start_idx, t_end_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode = "null+(for-null)", **kwargs
        ):
        '''
        Prompt
            (CFG)       pos : for_prompt, neg : neg_prompt
            (no CFG)    pos : for_prompt
        '''
        assert mode in ["null+(for-null)+(edit-null)","null+(for-null)","null+(edit-null)","edit-proj[for](edit)","null+for+edit-proj[for](edit)"]
        print('start DDIMforward')
        # before start
        num_inference_steps = self.for_steps
        do_classifier_free_guidance = self.guidance_scale > 1.0
        # cross_attention_kwargs      = None
        memory_bound = self.memory_bound // 2 if do_classifier_free_guidance else self.memory_bound
        print(memory_bound)
        print('do_classifier_free_guidance : ', do_classifier_free_guidance)

        # set timestep (we do not use default scheduler set timestep method)
        if self.use_yh_custom_scheduler:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        else:
            self.scheduler = DDIMScheduler.from_config(self.scheduler.config)
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # save traj
        latents = zt

        #############################################
        # denoising loop (t_start_idx -> t_end_idx) #
        #############################################
        for t_idx, t in enumerate(self.scheduler.timesteps):
            # skip
            if (t_idx < t_start_idx): 
                continue
                
            # start sampling
            elif t_start_idx == t_idx:
                # print('t_start_idx : ', t_idx)
                pass

            # end sampling
            elif t_idx == t_end_idx:
                # print('t_end_idx : ', t_idx)
                return latents, t, t_idx

            # split zt to avoid OOM
            latents = latents.to(device=self.buffer_device)
            if latents.size(0) == 1:
                latents_buffer = [latents]
            else:
                latents_buffer = list(latents.chunk(latents.size(0) // memory_bound))

            # loop over buffer
            for buffer_idx, latents in enumerate(latents_buffer):
                # overload to device
                latents = latents.to(device=self.device)

                noise_pred = self._classifer_free_guidance(latents, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode = mode, do_classifier_free_guidance = do_classifier_free_guidance)
                # print("device check:", t.device)
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, eta=0).prev_sample

                # save latents in buffer
                latents_buffer[buffer_idx] = latents.to(self.buffer_device)

            latents = torch.cat(latents_buffer, dim=0)
            latents = latents.to(device=self.device)
            del latents_buffer
            torch.cuda.empty_cache()

        # decode with vae
        latents = 1 / 0.18215 * latents
        x0 = self.vae.decode(latents).sample
        x0 = (x0 / 2 + 0.5).clamp(0, 1)
        tvu.save_image(x0, os.path.join(self.result_folder, f'{self.EXP_NAME}.png'), nrow = x0.size(0))
        x0 = (x0 * 255).to(torch.uint8).permute(0, 2, 3, 1)
        return latents, x0


    def get_x0(self, zt, t, t_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mask = None, mode = "null+(for-null)+(edit-null)", flatten=False):
        assert mode in ["null+(for-null)+(edit-null)","null+(for-null)","null+(edit-null)","(for-edit)"]
        
        do_classifier_free_guidance = self.guidance_scale > 1.0

        noise_pred = self._classifer_free_guidance(zt, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode = mode, do_classifier_free_guidance = do_classifier_free_guidance)

        at = extract(self.scheduler.alphas_cumprod, t, zt.shape)

        z0_hat = (zt - noise_pred * (1 - at).sqrt()) / at.sqrt()

        # decode
        z0_hat = 1 / 0.18215 * z0_hat
        x0_hat = self.vae.decode(z0_hat).sample

        if mask is not None:
            x0_hat = x0_hat[:, mask]
            return x0_hat

        # only use this for single xt and no mask:
        if flatten:
            # print(xt.shape, x0_hat.shape)
            c_i, w_i, h_i = x0_hat.size(1), x0_hat.size(2), x0_hat.size(3)
            x0_hat = x0_hat.view(-1, c_i*w_i*h_i)
        return x0_hat

    @torch.no_grad()
    def get_delta_zt_via_grad(self, zt, t, t_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mask = None, mode = "null+(for-null)+(edit-null)"):
        # assert mode in ["null+(for-null)+(edit-null)","null+(for-null)","null+(edit-null)","(for-edit)"]
        #import pdb; pdb.set_trace()
        do_classifier_free_guidance = self.guidance_scale > 1.0

        ### noise from original prompt
        noise_pred = self._classifer_free_guidance(zt, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode = "null+(for-null)", do_classifier_free_guidance = do_classifier_free_guidance)
        at = extract(self.scheduler.alphas_cumprod, t, zt.shape)
        z0_hat = (zt - noise_pred * (1 - at).sqrt()) / at.sqrt()
        # decode
        z0_hat = 1 / 0.18215 * z0_hat
        x0_hat = self.vae.decode(z0_hat).sample

        ### noise from edit prompt
        noise_pred_after= self._classifer_free_guidance(zt, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode = mode, do_classifier_free_guidance = do_classifier_free_guidance)
        at = extract(self.scheduler.alphas_cumprod, t, zt.shape)
        z0_hat_after = (zt - noise_pred_after * (1 - at).sqrt()) / at.sqrt()
        # decode
        z0_hat_after = 1 / 0.18215 * z0_hat_after
        x0_hat_after = self.vae.decode(z0_hat_after).sample

        ### semantic difference between original and edit prompt
        x0_hat_delta = x0_hat_after - x0_hat
        c_i, w_i, h_i = zt.size(1), zt.size(2), zt.size(3)

        # we can even add mask here
        if mask is not None:
            x0_hat_delta_flat = x0_hat_delta[:,mask]
            # print(x0_hat_delta_flat.shape)
        else:
            x0_hat_delta_flat = x0_hat_delta.view(-1,c_i*w_i*h_i)

        ### text-supervised editing direction within the mask, g: v
        g = lambda v : torch.sum(x0_hat_delta_flat * self.get_x0(v, t, t_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mask = mask, mode = mode, flatten=True))
        v_ = torch.autograd.functional.jacobian(g, zt)

        ### normalize the editing direction, v_: Vp
        v_ = v_.view(-1, c_i*w_i*h_i)
        v_ = v_ / v_.norm(dim=1, keepdim=True)

        zt_delta = v_.view(-1, c_i, w_i, h_i)
        zt_new = 10.0 * zt_delta + zt
        noise_pred_new = self._classifer_free_guidance(zt_new, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode = "null+(for-null)", do_classifier_free_guidance = do_classifier_free_guidance)
        z0_hat_new = (zt_new - noise_pred_new * (1 - at).sqrt()) / at.sqrt()
        # decode
        z0_hat_new = 1 / 0.18215 * z0_hat_new
        x0_hat_new = self.vae.decode(z0_hat_new).sample
        # tvu.save_image((x0_hat_new / 2 + 0.5).clamp(0, 1), os.path.join(self.result_folder, f'viagrad_x0_hat.png'))
        
        return v_

    def local_encoder_decoder_pullback_zt(
            self, zt, t, t_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, op=None, block_idx=None,
            pca_rank=50, chunk_size=25, min_iter=10, max_iter=100, convergence_threshold=1e-3, mask = None,
            mode = "null+(for-null)+(edit-null)"
        ):
        '''
        Args
            - zt : zt
            - op : ['down', 'mid', 'up']
            - block_idx : op == down, up : [0,1,2,3], op == mid : [0]
            - pooling : ['pixel-sum', 'channel-sum', 'single-channel', 'multiple-channel']
        Returns
            - h : hidden feature
        '''
        assert mode in ["null+(for-null)+(edit-null)","null+(for-null)","null+(edit-null)","(for-edit)","edit-proj[for](edit)","null+for+edit-proj[for](edit)"]
        num_chunk = pca_rank // chunk_size if pca_rank % chunk_size == 0 else pca_rank // chunk_size + 1

        # get h samples
        time_s = time.time()

        c_i, w_i, h_i = zt.size(1), zt.size(2), zt.size(3)
        if mask is None:
            c_o, w_o, h_o = c_i, w_i, h_i # output shape of x^0
        else:
            l_o = mask.sum().item()


        a = torch.tensor(0., device=zt.device, dtype=zt.dtype)

        # Algorithm 1
        vT = torch.randn(c_i*w_i*h_i, pca_rank, device=zt.device, dtype=torch.float)
        vT, _ = torch.linalg.qr(vT)
        v = vT.T
        v = v.view(-1, c_i, w_i, h_i)


        time_s = time.time()
        # Jacobian subspace iteration
        for i in range(max_iter):
            v = v.to(device=zt.device, dtype=zt.dtype)
            v_prev = v.detach().cpu().clone()
            
            u = []
            v_buffer = list(v.chunk(num_chunk))
            for vi in v_buffer:
                g = lambda a : self.get_x0(zt + a*vi, t, t_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mask = mask, mode=mode)
                
                ui = torch.func.jacfwd(g, argnums=0, has_aux=False, randomness='error')(a) # ui = J@vi
                u.append(ui.detach().cpu().clone())
            u = torch.cat(u, dim=0)
            u = u.to(zt.device, zt.dtype)

            if mask is None:
                g = lambda zt : einsum(
                    u, self.get_x0(zt, t, t_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mask=mask, mode=mode), 'b c w h, i c w h -> b'
                )
            else:
                g = lambda zt : einsum(
                    u, self.get_x0(zt, t, t_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mask=mask, mode=mode), 'b l, i l -> b'
                )                
            
            v_ = torch.autograd.functional.jacobian(g, zt) # vi = ui.T@J
            v_ = v_.view(-1, c_i*w_i*h_i)

            _, s, v = torch.linalg.svd(v_, full_matrices=False)
            v = v.view(-1, c_i, w_i, h_i)
            if mask is None:
                u = u.view(-1, c_o, w_o, h_o)
            else:
                u = u.view(-1, l_o)
            
            convergence = torch.dist(v_prev, v.detach().cpu()).item()
            print(f'power method : {i}-th step convergence : ', convergence)
            
            if torch.allclose(v_prev, v.detach().cpu(), atol=convergence_threshold) and (i > min_iter):
                print('reach convergence threshold : ', convergence)
                break

        time_e = time.time()
        print('power method runtime ==', time_e - time_s)

        if mask is None:
            u, s, vT = u.reshape(-1, c_o*w_o*h_o).T.detach(), s.sqrt().detach(), v.reshape(-1, c_i*w_i*h_i).detach()
        else:
            u, s, vT = u.reshape(-1, l_o).T.detach(), s.sqrt().detach(), v.reshape(-1, c_i*w_i*h_i).detach()
        return u, s, vT

    @torch.no_grad()
    def run_edit_null_space_projection_zt_semantic(
            self, op, block_idx, vis_num, mask_index = 0, vis_num_pc=1, vis_vT=False, pca_rank=50, edit_prompt=None, null_space_projection = False, pca_rank_null=50, 
        ):
        print(f'current experiment : op : {op}, block_idx : {block_idx}, vis_num : {vis_num}, vis_num_pc : {vis_num_pc}, pca_rank : {pca_rank}, edit_prompt : {edit_prompt}, null_space_projection = {null_space_projection}, pca_rank_null={pca_rank_null}')
        '''
        1. z0 -> zT -> zt -> z0 ; we edit latent variable zt
        2. get local basis of h-space (u) and x-space (v) by using the power method
        3. edit sample with x-space guidance
        '''
        #import pdb; pdb.set_trace()

        # set edit prompt
        if edit_prompt is not None:
            self.edit_prompt = edit_prompt
            self.edit_prompt_emb = self._get_prompt_emb(self.edit_prompt)

        # set edit_t
        self.scheduler.set_timesteps(self.for_steps)

        # get latent code (zT -> zt)
        if self.dataset_name == 'Random':
            zT = torch.randn(1, 4, 64, 64, dtype=self.dtype, device=self.device)
        
        self.EXP_NAME = "original"
        if (not os.path.exists(os.path.join(self.result_folder, "original.png"))) or (not os.path.exists(os.path.join(self.result_folder, "mask/mask.pt"))):
            print("Generating images and creating masks......")
            _, x0 = self.DDIMforwardsteps(zT, t_start_idx=0, t_end_idx=-1, 
                                          for_prompt_emb=self.for_prompt_emb, 
                                          edit_prompt_emb=self.edit_prompt_emb, 
                                          null_prompt_emb=self.null_prompt_emb,
                                          mode="null+(for-null)")
            masks = self.sam.mask_segmentation(Image.fromarray(np.array(x0[0].detach().cpu())), resolution=512)

        else:
            print("Loading masks......")
            masks = torch.load(os.path.join(self.result_folder, "mask/mask.pt"))
        
        if self.sampling_mode:
            return None
        mask = masks[mask_index].squeeze(dim=0).repeat(3, 1, 1)
        
        zt, t, t_idx = self.DDIMforwardsteps(zT, t_start_idx=0, t_end_idx=self.edit_t_idx, 
                                            for_prompt_emb=self.for_prompt_emb, 
                                            edit_prompt_emb=self.edit_prompt_emb, 
                                            null_prompt_emb=self.null_prompt_emb,
                                            mode="null+(for-null)")
        assert t_idx == self.edit_t_idx



        # get local basis
        if not self.use_sega:        
            save_dir = os.path.join(self.result_folder, "basis", f'local_basis-{self.edit_t}T-"{self.edit_prompt}"-pca-rank-{pca_rank}-select-mask{mask_index}')
            os.makedirs(save_dir, exist_ok=True)
            u_modify_path = os.path.join(save_dir, f'u-modify.pt')
            vT_modify_path = os.path.join(save_dir, f'vT-modify.pt')
            u_null_path = os.path.join(save_dir, f'u-null-null_space_rank_{pca_rank_null}.pt')
            vT_null_path = os.path.join(save_dir, f'vT-null-null_space_rank_{pca_rank_null}.pt')        
            # load pre-computed local basis
            if os.path.exists(u_modify_path) and os.path.exists(vT_modify_path) and os.path.exists(u_null_path) and os.path.exists(vT_null_path):
                u_modify = torch.load(u_modify_path, map_location=self.device).type(self.dtype)
                vT_modify = torch.load(vT_modify_path, map_location=self.device).type(self.dtype)
                u_null = torch.load(u_null_path, map_location=self.device).type(self.dtype)
                vT_null = torch.load(vT_null_path, map_location=self.device).type(self.dtype)

            else:
                print('!!!RUN LOCAL PULLBACK!!!')
                zt = zt.to(device=self.device, dtype=self.dtype)

                vT_modify = self.get_delta_zt_via_grad(zt, t, t_idx, self.for_prompt_emb, self.edit_prompt_emb, self.null_prompt_emb, mask = mask, mode = self.tilda_v_score_type)

                torch.save(vT_modify, vT_modify_path)

                if null_space_projection:
                    u_null, s_null, vT_null = self.local_encoder_decoder_pullback_zt(
                    zt, t, t_idx, self.for_prompt_emb, self.edit_prompt_emb, self.null_prompt_emb, op=op, block_idx=block_idx,
                    pca_rank=pca_rank_null, chunk_size=5, min_iter=10, max_iter=50, convergence_threshold=1e-3, mask = ~mask, mode="null+(for-null)",
                    )
                    
                    torch.save(u_null, u_null_path)
                    torch.save(vT_null, vT_null_path)

            # normalize u, vT
            if not null_space_projection:
                vT = vT_modify / vT_modify.norm(dim=1, keepdim=True)
            else:
                vT_null = vT_null[:pca_rank_null, :]
                vT = (vT_null.T @ (vT_null @ vT_modify.T)).T
                vT = vT_modify - vT
                vT = vT / vT.norm(dim=1, keepdim=True)

            original_zt = zt.clone()
            for pc_idx in range(vis_num_pc):
                zts = {
                    -1: None,
                    1: None,
                }
                self.EXP_NAME = f'Edit_zt-edit_{self.edit_t}T-{op}-block_{block_idx}-pc_{pc_idx:0=3d}_pos-edit_prompt-{self.edit_prompt}_select_mask{mask_index}_null_space_projection_{null_space_projection}_null_space_rank_{pca_rank_null}_{self.tilda_v_score_type}'      
                for direction in [1, -1]:
                    vk = direction*vT[pc_idx, :].view(-1, *zT.shape[1:])
                    # edit zt along vk direction with **x-space guidance**
                    zt_list = [original_zt.clone()]
                    for _ in tqdm(range(self.x_space_guidance_num_step), desc='x_space_guidance edit'):
                        zt_edit = self.x_space_guidance_direct(
                            zt_list[-1], t_idx=self.edit_t_idx, vk=vk, 
                            single_edit_step=self.x_space_guidance_edit_step,
                        )
                        zt_list.append(zt_edit)
                    zt = torch.cat(zt_list, dim=0)
                    if vis_num == 1:
                        zt = zt[[0,-1],:]
                    else:
                        zt = zt[::(zt.size(0) // vis_num)]
                    zts[direction] = zt
                    # zt -> z0
                zt = torch.cat([(zts[-1].flip(dims=[0]))[:-1], zts[1]], dim=0)

            self.DDIMforwardsteps(
                zt, t_start_idx=self.edit_t_idx, t_end_idx=-1, 
                for_prompt_emb=self.for_prompt_emb, 
                edit_prompt_emb=self.edit_prompt_emb, 
                null_prompt_emb=self.null_prompt_emb,
                mode="null+(for-null)")
        else:
            self.EXP_NAME = f'sega-edit_prompt-{self.edit_prompt}'
            self.DDIMforwardsteps(
                zt, t_start_idx=self.edit_t_idx, t_end_idx=-1, 
                for_prompt_emb=self.for_prompt_emb, 
                edit_prompt_emb=self.edit_prompt_emb, 
                null_prompt_emb=self.null_prompt_emb,
                mode="null+(for-null)+(edit-null)")


    @torch.no_grad()
    def x_space_guidance_direct(self, zt, t_idx, vk, single_edit_step):
        # necesary parameters
        t = self.scheduler.timesteps[t_idx]

        # edit xt with vk
        zt_edit = zt + self.x_space_guidance_scale * single_edit_step * vk

        return zt_edit
    
    

    # utils
    def _get_prompt_emb(self, prompt):
        prompt_embeds = self.pipe.encode_prompt(
            prompt,
            device = self.device,
            num_images_per_prompt = 1,
            do_classifier_free_guidance = False,
        )[0]
        return prompt_embeds
    

    @torch.no_grad()
    def get_semantic_direction(
            self, op, block_idx, vis_num, mask_index = 0, vis_num_pc=1, vis_vT=False, pca_rank=50, edit_prompt=None, null_space_projection = False, pca_rank_null=50, mask = None
        ):
        print(f'current experiment : op : {op}, block_idx : {block_idx}, vis_num : {vis_num}, vis_num_pc : {vis_num_pc}, pca_rank : {pca_rank}, edit_prompt : {edit_prompt}, null_space_projection = {null_space_projection}, pca_rank_null={pca_rank_null}')
        '''
        1. z0 -> zT -> zt -> z0 ; we edit latent variable zt
        2. get local basis of h-space (u) and x-space (v) by using the power method
        3. edit sample with x-space guidance
        '''
        #import pdb; pdb.set_trace()

        # set edit prompt
        if edit_prompt is not None:
            self.edit_prompt = edit_prompt
            self.edit_prompt_emb = self._get_prompt_emb(self.edit_prompt)

        # set edit_t
        self.scheduler.set_timesteps(self.for_steps)

        # get latent code (zT -> zt)
        if self.dataset_name == 'Random':
            zT = torch.randn(1, 4, 64, 64, dtype=self.dtype, device=self.device)
        
        self.EXP_NAME = "original"
        if (not os.path.exists(os.path.join(self.result_folder, "original.png"))) or (not os.path.exists(os.path.join(self.result_folder, "mask/mask.pt"))):
            print("Generating images and creating masks......")
            _, x0 = self.DDIMforwardsteps(zT, t_start_idx=0, t_end_idx=-1, 
                                          for_prompt_emb=self.for_prompt_emb, 
                                          edit_prompt_emb=self.edit_prompt_emb, 
                                          null_prompt_emb=self.null_prompt_emb,
                                          mode="null+(for-null)")
            masks = self.sam.mask_segmentation(Image.fromarray(np.array(x0[0].detach().cpu())), resolution=512)

        else:
            print("Loading masks......")
            masks = torch.load(os.path.join(self.result_folder, "mask/mask.pt"))
        
        if self.sampling_mode:
            return None
        mask = masks[mask_index].squeeze(dim=0).repeat(3, 1, 1)
        
        zt, t, t_idx = self.DDIMforwardsteps(zT, t_start_idx=0, t_end_idx=self.edit_t_idx, 
                                            for_prompt_emb=self.for_prompt_emb, 
                                            edit_prompt_emb=self.edit_prompt_emb, 
                                            null_prompt_emb=self.null_prompt_emb,
                                            mode="null+(for-null)")
        assert t_idx == self.edit_t_idx



        # get local basis
        if not self.use_sega:        
            save_dir = os.path.join(self.result_folder, "basis", f'local_basis-{self.edit_t}T-"{self.edit_prompt}"-pca-rank-{pca_rank}-select-mask{mask_index}')
            os.makedirs(save_dir, exist_ok=True)
            u_modify_path = os.path.join(save_dir, f'u-modify.pt')
            vT_modify_path = os.path.join(save_dir, f'vT-modify.pt')
            u_null_path = os.path.join(save_dir, f'u-null-null_space_rank_{pca_rank_null}.pt')
            vT_null_path = os.path.join(save_dir, f'vT-null-null_space_rank_{pca_rank_null}.pt')        
            # load pre-computed local basis
            if os.path.exists(u_modify_path) and os.path.exists(vT_modify_path) and os.path.exists(u_null_path) and os.path.exists(vT_null_path):
                u_modify = torch.load(u_modify_path, map_location=self.device).type(self.dtype)
                vT_modify = torch.load(vT_modify_path, map_location=self.device).type(self.dtype)
                u_null = torch.load(u_null_path, map_location=self.device).type(self.dtype)
                vT_null = torch.load(vT_null_path, map_location=self.device).type(self.dtype)

            else:
                print('!!!RUN LOCAL PULLBACK!!!')
                zt = zt.to(device=self.device, dtype=self.dtype)

                vT_modify = self.get_delta_zt_via_grad(zt, t, t_idx, self.for_prompt_emb, self.edit_prompt_emb, self.null_prompt_emb, mask = mask, mode = self.tilda_v_score_type)

                torch.save(vT_modify, vT_modify_path)

                if null_space_projection:
                    u_null, s_null, vT_null = self.local_encoder_decoder_pullback_zt(
                    zt, t, t_idx, self.for_prompt_emb, self.edit_prompt_emb, self.null_prompt_emb, op=op, block_idx=block_idx,
                    pca_rank=pca_rank_null, chunk_size=5, min_iter=10, max_iter=50, convergence_threshold=1e-3, mask = ~mask, mode="null+(for-null)",
                    )
                    
                    torch.save(u_null, u_null_path)
                    torch.save(vT_null, vT_null_path)

            # normalize u, vT
            if not null_space_projection:
                vT = vT_modify / vT_modify.norm(dim=1, keepdim=True)
            else:
                vT_null = vT_null[:pca_rank_null, :]
                vT = (vT_null.T @ (vT_null @ vT_modify.T)).T
                vT = vT_modify - vT
                vT = vT / vT.norm(dim=1, keepdim=True)

            original_zt = zt.clone()
            for pc_idx in range(vis_num_pc):
                zts = {
                    -1: None,
                    1: None,
                }
                self.EXP_NAME = f'Edit_zt-edit_{self.edit_t}T-{op}-block_{block_idx}-pc_{pc_idx:0=3d}_pos-edit_prompt-{self.edit_prompt}_select_mask{mask_index}_null_space_projection_{null_space_projection}_null_space_rank_{pca_rank_null}_{self.tilda_v_score_type}'      
                for direction in [1, -1]:
                    vk = direction*vT[pc_idx, :].view(-1, *zT.shape[1:])
                    # edit zt along vk direction with **x-space guidance**
                    zt_list = [original_zt.clone()]
                    for _ in tqdm(range(self.x_space_guidance_num_step), desc='x_space_guidance edit'):
                        zt_edit = self.x_space_guidance_direct(
                            zt_list[-1], t_idx=self.edit_t_idx, vk=vk, 
                            single_edit_step=self.x_space_guidance_edit_step,
                        )
                        zt_list.append(zt_edit)
                    zt = torch.cat(zt_list, dim=0)
                    if vis_num == 1:
                        zt = zt[[0,-1],:]
                    else:
                        zt = zt[::(zt.size(0) // vis_num)]
                    zts[direction] = zt
                    # zt -> z0
                zt = torch.cat([(zts[-1].flip(dims=[0]))[:-1], zts[1]], dim=0)

            self.DDIMforwardsteps(
                zt, t_start_idx=self.edit_t_idx, t_end_idx=-1, 
                for_prompt_emb=self.for_prompt_emb, 
                edit_prompt_emb=self.edit_prompt_emb, 
                null_prompt_emb=self.null_prompt_emb,
                mode="null+(for-null)")
        else:
            self.EXP_NAME = f'sega-edit_prompt-{self.edit_prompt}'
            self.DDIMforwardsteps(
                zt, t_start_idx=self.edit_t_idx, t_end_idx=-1, 
                for_prompt_emb=self.for_prompt_emb, 
                edit_prompt_emb=self.edit_prompt_emb, 
                null_prompt_emb=self.null_prompt_emb,
                mode="null+(for-null)+(edit-null)")
            
    @torch.no_grad()
    def project_clip_grad_into_semantic_subspace(
        self,
        grad_clip,         # Tensor of shape [1, C, H, W] - gradient from CLIP
        zt,
        t,
        t_idx,      # latent, timestep tensor, and step index
        for_prompt_emb,
        edit_prompt_emb,
        null_prompt_emb="",
        mask=None,         # semantic region mask
        null_space_projection=False,
        pca_rank_null=30
    ):
        """
        Project a CLIP-derived gradient into the LOCOEdit-derived semantic subspace (optionally null-space projected).
        """
        device = zt.device
        B, C, H, W = zt.shape

        # === 1. Get semantic direction from CLIP gradient via text-supervised Jacobian ===
        vT_modify = self.get_delta_zt_via_grad(
            zt=zt,
            t=t,
            t_idx=t_idx,
            for_prompt_emb=for_prompt_emb,
            edit_prompt_emb=edit_prompt_emb,
            null_prompt_emb=null_prompt_emb,
            mask=mask,
            mode="null+(for-null)+(edit-null)",
        )  # [1, C, H, W]
        vT = vT_modify.view(1, -1)
        vT = F.normalize(vT, dim=1)  # unit vector

        # === 2. Optional null space projection ===
        if null_space_projection:
            _, _, vT_null = self.local_encoder_decoder_pullback_zt(
                zt=zt,
                t=t,
                t_idx=t_idx,
                for_prompt_emb=for_prompt_emb,
                edit_prompt_emb=edit_prompt_emb,
                null_prompt_emb=null_prompt_emb,
                op=None,
                block_idx=None,
                pca_rank=pca_rank_null,
                chunk_size=5,
                min_iter=10,
                max_iter=50,
                convergence_threshold=1e-3,
                mask=~mask if mask is not None else None,
                mode="null+(for-null)"
            )  # vT_null shape: [pca_rank_null, C*H*W]
            vT_null = vT_null[:pca_rank_null, :]  # [r, CHW]

            # projection: vT ← vT - Proj_null(vT)
            vT_proj = vT - (vT_null.T @ (vT_null @ vT.T)).T
            vT = F.normalize(vT_proj, dim=1)

        # === 3. Project clip gradient ===
        grad_flat = grad_clip.view(1, -1)  # [1, CHW]
        grad_proj = vT.T @ (vT @ grad_flat.T)  # [CHW, 1]
        grad_projected = grad_proj.T.view_as(grad_clip)

        return grad_projected


    
####################
# Custom timesteps #
####################
from functools import partial
from typing import Union

def custom_set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None, inversion_flag: bool = False):
    """
    Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.
    Args:
        num_inference_steps (`int`):
            the number of diffusion steps used when generating samples with a pre-trained model.
    """

    if num_inference_steps > self.config.num_train_timesteps:
        raise ValueError(
            f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
            f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
            f" maximal {self.config.num_train_timesteps} timesteps."
        )

    self.num_inference_steps = num_inference_steps
    step_ratio = self.config.num_train_timesteps // self.num_inference_steps
    # creates integer timesteps by multiplying by ratio
    # casting to int to avoid issues when num_inference_step is power of 3
    # timesteps = (np.arange(0, num_inference_steps) * step_ratio).round().copy().astype(np.int64)
    timesteps = np.linspace(0, 1, num_inference_steps) * (self.config.num_train_timesteps-2) # T=999
    timesteps = timesteps + 1e-6
    timesteps = timesteps.round().astype(np.int64)
    # reverse timesteps except inverse diffusion
    # keep it numpy array
    if not inversion_flag:
        timesteps = np.flip(timesteps).copy()

    self.timesteps = torch.from_numpy(timesteps).to(device)
    self.timesteps += self.config.steps_offset