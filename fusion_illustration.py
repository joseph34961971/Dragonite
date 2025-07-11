import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Hyper-parameters and vectors
# -------------------------------
alpha               = 1.0          # step size
clip_motion_vector  = 1.0          # can be any scalar or 1-D component
move_vector_initial = 0.0          # starting value

# ---------------------------------
# 2. Prepare cos_sim value domain
# ---------------------------------
cos_sim = np.linspace(-1.0, 1.0, 500)

# ---------------------------------
# 3. Compute the update for both
#    branches of the conditional
# ---------------------------------
# Positive branch (cos_sim ≥ 0)
mask_pos = cos_sim >= 0
delta_pos = alpha * np.sqrt(1.0 - cos_sim[mask_pos]**2)           # α·√(1 − cos²)
move_vector_pos = move_vector_initial + delta_pos * clip_motion_vector

# Negative branch (cos_sim < 0)
mask_neg = cos_sim < 0
delta_neg = alpha * cos_sim[mask_neg]                             # α·cos_sim
move_vector_neg = move_vector_initial + delta_neg * clip_motion_vector

# -------------------------------
# 4. Plot the two cases separately
#    (one chart per figure)
# -------------------------------
# Case 1: cos_sim ≥ 0
plt.figure()
plt.plot(cos_sim[mask_pos], move_vector_pos)
plt.title(r'posetive')
plt.xlabel(r'$\cos\_{sim}$')
plt.ylabel('move_vector after update')
plt.grid(True)
plt.tight_layout()
plt.savefig('move_vector_positive.png')   # optional: save to file
plt.show()

# Case 2: cos_sim < 0
plt.figure()
plt.plot(cos_sim[mask_neg], move_vector_neg)
plt.title(r'negative')
plt.xlabel(r'$\cos\_{sim}$')
plt.ylabel('move_vector after update')
plt.grid(True)
plt.tight_layout()
plt.savefig('move_vector_negative.png')   # optional: save to file
plt.show()