# Run EFWI with Diffusion regularization and ILVR sampling
python Example-1-bp-salt.py --fwi_iteration=10 --num_particles=1 --ilvr_sampling='y' --Ns 8 16 32 64 128

# Run EFWI without Diffusion regularization
python Example-1-bp-salt.py --fwi_iteration=10 --num_particles=1 --diffusion='n' --Ns 1 1 1

# Run EFWI with L1 regularization
python Example-1-bp-salt.py --fwi_iteration=10 --num_particles=1 --diffusion='y' --regularization=tikhonov_1st --regularization_weight=1e-13 --Ns 1 1 1

# Run EFWI with TV regularization
python Example-1-bp-salt.py --fwi_iteration=10 --num_particles=1 --diffusion='y' --regularization=tv_l1 --regularization_weight=1e-13 --Ns 1 1 1