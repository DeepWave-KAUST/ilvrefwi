import sys
sys.path.append("/home/taufikmh/KAUST/summer_2023/diffefwi/src/diffefwi/")

from efwi import *
from diffusion import *
from plots import *
from utils import *

from scipy.ndimage import gaussian_filter
from argparse import ArgumentParser

plt.style.use("~/science.mplstyle")

if __name__ == "__main__":

    parser = ArgumentParser(description="Without smoothing and scaled data")
    parser.add_argument(
        "--loss_type",
        type=str,
        default='l2',
        help="Type of objective function.",
    )
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default='adam',
        help="Type of optimization algorithm.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=10.,
        help="Learning rate for non-LBFGS algorithm.",
    )

    args = parser.parse_args()
    dict_args = vars(args)
    print(dict_args)

    import wandb
    from torch.utils.tensorboard import SummaryWriter

    # Change these lines for the wandb setup
    # wandb.init(project='DiffusionEFWI-01-Inversion')
    # wandb.run.log_code(".")
    wandb_dir = './wandb/latest-run/files/' #wandb.run.dir
    # wandb.config.update(args)

    # Set parameters
    freq = 10
    dx = 25.0
    dt = 0.002 # 1ms
    nt = int(6 / dt) # 4s
    num_dims = 2
    num_shots = 10 #259
    num_sources_per_shot = 1
    num_receivers_per_shot = 399 #256
    source_spacing = 399 #30.0
    receiver_spacing = 10.0
    device = torch.device('cuda')

    vp_true = torch.from_numpy(np.fromfile('/home/taufikmh/temp_files/ARID/Arid model/Arid_vp_LR', np.float32).reshape(400,400,600)[:,200,20:200].T).to(device)#/1000
    vs_true = torch.from_numpy(np.fromfile('/home/taufikmh/temp_files/ARID/Arid model/Arid_vs_LR', np.float32).reshape(400,400,600)[:,200,20:200].T).to(device)#/1000
    rho_true = torch.from_numpy(np.fromfile('/home/taufikmh/temp_files/ARID/Arid model/Arid_rho_LR', np.float32).reshape(400,400,600)[:,200,20:200].T).to(device)*1000
    print(vp_true.shape)

    # Shift the value range
    vp_true = denormalize_vp(normalize_vp(vp_true, vmin=vp_true.min(), vmax=vp_true.max()))
    vs_true = denormalize_vs(normalize_vs(vs_true, vmin=vs_true.min(), vmax=vs_true.max()))
    rho_true = denormalize_rho(normalize_rho(rho_true, vmin=rho_true.min(), vmax=rho_true.max()))
    
    # Mask water column
    mask = torch.ones_like(vp_true).to(device)
    # mask[:21,:] = 0

    # Smoothed initial model
    vp_init = torch.from_numpy(gaussian_filter(vp_true.detach().cpu().numpy(), sigma=[20,30])).to(device)
    # vs_init = torch.from_numpy(gaussian_filter(vs_true.detach().cpu().numpy(), sigma=[20,30])).to(device)
    # rho_init = torch.from_numpy(gaussian_filter(rho_true.detach().cpu().numpy(), sigma=[20,30])).to(device)
    vs_init = vp_to_vs(vp_init)
    rho_init = vp_to_rho(vp_init)*1000

    # FWI parameters
    vp = vp_init.clone().requires_grad_().to(device)
    vs = vs_init.clone().requires_grad_().to(device)
    rho = rho_init.clone().requires_grad_().to(device)

    source_locations = torch.zeros(num_shots, num_sources_per_shot, num_dims)
    source_locations[:, 0, 1] = torch.arange(num_shots).float() * source_spacing
    source_locations[:, 0, 0] += dx
    source_locations = source_locations/10

    source_amplitudes_true = (
        deepwave.wavelets.ricker(freq, nt, dt, 1/freq)
        .repeat(num_shots, num_sources_per_shot, 1)
        .to(device)
    )
    source_amplitudes_init = (
        deepwave.wavelets.ricker(freq, nt, dt, 1/freq)
        .repeat(num_shots, num_sources_per_shot, 1)
        .to(device)
    )
    source_amplitudes = source_amplitudes_init.clone()
    source_amplitudes = source_amplitudes.to(device)
    
    receiver_locations = torch.zeros(num_shots, num_receivers_per_shot, num_dims)
    receiver_locations[:, :, 0] += dx
    receiver_locations[0, :, 1] = torch.arange(num_receivers_per_shot).float() * receiver_spacing
    receiver_locations[:, :, 1] = receiver_locations[0, :, 1].repeat(num_shots, 1)
    receiver_locations = receiver_locations/10
    
    # Propagate
    out = deepwave.elastic(
        *deepwave.common.vpvsrho_to_lambmubuoyancy(vp_true, vs_true, rho_true),
        dx, dt, source_amplitudes_y=source_amplitudes.to(device),
        source_locations_y=source_locations.to(device),
        receiver_locations_y=receiver_locations.to(device),
        accuracy=4,
        pml_freq=freq,
        # pml_width=[0, 50, 50, 50]
    )[-2]
    print(source_amplitudes.shape, source_locations.shape, receiver_locations.shape)
    receiver_amplitudes_true = out

    model_unet = UNetModel(
        in_channels=3,
        model_channels=128,
        out_channels=3,
        channel_mult=(1, 2, 4, 8, 16),
        num_res_blocks=3
    )
    model_unet.to(device)
    model_unet.load_state_dict(torch.load('/home/taufikmh/KAUST/summer_2023/diffefwi/saves/pretraining/combinedsmall-25616-model-large-ddpm_14.pt'))
    timesteps=1000    
    ddpm = DenoisingDiffusionProbabilisticModel(timesteps)

    # Run Diffusion FWI
    vp_diff,vs_diff,rho_diff,loss_fwi_diffusionFWI = ddpm.efwi_sample(
        model_unet, (1, 3, 64, 64),
        900, 
        normalize_vp(vp.clone().detach()).unsqueeze(0).unsqueeze(0),
        normalize_vs(vs.clone().detach()).unsqueeze(0).unsqueeze(0),
        normalize_rho(rho.clone().detach()).unsqueeze(0).unsqueeze(0),
        10, num_shots, dx, dt,
        receiver_locations, receiver_amplitudes_true,
        source_locations, source_amplitudes, 
        freq=freq, data_weight=1.0,
        loss_type='l2', learning_rate=2, optim='adam',save_dir=wandb_dir,
        maxs=[5000,2887,2607], mins=[3000,1732,2294]
    )

    results = {
    'vp_init'   :vp_init.detach().cpu().numpy(),
    'vs_init'   :vs_init.detach().cpu().numpy(),
    'rho_init'  :rho_init.detach().cpu().numpy(),
    'vp_inve'   :vp_diff,
    'vs_inve'   :vs_diff,
    'rho_inve'  :rho_diff,
    'loss'      :loss_fwi_diffusionFWI,
    }
    torch.save(results,wandb_dir+'/result.tz')