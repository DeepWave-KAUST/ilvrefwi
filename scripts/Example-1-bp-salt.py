import wandb

from ilvrefwi.fwi import *
from ilvrefwi.diffusion import *
from ilvrefwi.plots import *
from ilvrefwi.utils import *

from scipy.ndimage import gaussian_filter
from argparse import ArgumentParser

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.regression import MeanSquaredError

import torch

plt.style.use("~/science.mplstyle")

if __name__ == "__main__":
    
    print('#GPUs: ',torch.cuda.device_count())
    print('#GPU: ',torch.cuda.current_device())

    parser = ArgumentParser(description="Diffusion EFWI training.")
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
        "--ilvr_sampling",
        type=str,
        default='n',
        help="Type of optimization algorithm.",
    )
    parser.add_argument(
        "--description",
        type=str,
        default='Diffusion EFWI training.',
        help="Project desciprtion.",
    )
    parser.add_argument(
        "--density_scaler",
        type=float,
        default=1.,
        help="Scaling for meidum density.",
    )
    parser.add_argument(
        "--ilvr_weight",
        type=float,
        default=0.05,
        help="Scaling for meidum density.",
    )
    parser.add_argument(
        "--data_weight",
        type=float,
        default=1.,
        help="Weighting for the observed data.",
    )
    parser.add_argument(
        "--galat",
        type=float,
        default=50.,
        help="Elastic moduli galat bounds.",
    )
    parser.add_argument(
        "--gaussian_window",
        type=float,
        default=1.,
        help="Window size for Gaussian gradient smoothing.",
    )
    parser.add_argument(
        "--fwi_iteration",
        type=int,
        default=5,
        help="Number of FWI iterations.",
    )
    parser.add_argument(
        "--num_particles",
        type=int,
        default=1,
        help="Number of FWI samples.",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.,
        help="Gradient clipping scaler.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=20,
        help="Learning rate.",
    )
    parser.add_argument(
        "--vp_max",
        type=float,
        default=2000.,
        help="Maximum compressional velocity in m/s.",
    )
    parser.add_argument(
        "--regularization",
        type=str,
        default=None,
        help="Regularization type.",
    )
    parser.add_argument(
        "--cnn",
        type=str,
        default=None,
        help="Regularization type.",
    )
    parser.add_argument(
        "--regularization_weight",
        type=float,
        default=1e-11,
        help="Regularization weight.",
    )
    parser.add_argument(
        "--Ns", 
        type=int, 
        nargs='+'
    )
    parser.add_argument(
        "--diffusion",
        type=str,
        default='n',
        help="Regularization type.",
    )

    args = parser.parse_args()
    dict_args = vars(args)
    print(args.description)
    print(dict_args)
    
    # # Change these lines for the wandb setup
    # wandb.init(project='ILVREFWI-01-Inversion')
    # wandb.run.log_code(".")
    # wandb_dir = wandb.run.dir
    # wandb.config.update(args)

    use_wandb=False
    wandb_dir = './'

    # Set seed
    set_seed(5637)
    
    # Set parameters
    freq = 8
    dx = 12.5
    dt = 0.002 # 1ms
    nt = int(6 / dt) # 4s
    num_dims = 2
    num_shots = 10 #259
    num_sources_per_shot = 1
    num_receivers_per_shot = 199 #256
    source_spacing = 399 #30.0
    receiver_spacing = 20.0
    device = torch.device('cuda')

    vp_true = torch.from_numpy(np.fromfile('../data/vp', np.float32)).to(device)
    vs_true = vp_to_vs(vp_true)
    rho_true = torch.from_numpy(np.fromfile('../data/rho', np.float32)).to(device)*1000

    # Shift the value range
    vp_true = denormalize_vp(normalize_vp(vp_true, vmin=vp_true.min(), vmax=vp_true.max()))
    vs_true = denormalize_vs(normalize_vs(vs_true, vmin=vs_true.min(), vmax=vs_true.max()))
    rho_true = denormalize_rho(normalize_rho(rho_true, vmin=rho_true.min(), vmax=rho_true.max()))
    
    print(vp_true.shape)
    
    # Mask water column
    mask = torch.ones_like(vp_true).to(device)
    # Shift the value range
    vp_true = denormalize_vp(normalize_vp(vp_true, vmin=vp_true.min(), vmax=vp_true.max()))
    vs_true = denormalize_vs(normalize_vs(vs_true, vmin=vs_true.min(), vmax=vs_true.max()))
    rho_true = denormalize_rho(normalize_rho(rho_true, vmin=rho_true.min(), vmax=rho_true.max()))

    # Smoothed initial model
    vp_init = torch.from_numpy(gaussian_filter(vp_true.detach().cpu().numpy(), sigma=[10,30])).to(device)
    vs_init = torch.from_numpy(gaussian_filter(vs_true.detach().cpu().numpy(), sigma=[10,30])).to(device)
    rho_init = torch.from_numpy(gaussian_filter(rho_true.detach().cpu().numpy(), sigma=[10,30])).to(device)

    # FWI parameters
    vp = vp_init.clone().requires_grad_().to(device)
    vs = vs_init.clone().requires_grad_().to(device)
    rho = rho_init.clone().requires_grad_().to(device)
    
    source_locations = torch.zeros(num_shots, num_sources_per_shot, num_dims)
    source_locations[:, 0, 1] = torch.arange(num_shots).float() * source_spacing
    source_locations[:, 0, 0] += dx
    receiver_locations = torch.zeros(num_shots, num_receivers_per_shot, num_dims)
    receiver_locations[:, :, 0] += dx
    receiver_locations[0, :, 1] = torch.arange(num_receivers_per_shot).float() * receiver_spacing
    receiver_locations[:, :, 1] = receiver_locations[0, :, 1].repeat(num_shots, 1)

    source_locations = source_locations/10
    receiver_locations = receiver_locations/10

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

    # Propagate
    out = deepwave.elastic(
        *deepwave.common.vpvsrho_to_lambmubuoyancy(vp_true, vs_true, rho_true),
        [dx/2, dx], dt, source_amplitudes_y=source_amplitudes.to(device),
        source_locations_y=source_locations.to(device),
        receiver_locations_y=receiver_locations.to(device),
        accuracy=4,
        pml_freq=freq,
    )[-2]
    print(source_amplitudes.shape, source_locations.shape, receiver_locations.shape)
    receiver_amplitudes_true = out

    model = UNetModel(
        in_channels=3,
        model_channels=128,
        out_channels=3,
        channel_mult=(1, 2, 4, 8, 16),
        num_res_blocks=3
    )
    model.load_state_dict(torch.load('../saves/diffusion.pt'))
    model.to(device)
    timesteps=1000
    ddpm = DenoisingDiffusionProbabilisticModel(timesteps)
    
    fwi_dict = {
        'source_locations':source_locations, 'receiver_locations': receiver_locations, 'dx': dx, 'dt': dt, 
        'num_shots': num_shots, 'source_amplitudes': source_amplitudes, 
        'receiver_amplitudes_true': receiver_amplitudes_true+2e-9*torch.randn_like(receiver_amplitudes_true)
    }

    start_timesteps=900
    
    # Subsampling factors in a reverse order
    Ns = args.Ns # [32,16,8,4]
    # Ns = [4]

    # Run Diffusion FWI
    vp_diff,vs_diff,rho_diff,loss_fwi_diffusionFWI = ddpm.fwi_sample(
        model, (1, 3, 256, 256),
        start_timesteps, 
        normalize_vp(
            vp.clone().detach(), vmax=vp_true.max(), vmin=vp_true.min()
        ).unsqueeze(0).unsqueeze(0),
        normalize_vs(
            vs.clone().detach(), vmax=vs_true.max(), vmin=vs_true.min()             
        ).unsqueeze(0).unsqueeze(0),
        normalize_rho(
            rho.clone().detach(), vmax=rho_true.max(), vmin=rho_true.min()  
        ).unsqueeze(0).unsqueeze(0),
        args.fwi_iteration, num_shots, dx/2, dx, dt,
        receiver_locations, receiver_amplitudes_true.float().to(device),
        source_locations, source_amplitudes, 
        freq=freq, data_weight=[2e8, 2e8],
        loss_type=args.loss_type, learning_rate=args.learning_rate, optim='adam',save_dir=wandb_dir,
        maxs=[vp_true.max(),vs_true.max(),rho_true.max()], 
        mins=[vp_true.min(),vs_true.min(),rho_true.min()],
        diffusion=args.diffusion, num_batches=1, use_wandb=use_wandb,
        ilvr_sampling=args.ilvr_sampling, num_particles=args.num_particles, 
        down_n=list(np.repeat(Ns, int((timesteps-start_timesteps)/len(Ns)))), ilvr_weight=args.ilvr_weight,
        filter=3
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
    
    # Metrics
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    psnr = PeakSignalNoiseRatio()
    mse = MeanSquaredError()
    
    print(mse(vp_true.detach().cpu().reshape(-1), denormalize_vp(torch.from_numpy(vp_diff[-1]), vmax=vp_true.max().cpu(), vmin=vp_true.min().cpu()).reshape(-1)))
    print(mse(vs_true.detach().cpu().reshape(-1), denormalize_vs(torch.from_numpy(vs_diff[-1]), vmax=vs_true.max().cpu(), vmin=vs_true.min().cpu()).reshape(-1)))
    print(mse(rho_true.detach().cpu().reshape(-1), denormalize_rho(torch.from_numpy(rho_diff[-1]), vmax=rho_true.max().cpu(), vmin=rho_true.min().cpu()).reshape(-1)))
    
    if use_wandb:
        wandb.log({"vp_mse": mse(vp_true.detach().cpu().reshape(-1), denormalize_vp(torch.from_numpy(vp_diff[-1]), vmax=vp_true.max().cpu(), vmin=vp_true.min().cpu()).reshape(-1))})
        wandb.log({"vs_mse": mse(vs_true.detach().cpu().reshape(-1), denormalize_vs(torch.from_numpy(vs_diff[-1]), vmax=vs_true.max().cpu(), vmin=vs_true.min().cpu()).reshape(-1))})
        wandb.log({"rho_mse": mse(rho_true.detach().cpu().reshape(-1), denormalize_rho(torch.from_numpy(rho_diff[-1]), vmax=rho_true.max().cpu(), vmin=rho_true.min().cpu()).reshape(-1))})
