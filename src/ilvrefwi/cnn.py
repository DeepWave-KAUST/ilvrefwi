import sys
sys.path.insert(1, '/home/taufikmh/KAUST/summer_2023/diffefwi/repos/pnp-recovery')

from iterAlgs import *
from util import *
from robjects import *
from CSClass_color_torch import CSClass

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json

import numpy as np
from tqdm import tqdm
import scipy.io as sio
from datetime import datetime
from skimage.metrics import structural_similarity as ssim

from skimage.transform import resize

def normalize_(vp, vmax=5000, vmin=3000):
    """
    Normalize compressional velocity.
    
    :param vp: a compressional velocity array.
    :param vmax: maximum vp.
    :param vmin: minimum vp.
    
    :return vp: a normalized vp [0-1].
    """
    
    return (vp - vmin)/(vmax-vmin)*2.0 - 1.0

def cnn_reconstruction(vp, vs, rho, type='pnp-ar'):

    with open('/home/taufikmh/KAUST/summer_2023/diffefwi/repos/pnp-recovery/configs/config_nature_color.json') as File:
        config = json.load(File)

    # set the random seed
    np.random.seed(128)

    # general settings
    cs_ratio = config["dataset"]['CS_ratio']
    IMG_Patch = config['dataset']['IMG_Patch']

    # Read Image path
    filepaths_test = get_image_paths(config["dataset"]['valid_datapath'])

    # Load CS Sampling Matrix: phi
    Phi_data_Name = '%s/cs_%d_color_33.mat' % (config["dataset"]['Phi_datapath'], cs_ratio)
    Phi_input = sio.loadmat(Phi_data_Name)['phi']

    # Load Initialization Matrix: Qinit
    Qinit_data_Name = '%s/Initialization_Matrix_%d_color.mat' % (config["dataset"]['Qinit_datapath'], cs_ratio)
    Qinit = sio.loadmat(Qinit_data_Name)['Qinit']

    # Convert from numpy to torch:
    Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
    Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor)
    Phi = Phi.cuda()
    Qinit = Qinit.cuda()

    #####################################
    #               Valid               # 
    #####################################

    is_improve = False
    ImgNum = len(filepaths_test)

    if is_improve:
        index_2run = []
    else:
        index_2run = range(1, ImgNum+1)

    ####################################################
    ####                   PnP/RED                   ###
    ####################################################
    print('Start Running PnP/RED!')
    # TODO: Try Nesterov's acceleration.
    
    for img_no in index_2run:
    
        imgName = filepaths_test[img_no-1]
        print("==============================================================================\n")
        print("img_no: [%d]" % img_no)
        print('imgName: ', imgName)
        print("\n==============================================================================")
        print("")
        # Img_yuv = imread_uint(imgName, n_channels=3)
        # Iorg_y = Img_yuv.astype(np.float64)/255.0
        # Iorg_y = Iorg_y.transpose(2,0,1)

        Iorg_y = np.concatenate((vp, vs, rho), 0)*0.5 + 0.5
        Iorg_y_torch = torch.from_numpy(Iorg_y).type(torch.FloatTensor).cuda()
        print(Iorg_y.shape, Iorg_y.dtype, Iorg_y_torch.shape)
        [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_torch_color(Iorg_y_torch, IMG_Patch) 
        Icol = img2col_torch_color(Ipad, IMG_Patch).transpose(1,0)
        Img_output = Icol
        print("Img_output: ", Img_output.shape)

        batch_x = Img_output
        Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1)).cuda()
        dObj = CSClass(Iorg_y_torch, batch_x, Phix, Phi, Qinit, IMG_Patch)
        
        if type=='pnp-ar':
            ####################################################
            ####                  PnP (AR)                   ###
            ####################################################
            print('Start Running PnP (AR)!')

            tau = config['pnp(AR)']['tau']
            alpha = config['pnp(AR)']['alpha']
            cs_ratio_ar = config['pnp(AR)']['cs']
            numIter = config['pnp(AR)']['numIter']
            gamma = config['pnp(AR)']['gamma_inti']

            rObj = DnCNNClass(config, model_path=config['pnp(AR)']['model_path'], cs=cs_ratio_ar, img_channel=3)
            save_path = "/home/taufikmh/KAUST/summer_2023/diffefwi/repos/pnp-recovery/results/nature_color/CS=%d/PnP-AR/img_%d_arcs=%d_sigma_0" %(cs_ratio, img_no, cs_ratio_ar)
            #-- Reconstruction --# 
            recon, out_each, psnr_each = apgmEst(dObj, rObj, numIter=numIter, step=gamma, accelerate=True, mode='PROX', useNoise=False, 
                                    is_save=True, save_path=save_path, xtrue=[Iorg_y, Iorg_y_torch], xinit='Qinti', save_iter=350, clip=True, tau=tau, alpha=alpha)
        elif type=='pnp-denoising':    
            ####################################################
            ####               PnP （denoising)              ###
            ####################################################
            print('Start Running PnP (denoising)!')

            tau = config['pnp(denoising)']['tau']
            sigma = config['pnp(denoising)']['sigma']
            alpha = config['pnp(denoising)']['alpha']
            numIter = config['pnp(denoising)']['numIter']
            gamma = config['pnp(denoising)']['gamma_inti']

            rObj = DnCNNClass(config, model_path=config['pnp(denoising)']['model_path'], sigma=sigma, img_channel=3)
            save_path = "/home/taufikmh/KAUST/summer_2023/diffefwi/repos/pnp-recovery/results/nature_color/CS=%d/PnP-denoising/img_%d_sigma_%d" %(cs_ratio, img_no, sigma)
            #-- Reconstruction --# 
            recon, out_each, psnr_each = apgmEst(dObj, rObj, numIter=numIter, step=gamma, accelerate=True, mode='PROX', useNoise=False, 
                                    is_save=True, save_path=save_path, xtrue=[Iorg_y, Iorg_y_torch], xinit='Qinti', save_iter=350, clip=True, tau=tau, alpha=alpha)
            
            
        elif type=='red-denoising':
            ###################################################
            ###               RED （denoising)              ###
            ###################################################
            print('Start Running RED (denoising)!')

            tau = config['red(denoising)']['tau']
            sigma = config['red(denoising)']['sigma']
            alpha = config['red(denoising)']['alpha']
            numIter = config['red(denoising)']['numIter']
            gamma = config['red(denoising)']['gamma_inti']

            rObj = DnCNNClass(config, model_path=config['red(denoising)']['model_path'], sigma=sigma, img_channel=3)
            save_path = "/home/taufikmh/KAUST/summer_2023/diffefwi/repos/pnp-recovery/results/nature_color/CS=%d/RED-denoising/img_%d_sigma_%d" %(cs_ratio, img_no, sigma)
            #-- Reconstruction --# 
            recon, out_each, psnr_each = apgmEst(dObj, rObj, numIter=numIter, step=gamma/(1 + 2*tau), accelerate=True, mode='RED', useNoise=False, 
                                    is_save=True, save_path=save_path, xtrue=[Iorg_y, Iorg_y_torch], xinit='Qinti', save_iter=350, clip=True, tau=tau, alpha=alpha)


    return (recon-0.5)*2


# if __name__ == "__main__":
    
#     vp = resize((np.fromfile('/home/taufikmh/temp_files/ARID/Arid model/Arid_vp_LR', np.float32).reshape(400,400,600)[:,200,:].T), (256,256))
#     vs = resize((np.fromfile('/home/taufikmh/temp_files/ARID/Arid model/Arid_vs_LR', np.float32).reshape(400,400,600)[:,200,:].T), (256,256))
#     rho = resize((np.fromfile('/home/taufikmh/temp_files/ARID/Arid model/Arid_rho_LR', np.float32).reshape(400,400,600)[:,200,:].T), (256,256))

#     vp_n = normalize_(vp, vmin=vp.min(), vmax=vp.max()).reshape(1,256,256)
#     vs_n = normalize_(vs, vmin=vs.min(), vmax=vs.max()).reshape(1,256,256)
#     rho_n = normalize_(rho, vmin=rho.min(), vmax=rho.max()).reshape(1,256,256)
    
#     recon = cnn_reconstruction(vp_n, vs_n, rho_n, 'pnp-ar')
    
#     print(recon.max())