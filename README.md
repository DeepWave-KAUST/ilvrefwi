![LOGO](https://github.com/DeepWave-KAUST/ilvrefwi/blob/main/asset/logo.png)

Reproducible material for  **Wavenumber-aware diffusion sampling to regularize multi-parameter elastic full waveform inversion**

# Project structure
This repository is organized as follows:

* :open_file_folder: **asset**: folder containing logo.
* :open_file_folder: **data**: a folder containing the subsampled velocity models used to train the diffusion model.
* :open_file_folder: **saves**: a folder containing the trained diffusion model's weight, `diffusion.pt`. *It needs to be downloaded from the Restricted Area above*.
* :open_file_folder: **scripts**: a set of Python scripts used to run diffusion training, diffusion with ILVR sampling, and EFWI.
* :open_file_folder: **src**: a folder containing routines for the `ilvrefwi` source file.

## Getting started :space_invader: :robot:
To ensure the reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

To install the environment, run the following command:
```
./install_env.sh
```
It will take some time, but if, in the end, you see the word `Done!` on your terminal, you are ready to go. 

Remember to always activate the environment by typing:
```
conda activate ilvrefwi
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) Silver 4316 CPU @ 2.30GHz equipped with a single NVIDIA A100 GPU. Different environment configurations may be required for different combinations of workstation and GPU.

