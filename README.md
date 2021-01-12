# CardiacMRI-Segmentation
PyTorch implementation of both vanilla segmentation networks (UNet and Attention UNet) and a multi-task learning (MTL) UNet. The models are trained to segment the right and left ventricles and surrounding myocardium tissue from magnetic resonance images (ACDC'17 dataset). 

This work has two key branches:
- improving the segmentation accuracy with a novel multi-task learning U-Net (collaboration).
- investigating the effects of variations to the optimisation process on performance and generalisation: e.g. does a noisier SGD improve performance and generalisation [1] for medical imaging data, which is inherently noisy, over an adaptive gradient algorithm (e.g. RMSProp, Adam)? how good is the minima found by these algorithms, and are there mods we can make to the optimiser to better guide SGD steps [2]? how strongly do the SGD steps at the early phases of training [2,3] impact on the final minima (thus performance and generalisation?). do these effects translate over into other imaging modalities (e.g. CT, X-ray, PET, Ultrasound), and are there features inherent in medical data we can exploit to guide optimiser to better minima?

A Jupyter Notebok is available (acdc_pytorch.ipynb) to run the models on Google Colab. 

[1] S.L. Smith, E. Elsen, and S. De: On the generalisation benefit of noise is stochastic  gradient descent. ICMR 2020. 

[2] S. Jastrzebski et al: On the relation between the sharpest directions of DNN loss and the SGD step length. ICLR 2019.

[3] S. Jastrzebski et al: The break-even point on optimisation trajectories of deep neural networks. ICLR 2020. 
