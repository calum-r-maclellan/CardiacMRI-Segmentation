# CardiacMRI-Segmentation
PyTorch implementation of both vanilla segmentation networks (UNet and Attention UNet) and a multi-task learning (MTL) UNet. The models are trained to segment the right and left ventricles and surrounding myocardium tissue from magnetic resonance images (ACDC'17 dataset). 

This work has two key branches:
- improving the segmentation accuracy with a novel multi-task learning U-Net.
- investigating the effects of variations to the optimisation process on performance and generalisation (e.g. does a noisier SGD improve generalisation for medical imaging data, which is inherently noisy, over an adaptive gradient algorithm e.g. RMSProp, Adam? how good is the minima found by these algorithms?).

A Jupyter Notebok is available (acdc_pytorch.ipynb) to run the models on Google Colab. 
