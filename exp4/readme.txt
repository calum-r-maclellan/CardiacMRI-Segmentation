In this folder, I've stored the train/val results to variations of the UNet architecture. 

Settings maintained:
- lr=1e-3
- multistep lr, gamma=0.1, milestones=[50,100]
- adam optimiser with standard 1st/2nd moment weights
- batch size of 16 
- batch norm layers 
- input resize is 128x128
- 200 epochs
- lobjective: Dice loss

============
1. Unet:
- 512->512 bottleneck, with 1024->256 at 4th up-layer
- 13.4m params
- done: 8/1/21.



===========
2. Deeper Unet:
- 512->1024 bottleck, with 1536->256 at 4th up conv
- 24m params
- done: 8/1/21.








===========
3. AG-UNet
- same as above, but with an AG on each skip-connection 
- 14M params
- done: 8/1/21.



============
4. Suns model
- MTL-Unet
- UNet + EE + FB and dice + ce loss with weighting 
- 30M params

