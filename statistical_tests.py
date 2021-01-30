

''' 
Python script for computing statistical stuff over deep learning model 
predictions on medical imaging. Here we deal with cardiac MRI from the 
ACDC17 dataset, a segmentation dataset. 
Use bootstrapping (ie sample with replacement) on test dataset to 
compute N sample means, and work out estimate of pop mean
which we can compare with different models/results for statistical testing.

Bootstrapping technique: 
resampling method. used to estimate confidence intervals and do 
statistical testing but resampling (ie sampling with replacement) using test dataset.

Process:
- decide how many resamples I want  to do (e.g. repeat N=100 times) 
- decide how large the sample size is for each resampling (e.g. S=64)
- idea: this means we can send 64 images through model, and repeat this 100 times. 
- each time we get a set of prediction scores, compute mean over S and .append to list
- once we reach n=N, we'll have an array of mean prediction scores -> sample mean.
- we can then compute statistics to get estimated population mean.

Usage:
- in main(), initialse as eg bootstrap = Bootstrapper(root) with the root to test dataset.
- then call it: mean, CI = boostrap(model, test_dataset) to return mean and CI of bootstrapped samples.

Started: 29.1.21. 
Updates: 

@author: calmac

'''

import numpy as np
from numpy import default_rng
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image
import settings
args = settings.parse_arguments()


class Bootstrapper(data.Dataset):

	def __init__(self, root):

		# format data as list
		self.file_list = glob.glob(root) 

	# extract data from file lists and format properly for model
	def _processData(self, samples):
		combine = np.load(samples)

		# deal with images first
		imgs = combine[0, :, :]
		imgs = Image.fromarray(np.int16(imgs))
		test_imgs = _transformImage(imgs) # perform multiple ops to format

		# now for the labels
		labels = combine[1, :, :]
		labels = Image.fromarray(np.uint8(labels))
		labels = labels.resize((args.input_resize,args.input_resize))
		test_lbls = torch.from_numpy(np.array(labels)).float()

		return test_imgs, test_lbls

	# normalise data prior to sending through model
	def _transformImage(self, imgs):
		imgs = imgs.resize((args.input_resize,args.input_resize))
		imgs = self._convert_to_RGB(imgs)
		normalise = transforms.ToTensor() # call function for normalising 
		return normalise(imgs) # convert to tensor and normalise

	# convert shape from [h,w,1] to [h,w,3]
	def _convert_to_RGB(self,imgs):
		return np.stack((imgs,imgs,imgs), axis=2) # new shape= (H,W,3)

	def __call__(self, model, test_dataset): # images and labels ready for sending through modell
		rng = default_rng()
		# Draw some random sample with replacement, compute mean perf, and append to meanDice
		meanDice = [] 
		sims = 1000
		sample_size = 64
		for i in range(sims):
			# randomly select 30 row ids in the range 0 to 375 (ie len(testset))
		    sample_ids = rng.choice(len(test_dataset), sample_size, replace=True) 
		    # use these ids and extract test images at those rows 
		    samples = self.file_list[sample_ids]
		    # get data ready for sending through model
		    imgs, lbls = self._processData(samples)
		    # send images through model and get scores
		    preds = model(imgs)
		    # compute stats: return avg. dice scores across all samples 
		    class_scores = utils.segmentation_stats(preds, labels, n_classes=args.n_classes)
		    # compute mean over classes
		    meanScores = np.mean(class_scores)
		    # append to list. 
		    meanDice.append(meanScores) 
		    # repeat n times.
		    
		# Calculate bootstrapped mean and 95% confidence interval.
		boot_mean = np.mean(meanDice)
		boot_95_ci = np.percentile(meanDice, [2.5, 97.5])
		print("Bootstrapped Mean Length = {}, 95% CI = {}".format(boot_mean, boot_95_ci))

		return boot_mean, boot_95_ci

