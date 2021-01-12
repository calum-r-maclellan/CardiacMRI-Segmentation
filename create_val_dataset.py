#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 20:25:56 2020

@author: calmac
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
import settings 

def assign_val_data(args):
  file_list = glob.glob(args.root_trainData)
  train_val_dict = split_train_val(file_list, args.val_percent)
  return train_val_dict

def split_train_val(dataset, val_percent):
  length = len(dataset)
  n = int(length * val_percent)
  return {'train': dataset[:-n], 'val': dataset[-n:]} # train = all data minus number of validation examples
                                                        # val   = the remaining number of examples

if __name__ == '__main__':
  args = settings.parse_arguments()
#  os.makedirs(args.root_myTrainData, exist_ok=True)
#  os.makedirs(args.root_valData, exist_ok=True)
  train_val_dict = assign_val_data(args)
  train_list = train_val_dict['train']
  val_list = train_val_dict['val']
  
  
  
