# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--gt_dir", help="path to folder containing ground truth")
parser.add_argument("--pred_dir", help="path to folder containing output images of the model")
parser.add_argument("--result_txt", help="path to txt file saving results")
a = parser.parse_args()

filenames = os.listdir(a.gt_dir)
mean_iou = 0
count = 0

if os.path.exists(a.result_txt):
	
	os.remove(a.result_txt)

fd = open(a.result_txt, 'a')
for filename in filenames:
	
	if os.path.exists(a.pred_dir + "/" + filename):
		
		gt = Image.open(a.gt_dir + "/" + filename).convert("L")
		gt_map = np.array(gt, np.bool).astype(np.float32)
		gt_count = np.sum(gt_map)
		
		pred = Image.open(a.pred_dir + "/" + filename).convert("L")
		pred_map = np.array(pred, np.bool).astype(np.float32)
		pred_count = np.sum(pred_map)
		
		overlap_map = output_map * gt_map
		overlap_count = np.sum(overlap_map)
		
		iou = overlap_count / (gt_count + output_count - overlap_count)
		mean_iou += iou
		count += 1
		
		print("IOU of " + filename + " is: " + str(iou))
		fd.write("IOU of " + filename + " is: " + str(iou) + "\n")

if count == 0:
	
	print("No images exist in both " + a.gt_dir + " and " + a.pred_dir)
	fd.write("No images exist in both " + a.gt_dir + " and " + a.pred_dir)

else:
	
	mean_iou /= count
	print("mean iou: {}".format(mean_iou))
	fd.write("mean iou: {}".format(mean_iou))
