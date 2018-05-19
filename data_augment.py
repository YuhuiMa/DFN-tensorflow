# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="path to folder containing images to augment")
a = parser.parse_args()

filenames = os.listdir(a.dir + "/main")
for filename in filenames:
	
	if os.path.exists(a.dir + "/segmentation/" + filename):
		
		img = cv2.imread(a.dir + "/main/" + filename)
		label = cv2.imread(a.dir + "/segmentation/" + filename)
		img = cv2.flip(img, 1)
		label = cv2.flip(label, 1)
		cv2.imwrite(a.dir + "/main/aug_" + filename, img)
		cv2.imwrite(a.dir + "/segmentation/aug_" + filename, label)

print("Data augmentation has been finished.")
