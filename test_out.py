import os
import sys
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vgg19
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import torch.nn.functional as F


image_paths = [r"C:\Users\Gigabyte\Downloads\enel_645_final\testing\images\Cancer (1413).jpg", 
               r"C:\Users\Gigabyte\Downloads\enel_645_final\testing\images\Cancer (1609).jpg",
               r"C:\Users\Gigabyte\Downloads\enel_645_final\testing\images\Cancer (2406).jpg",
               r"C:\Users\Gigabyte\Downloads\enel_645_final\testing\images\Cancer (2425).jpg",
               r"C:\Users\Gigabyte\Downloads\enel_645_final\testing\images\Not Cancer  (13).jpg",
               r"C:\Users\Gigabyte\Downloads\enel_645_final\testing\images\Not Cancer  (52).jpg"]

image_step1 = []
for image_path in image_paths:
    image_step1.append(cv2.imread(image_path, cv2.IMREAD_COLOR))

image_step2 = []
for image in image_step1:
    image_step2.append(cv2.resize(image, (224, 224)))

image_tensor = []
for image in image_step2:
    image_tensor.append(transforms.ToTensor()(image).unsqueeze(0))



class_labels = ['Healthy', 'Brain Tumor']  # Replace with your actual class labels

images =[1,2,3]

a= 0
for each in images:
    output_image_path = "maps/densenet/gradcam_output" + str(a) + ".png"
    print(output_image_path)
    a = a + 1

image_paths = [
    (r"C:\Users\Gigabyte\Downloads\enel_645_final\testing\images\Cancer (1413).jpg",0), 
    (r"C:\Users\Gigabyte\Downloads\enel_645_final\testing\images\Cancer (1609).jpg",0), 
    (r"C:\Users\Gigabyte\Downloads\enel_645_final\testing\images\Cancer (2406).jpg",0), 
    (r"C:\Users\Gigabyte\Downloads\enel_645_final\testing\images\Cancer (2425).jpg",0), 
    (r"C:\Users\Gigabyte\Downloads\enel_645_final\testing\images\Not Cancer  (13).jpg",0), 
    (r"C:\Users\Gigabyte\Downloads\enel_645_final\testing\images\Not Cancer  (52).jpg",1)]

bfull = [0,0,0,0,1,1]

for image_path, idt in image_paths:
    # Load and preprocess the image
    print(image_path, idt)
    

