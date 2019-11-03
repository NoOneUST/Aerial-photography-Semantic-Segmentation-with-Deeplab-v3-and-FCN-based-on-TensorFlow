from PIL import Image
import numpy as np
import os
import pandas as pd

filePath='/home/cqiuac/Tensorflow_Models/models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VisionTask2/SegmentationClassRaw/'

A=np.zeros((7,),dtype=np.int)
C=np.zeros((7,))
for fileName in os.listdir(filePath):
    im=Image.open(filePath+fileName)
    im=np.array(im)
    B=np.bincount(im.flatten())
    for j in range(len(B)):
        A[j]+=B[j]
for i in range(len(A)):
    C[i]=A[i]/sum(A)
    C[i]=1/C[i]
print(C)