import cv2
import numpy as np
import glob
import os
import sys
import pathlib
from pathlib import Path
from PIL import Image


maskfiles1 =[]
mask_path = '/path/to/Celeba_HQ_masks'
for path in Path(f"{mask_path}").glob('*.png'):
    fname =path.stem
    if 'l_brow' in str(fname): #[l_brow, l_eye, l_ear, lower_lip]
        maskfiles1.append(str(path))
print(len(maskfiles1))

maskfiles2 =[]
for path in Path(f"{mask_path}").glob('*.png'):
    fname =path.stem
    if 'r_brow' in str(fname): #[r_brow, r_eye, r_ear, upper_lip]
        maskfiles2.append(str(path))
print(len(maskfiles2))

for mask1 in maskfiles1:
        fname1= pathlib.PurePath(str(mask1))
        pathnm1 = fname1.stem
        split1= pathnm1.split('_')
        mask1_img = cv2.resize(cv2.imread(f"{mask1}"),(1024,1024))
        for mask2 in maskfiles2:
            fname2= pathlib.PurePath(str(mask2))
            pathnm2 = fname2.stem
            split2= pathnm2.split('_')
            if int(split1[0]) == int(split2[0]):
                mask2_img = cv2.resize(cv2.imread(f"{mask2}"),(1024,1024))
                # make list
                masks = [mask1_img, mask2_img]
                # add masks
                h, w, c = mask1_img.shape
                resultmask = np.full((h,w,c), (0,0,0), dtype=np.uint8)
                for mask in masks:
                    resultmask = cv2.add(resultmask, mask)
                cv2.imwrite(os.path.join(mask_path, f'{split1[0]}_botheyebrows.png'), resultmask)


