import glob
import os
import pickle
import sys
import pathlib
from pathlib import Path

from deepface import DeepFace

startind=int(sys.argv[0])
attribute=sys.argv[1]

# Define paths to the ground truth images and generated images
### LFW original images
ground_truth_images_path = '/path/to/lfw/original images'
orifiles = sorted(glob.glob(f'{ground_truth_images_path}/*.jpg')) 
print(len(orifiles))

### LFW generated images
generated_images_path = '/path/to/LFW/{attribute}_edited_images'
genfiles = sorted(glob.glob(f'{generated_images_path}/*.png')) 
print(len(genfiles))

gen_scores=[]
imp_scores=[]

for ind, img1 in enumerate(orifiles[startind:], start=startind):
    print(ind)
    fname1= pathlib.PurePath(str(img1))
    subid1 = Path(str(fname1)).stem  # original image filename is '00001.png'

   
    for img2 in genfiles:
        fname2= pathlib.PurePath(str(img2))
        pathnm2 = Path(str(fname2)).stem
        split2 = pathnm2.split('_')
        subid2 = split2[0] # attribute edited image filename '00001_brownhair.png'
        try:
          score = DeepFace.verify(img1, img2, model_name="ArcFace", detector_backend='retinaface')['distance'] 
          if subid1 == subid2:
            gen_scores.append(score)
          else:
            imp_scores.append(score)
        except:
          continue
pickle.dump(gen_scores, open(f'LFW/{attribute}/gen_scores.pkl', 'wb+'))
pickle.dump(imp_scores, open(f'LFW/{attribute}/gen_scores.pkl', 'wb+'))

