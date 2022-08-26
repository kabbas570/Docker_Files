inputDir = "/input"
path_img="/output"

#inputDir = r"C:\My_Data\sateg0\task_1_both_data\task1_2d\valid\img"
#path_img=r"C:\My_Data\sateg0\task_1_both_data/"

import glob
import os
import numpy as np

mask_id = []
for infile in sorted(glob.glob(inputDir+'/*.npy')):
    mask_id.append(infile)
    
print(len(mask_id))

for i in range(5):
    arr=np.load(mask_id[i])
    print(arr.shape)
    filename = os.path.join(path_img+'/'+str(i+1))
    np.save(filename, arr)