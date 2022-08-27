   #### Specify all the paths here #####
   
test_imgs = "/input/"
path_to_checkpoints="m_unet4.pth.tar"
path_SAVE="/output/"

# test_imgs = r"C:\My_Data\sateg0\task_1_both_data\task1_2d\valid\img"
# path_SAVE =  r"C:\My_Data\sateg0\multi_stage"
# path_to_checkpoints=chek_path + "/"+"m_unet4.pth.tar"

# path_img=r"C:\My_Data\sateg0\task_1_both_data/"

        #### Set Hyperparameters ####
        #######################################

batch_size=1

#### Import All libraies used for training  #####

from tqdm import tqdm
import torch
import os 
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import torch.optim as optim
#from Early_Stopping import EarlyStopping

  ### Data_Generators ########
  
from read_data1 import Data_Loader
test_loader=Data_Loader(test_imgs,batch_size)

print(len(test_loader))

   ### LOAD MODELS #####
#######################################
from models import m_unet4
model=m_unet4()

#model=m_unet6()

def Evaluate_model(loader, model, device=DEVICE):
    loop = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for batch_idx, (img1,img2,label,org_dim) in enumerate(loop):
            img1 = img1.to(device=DEVICE,dtype=torch.float)
            img2 = img2.to(device=DEVICE,dtype=torch.float)
            
            p1 = model(img1,img2)  
            p1 = (p1 > 0.5) * 1
            
            filename = os.path.join(path_SAVE+'/'+str(batch_idx))
            np.save(filename, p1)
                     
def eval_():
    model.to(device=DEVICE,dtype=torch.float)
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999),lr=0)
    checkpoint = torch.load(path_to_checkpoints,map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    Evaluate_model(test_loader,model, device=DEVICE)

if __name__ == "__main__":
    eval_()
    
    