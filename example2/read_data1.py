import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A


NUM_WORKERS=0
PIN_MEMORY=True


transform2 = A.Compose([
    A.Resize(width=320, height=320)
])
 
class Dataset_(Dataset):
    def __init__(self, image_dir,transform2=transform2):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)

        self.transform2 = transform2

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])

        image = np.load(img_path,allow_pickle=True, fix_imports=True)
    
        org_dim=image.shape[0]
        
        mean=np.mean(image)
        std=np.std(image)
        image=(image-mean)/std
        
        
        if org_dim==576:
          temp=np.zeros([640,640])
          
          temp[32:608, 32:608] = image
          image=temp
          
        if org_dim==480:
         temp=np.zeros([640,640])
         temp[80:560, 80:560] = image
         image=temp
    
          
        if org_dim==864:
         temp=np.zeros([640,640])
         temp[:,:] = image[112:752,112:752]
         image=temp
         
        
        if org_dim==784:
         temp=np.zeros([640,640])
         temp[:,:] = image[72:712,72:712]
         image=temp
         
         
        if org_dim==768:
         temp=np.zeros([640,640])
         temp[:,:] = image[64:704,64:704]
         image=temp
         
        if self.transform2 is not None:
            augmentations2 = self.transform2(image=image)
            image2 = augmentations2["image"]

            image=np.expand_dims(image, axis=0)
            image2=np.expand_dims(image2, axis=0)
          
        return image,image2,self.images[index][:-4],org_dim
    
def Data_Loader( test_dir,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    
    test_ids = Dataset_( image_dir=test_dir)

    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    
    return data_loader


# batch_size=5
# val_imgs=r'C:\My_Data\sateg0\task_1_both_data\task1_2d\valid\img'
# val_masks=r'C:\My_Data\sateg0\task_1_both_data\task1_2d\valid\seg_gt'
# val_loader=Data_Loader(val_imgs,batch_size)
# a=iter(val_loader)
# a1=next(a)
# o=a1[1].numpy()