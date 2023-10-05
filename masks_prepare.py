import os
import sys
import numpy as np
from PIL import Image
from zipfile import ZipFile
from natsort import natsorted
script_dir=os.path.abspath(os.path.dirname(sys.argv[0]))
default_path=script_dir+'/Original_Masks/'

def convert_one_channel(img):
    if len(img.shape)>2:
        img=img[:,:,0]
        return img
    else:
        return img


default_path=script_dir+'/Custom_Masks/'

def pre_splitted_masks(path=default_path):
    ZipFile(path+"/splitted_masks.zip").extractall(path+'/Masks/') 
    path=path+'/Masks/'
    dirs=natsorted(os.listdir(path))
    masks=img=Image.open(path+dirs[0])
    masks=convert_one_channel(np.asarray(masks))
    for i in range (1,len(dirs)):
        img=Image.open(path+dirs[i])
        img=convert_one_channel(np.asarray(img))
        masks=np.concatenate((masks,img))
    masks=np.reshape(masks,(len(dirs),512,512,1))
    return masks
    




    
