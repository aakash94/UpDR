import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import random
import math


class CustomDataLoader(Dataset):
    
    def __init__(self, root_dir):
        self.hdr_folder = root_dir+"/hdr/"
        self.sdr_folder = root_dir+"/sdr/"
        self.image_count = sum([len(files) for r, d, files in os.walk(self.sdr_folder)])
        self.digits_in_name = math.floor(math.log(self.image_count + 0.00001, 10))+1
        self.extention = ".png"
        self.transform = transforms.Compose([transforms.ToTensor()])

        
    def __getitem__(self, index):
        
        if index == self.image_count:
            index = random.randint(0,self.image_count)

        ind = str(index)
        ind = ind.zfill(self.digits_in_name)
        hdr_path = self.hdr_folder+ind+self.extention
        sdr_path = self.sdr_folder+ind+self.extention
        hdr_image = Image.open(hdr_path)
        sdr_image = Image.open(sdr_path)
        hdr_image = self.transform(hdr_image)
        sdr_image = self.transform(sdr_image)
        
        return hdr_image, sdr_image
        
        
    def __len__(self):
        return self.image_count
    
