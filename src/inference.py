import torch
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import model
data_path='./input/'
model_path='./model/'
image_path='./input/images/'
images=glob(image_path+"*.jpg")
images= [x.split('/')[3] for x in images]
# load the dataframe
attributes = pd.read_csv(os.path.join(data_path, "attributes.csv"))
for i in range(len(attributes)):
    if attributes['filename'][i] in images:
        pass
    else:
        attributes.drop(index=i, inplace=True)
df = attributes.reset_index(drop=True)
def infer_img(num):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    n_classes_neck = 7
    n_classes_sleeve = 4
    n_classes_pattern = 10
    model_t = model.Net(n_classes_neck, n_classes_sleeve, n_classes_pattern)
    model_t.load_state_dict(torch.load(model_path + 'model.pth'))
    model_t.to(device)
    model_t.eval();
    image = Image.open(image_path + df['filename'][num])
    img = np.asarray(image).astype(np.float32)
    img_t = torch.tensor(img,dtype=torch.float32).transpose(0,2).unsqueeze(0)
    neck,sleeve,pattern = model_t(img_t.to(device))
    neck = torch.argmax(neck.squeeze(0)).item()
    sleeve = torch.argmax(sleeve.squeeze(0)).item()
    pattern = torch.argmax(pattern.squeeze(0)).item()
    plt.imshow(image)
    plt.title('neck:' + str(df['neck'][num]) + ' sleeve_len:' + str(df['sleeve_length'][num]) + ' pat:' + str(
        df['pattern'][num])+"\n"+'pred_neck:' + str(neck) + ' pred_sleeve_len:' + str(sleeve) + ' pred_pat:' + str(pattern));
    plt.axis('off')
    plt.show()


infer_img(60)