import os
import pandas as pd
from glob import glob
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import model
import dataset
import engine
from utils import sample_data, confusion_matrix_

if __name__ == "__main__":
    # location of train.csv and train_jpg folder
    # with all the jpg images
    data_path = "./input/"
    # cuda/cpu device
    device=torch.device('cuda' if torch.cuda.is_available() else "cpu")
    # let's train for 10 epochs
    epochs = 10
    # load the dataframe
    attributes = pd.read_csv(os.path.join(data_path, "attributes.csv"))
    #print(attributes.head())
    images=glob(data_path+'images/'+"*.jpg")
    images= [x.split('/')[3] for x in images]
    for i in range (len(attributes)):
      if attributes['filename'][i] in images:
        pass
      else:
        attributes.drop(index=i,inplace=True)
    attributes=attributes.reset_index(drop=True)
    attributes=attributes.fillna(-1)
    #print(attributes.head())
    train_df,val_df=train_test_split(attributes,test_size=0.2,random_state=42)
    train_df=train_df.reset_index(drop=True)
    val_df=val_df.reset_index(drop=True)
    data=sample_data(train_df,n_samples=100)
    val_df=val_df[(val_df['neck']!=-1) & (val_df['sleeve_length']!=-1) &(val_df['pattern']!=-1)]
    val_df=val_df.reset_index(drop=True)


    device=torch.device('cuda' if torch.cuda.is_available() else "cpu")
    n_classes_neck = 7
    n_classes_sleeve = 4
    n_classes_pattern = 10
    model = model.Net(n_classes_neck, n_classes_sleeve, n_classes_pattern)

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # fetch the ClassificationDataset class
    cloth_dataset_train=dataset.ClothData(data_path,data)

    # torch dataloader creates batches of data
    # from classification dataset class
    train_dataloader=DataLoader(cloth_dataset_train,batch_size=16,shuffle=True)
    #same for validation dataset
    cloth_dataset_valid=dataset.ClothData(data_path,val_df)
    valid_dataloader=DataLoader(cloth_dataset_valid,batch_size=16)
    epochs= 1
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train model
    for epcoh in range(epochs):
        engine.train(train_dataloader,model,optimizer,epochs,device=device)
        results=engine.evaluate(valid_dataloader,model,device=device)
        print(confusion_matrix_(results))










