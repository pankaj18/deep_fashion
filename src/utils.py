import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report

def sample_data(df,n_samples):
  df1=df[(df['neck']==0) & (df['sleeve_length']!=3)  & (df['pattern']!=9) ].sample(n_samples,replace=True)
  df2=df[(df['neck']==1)  & (df['sleeve_length']!=3)  & (df['pattern']!=9)].sample(n_samples,replace=True)
  df3=df[(df['neck']==2) & (df['sleeve_length']!=3)  & (df['pattern']!=9)].sample(n_samples,replace=True)
  df4=df[(df['neck']==3) & (df['sleeve_length']!=3)  & (df['pattern']!=9)].sample(n_samples,replace=True)
  df5=df[(df['neck']==4)& (df['sleeve_length']!=3)  & (df['pattern']!=9)].sample(n_samples,replace=True)
  df6=df[(df['neck']==5) & (df['sleeve_length']!=3)  & (df['pattern']!=9)].sample(n_samples,replace=True)
  df7=df[(df['neck']==6) & (df['sleeve_length']!=3)  & (df['pattern']!=9)].sample(n_samples,replace=True)

  df8=df[(df['sleeve_length']==0) & (df['neck']!=6) & (df['pattern']!=9) ].sample(n_samples,replace=True)
  df9=df[(df['sleeve_length']==1) & (df['neck']!=6) & (df['pattern']!=9)].sample(n_samples,replace=True)
  df10=df[(df['sleeve_length']==2) & (df['neck']!=6) & (df['pattern']!=9)].sample(n_samples,replace=True)
  df11=df[(df['sleeve_length']==3) & (df['neck']!=6) & (df['pattern']!=9)].sample(n_samples,replace=True)

  df12=df[(df['pattern']==0) ].sample(n_samples,replace=True)
  df13=df[(df['pattern']==1) & (df['sleeve_length']!=3) & (df['neck']!=6)].sample(n_samples,replace=True)
  df14=df[(df['pattern']==2) & (df['sleeve_length']!=3) & (df['neck']!=6)].sample(n_samples,replace=True)
  df15=df[(df['pattern']==3) & (df['sleeve_length']!=3) & (df['neck']!=6)].sample(n_samples,replace=True)
  df16=df[(df['pattern']==4) & (df['sleeve_length']!=3) & (df['neck']!=6)].sample(n_samples,replace=True)
  df17=df[(df['pattern']==5) & (df['sleeve_length']!=3) & (df['neck']!=6)].sample(n_samples,replace=True)
  df18=df[(df['pattern']==6) & (df['sleeve_length']!=3) & (df['neck']!=6)].sample(n_samples,replace=True)
  df19=df[(df['pattern']==7) & (df['sleeve_length']!=3) & (df['neck']!=6)].sample(n_samples,replace=True)
  df20=df[(df['pattern']==8) & (df['sleeve_length']!=3) & (df['neck']!=6)].sample(n_samples,replace=True)
  df21=df[(df['pattern']==9) & (df['sleeve_length']!=3) & (df['neck']!=6)].sample(n_samples,replace=True)

  data=pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,
              df12,df13,df14,df15,df16,df17,df18,df19,df20,df21])
  data=data.reset_index(drop=True)
  return data

def plot_barplot(df):
  plt.figure(figsize=(12,6))
  plt.subplot(311)
  plt.bar(df['neck'].value_counts().index,df['neck'].value_counts())
  plt.subplot(312)
  plt.bar(df['sleeve_length'].value_counts().index,df['sleeve_length'].value_counts())
  plt.subplot(313)
  plt.bar(df['pattern'].value_counts().index,df['pattern'].value_counts())


def plot_img(df, num):
    try:
        img = Image.open(path + 'images/' + df['filename'][num])
        plt.imshow(img)
        plt.title('neck:' + str(df['neck'][num]) + ' sleeve_len:' + str(df['sleeve_length'][num]) + ' pat:' + str(
            df['pattern'][num]));

        plt.axis('off')
    except IOError as e:
        pass

def get_weights(df):
  samples_neck=df['neck'].value_counts().drop(index=-1.0).sort_values()
  weights_neck=(1-samples_neck/samples_neck.sum()).to_numpy()
  weights_neck_tensor=torch.tensor(weights_neck,dtype=torch.float32).to(device)
  #weights_neck_tensor
  samples_sleeves=df['sleeve_length'].value_counts().drop(index=-1.0).sort_values()
  weights_sleeves=(1-samples_sleeves/samples_sleeves.sum()).to_numpy()
  weights_sleeves_tensor=torch.tensor(weights_sleeves,dtype=torch.float32).to(device)
  #weights_sleeves_tensor
  samples_pattern=df['pattern'].value_counts().drop(index=-1.0).sort_values()
  weights_pattern=(1-samples_pattern/samples_pattern.sum()).to_numpy()
  weights_pattern_tensor=torch.tensor(weights_pattern,dtype=torch.float32).to(device)
  #weights_pattern_tensor
  return  weights_neck_tensor,weights_sleeves_tensor,weights_pattern_tensor



def confusion_matrix_(results):
    true_labels_neck=results[0]
    pred_labels_neck=results[1]
    true_labels_sleeves=results[2]
    pred_labels_sleeves=results[3]
    true_labels_pattern=results[4]
    pred_labels_pattern=results[5]
    print("Neck")
    print(confusion_matrix(true_labels_neck, pred_labels_neck))
    print(classification_report(true_labels_neck, pred_labels_neck))

    print("Sleeve Length")
    print(confusion_matrix(true_labels_sleeves, pred_labels_sleeves))
    print(classification_report(true_labels_sleeves, pred_labels_sleeves))


    print("Pattern")
    print(confusion_matrix(true_labels_pattern, pred_labels_pattern))
    print(classification_report(true_labels_pattern, pred_labels_pattern))
