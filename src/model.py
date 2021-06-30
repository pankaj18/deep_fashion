import torch
import torch.nn as nn
from torchvision import models



class Net(nn.Module):
  def __init__(self,n_classes_neck,n_classes_sleeve,n_classes_pattern):
    super().__init__()
    # take the model without classifier
    self.base_model=models.mobilenet_v2(pretrained=True).features
    for param in self.base_model.parameters():
      param.requires_grad = False
    # size of the layer before the classifier
    last_channel = models.mobilenet_v2().last_channel
    self.pool = nn.AdaptiveAvgPool2d((1,1))
    # create separate classifiers for our outputs
    self.neck=nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features=last_channel,out_features=n_classes_neck)
    )
    self.sleeve=nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features=last_channel,out_features=n_classes_sleeve)
    )
    self.pattern=nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features=last_channel,out_features=n_classes_pattern)
    )
  def forward(self,x):
    x=self.base_model(x)
    x=self.pool(x)
    # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
    x=torch.flatten(x,start_dim=1)
    out1=self.neck(x)
    out2=self.sleeve(x)
    out3=self.pattern(x)
    return out1,out2,out3