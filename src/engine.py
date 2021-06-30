import torch
import torch.nn as nn
from tqdm import tqdm

def model_loss_function(outputs,targets):
  o1,o2,o3=outputs
  t1,t2,t3=targets
  #w1,w2,w3=get_weights()
  l1=nn.CrossEntropyLoss(ignore_index=-1)(o1,t1)
  l2=nn.CrossEntropyLoss(ignore_index=-1)(o2,t2)
  l3=nn.CrossEntropyLoss(ignore_index=-1)(o3,t3)
  loss=(l1+l2+l3)/3
  return loss

def train(data_loader, model, optimizer,epochs, device):
    epoch_loss = []
    for i in range(epochs):
        print(i)
        model.train()
        counter = 0
        train_running_loss = 0.0
        for data in tqdm(data_loader):
            counter += 1
            imgs = data['features'].to(device)
            target1 = data['label1'].to(device)
            target2 = data['label2'].to(device)
            target3 = data['label3'].to(device)
            outputs = model(imgs)
            targets = (target1, target2, target3)

            loss = model_loss_function(outputs, targets)
            train_running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / counter
        epoch_loss.append(train_loss)

def evaluate(data_loader, model, device):
    # put model in evaluation mode
    model.eval()
    true_labels_neck = []
    pred_labels_neck = []
    true_labels_sleeves = []
    pred_labels_sleeves = []
    true_labels_pattern = []
    pred_labels_pattern = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            imgs = data['features'].to(device)
            target1 = data['label1'].to(device)
            target2 = data['label2'].to(device)
            target3 = data['label3'].to(device)
            outputs = model(imgs)

            # convert targets and outputs to lists
            all_labels = []
            for out in outputs:
                # all_labels.append(int(np.argmax(out.detach().cpu())))
                all_labels.append(list(out.argmax(axis=1).detach().cpu().numpy()))

            targets = (target1, target2, target3)
            # get all the targets in int format from tensor format
            all_targets = []
            for target in targets:
                all_targets.append(list(target.squeeze(0).detach().cpu().numpy()))
            # print(target1,target2,target3)
            # print(outputs)
            true_labels_neck.extend(all_targets[0])
            pred_labels_neck.extend(all_labels[0])
            true_labels_sleeves.extend(all_targets[1])
            pred_labels_sleeves.extend(all_labels[1])
            true_labels_pattern.extend(all_targets[2])
            pred_labels_pattern.extend(all_labels[2])
    return true_labels_neck,pred_labels_neck,true_labels_sleeves,pred_labels_sleeves,true_labels_pattern,pred_labels_pattern


