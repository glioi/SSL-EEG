from distutils.command.build import build
import torch
from torch import double, optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from braindecode.models import TIDNet
import os
import math
import time
import random
import sys
print("Using pytorch version: " + torch.__version__)



### local imports
from utils import loso,train_epoch,validate_epoch,test

###  parameters (do it with argparse at some point!!)
path='/home/g20lioi/fixe/datasets/BCI/bci4/'
resultdir='/home/g20lioi/fixe/datasets/BCI/bci4/results_SSL_GIGI/'
batch_size=32
n_subjects=9 #total number of subjects in the dataset 
n_classes=4
cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
test_subject=3
learning_rate=1e-3
n_epochs=100
patience=40

print("Importing local files: ", end = '')


def build_network(n_classes,n_chans,time_samples):

    model=TIDNet(n_classes=n_classes, in_chans=n_chans, input_window_samples=time_samples, s_growth=24, t_filters=32,
                 drop_prob=0.4, pooling=15, temp_layers=2, spat_layers=2, temp_span=0.05,
                 bottleneck=3, summary=-1)
    summary(model, (batch_size,n_chans, time_samples))
    return model.double().to(device)


### generate the dataset split
T_x,T_y, V_x,V_y,Test_x,Test_y=loso(test_subject,n_subjects, path,device)
print('train data shape:',T_x.shape)

### model and optimizer
model=build_network(n_classes,T_x.shape[1],T_x.shape[2])
optimizer = optim.AdamW(model.parameters(),lr=learning_rate,  weight_decay=0.01, amsgrad=True)


Train_acc=[]
Val_acc=[]
Train_loss=[]
Val_loss=[]
Test_acc=[]


tr_iter=0
v_itr=0
maxv=0
cpt_early = 0



outputs=model(torch.rand(batch_size,T_x.shape[1],T_x.shape[2],dtype=double))
print(outputs.shape)

for e in range(n_epochs):
    print('epoch:',e)
    
    train_loss,train_acc,tr_iter = train_epoch(model, optimizer, T_x, T_y, batch_size, tr_iter)
    print('train loss {} accuracy {} epoch {} done'.format(train_loss,train_acc,e))

    val_loss,val_acc,v_itr = validate_epoch(model, V_x, V_y, batch_size, v_itr)
    print('val loss {} epoch {} done'.format(val_loss,e))

    Train_acc.append(train_acc)
    Val_acc.append(val_acc)
    Train_loss.append(train_loss)
    Val_loss.append(val_loss)

    if maxv<val_acc:
        print(f"Epoch {e}, new best val accuracy {val_acc} and loss {val_loss}")
        maxv=val_acc

        ckpt_dict = {
            'weights': model.state_dict(),
            'train_acc': Train_acc,
            'val_acc': Val_acc,
            'train_loss': Train_loss,
            'val_acc': Val_loss,
            'epoch': e
                }
        torch.save(ckpt_dict,os.path.join(resultdir,f"bestval.pth") )
        cpt_early = 0
    else:
        cpt_early +=1
            
        if cpt_early == patience:
            print("Early Stopping")
            break



