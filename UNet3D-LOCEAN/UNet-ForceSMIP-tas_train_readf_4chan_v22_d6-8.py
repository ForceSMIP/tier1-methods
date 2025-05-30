#!/usr/bin/env python
# coding: utf-8

# I/O / data wrangling
import os
import glob as gb
import re
import numpy as np
import xarray as xr
from scipy.stats import randint

# runtime metrics
import time as clocktime

# define a lambda function to perform natural sort
natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split("(\d+)", s)]

root_dir = "/gpfswork/rech/ryn/rces866/ForceSMIP/"  # path to forcesmip data (ETH)
#root_dir = "/glade/campaign/cgd/cas/asphilli/ForceSMIP/"  # path to forcesmip data (NCAR)

outdir = "/gpfswork/rech/ryn/rces866/tmpdir/"  # directory where output data should be saved

ncvar='tas'  # CMIP variable name to be used

# choices include: 'CESM2', 'CanESM5', 'MIROC-ES2L', 'MIROC6', 'MPI-ESM1-2-LR'
training_models = ["CESM2","CanESM5","MIROC-ES2L","MIROC6","MPI-ESM1-2-LR"]
#training_models = ["CESM2","MIROC-ES2L","MIROC6"]

n_members = 50  # number of members for training

nb_mb_mod = [50,25,30,50,30]
nb_mb ={ "CESM2":50,"CanESM5":25,"MIROC-ES2L":30,"MIROC6":50,"MPI-ESM1-2-LR":30}
nb_mb_t ={ "CESM2":25,"CanESM5":30,"MIROC-ES2L":25,"MIROC6":25,"MPI-ESM1-2-LR":25}

balanced=False

import torch
import torch.nn as nn

# Convert into tensor

valid_mod=["CESM2","CanESM5","MIROC-ES2L","MIROC6","MPI-ESM1-2-LR"]
index_val=0
nb_mb_mod.pop(index_val)
nb_mb_mod_cum = np.cumsum(nb_mb_mod)

reference_period_list = [("1880-01-01", "1952-12-31"),
                         ("1890-01-01", "1962-12-31"),
                         ("1900-01-01", "1972-12-31"),
                         ("1910-01-01", "1982-12-31"),
                         ("1920-01-01", "1992-12-31"),
                         ("1930-01-01", "2002-12-31"),
                         ("1940-01-01", "2012-12-31"),
                         ("1950-01-01", "2022-12-31"),
                         ("1960-01-01", "2032-12-31"),
                         ("1970-01-01", "2042-12-31"),
                         ("1980-01-01", "2052-12-31"),
                         ("1990-01-01", "2062-12-31"),
                         ("2000-01-01", "2072-12-31"),
                         ("2010-01-01", "2082-12-31"),
                         ("2020-01-01", "2092-12-31"),
                        ]

nb_per=len(reference_period_list)

d_begin=6
d_end=9

NB_epoch=90
# Read weight (based on aera)

version='d6-8_v38'

target=ncvar
wgts=torch.load(outdir+'wgts.pt')
wgts_oce=torch.load(outdir+'wgts_oce.pt')

if ( ncvar == 'tos' ):
    wgts=wgts_oce

#wgts_=np.stack([wgts,wgts,wgts,wgts_oce],axis=0)
#wgts_.shape

# In[6]:

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_X_IDs, list_y_IDs, list_std):
        'Initialization'
        self.list_X_IDs = list_X_IDs
        self.list_y_IDs = list_y_IDs
        self.list_std = list_std

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_X_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X_ID = self.list_X_IDs[index]
        y_ID = self.list_y_IDs[index]
        std_ID = self.list_std

        std_tas=torch.load(outdir+'X_std_tensor_'+list_std+'tas_v3.pt')
        std_psl=torch.load(outdir+'X_std_tensor_'+list_std+'psl_v3.pt')
        std_zmta=torch.load(outdir+'X_std_tensor_'+list_std+'zmta_v3.pt')
        std_tos=torch.load(outdir+'X_std_tensor_'+list_std+'tos_v3.pt')
        X = torch.stack(
            (torch.load(outdir+'X_norm_tensor_'+X_ID+'tas_v3.pt')/std_tas,
             torch.load(outdir+'X_norm_tensor_'+X_ID+'psl_v3.pt')/std_psl,
             torch.load(outdir+'X_norm_tensor_'+X_ID+'zmta_v3.pt')/std_zmta,
             torch.load(outdir+'X_norm_tensor_'+X_ID+'tos_v3.pt')/std_tos),
            dim=0) 
        n=torch.randint(0, 144, (1,))
        ind_permuted=np.concatenate((np.arange(n,144),
                         np.arange(0,n)  ))
        X[2,:,:,:]=X[2,:,:,ind_permuted]

        ncvar='tas'
        #std_ano=torch.load(outdir+'y-ANO_std_tensor_'+list_std+ncvar+'_v3.pt')
        #std_em=torch.load(outdir+'y-EM_std_tensor_'+list_std+ncvar+'_v3.pt')
        y = torch.load(outdir+'y-EM_norm_tensor_'+y_ID+ncvar+'_v3.pt') #/std_em,

        return X.float(), y.float()

reference_period=reference_period_list[7]

partition_X={'validation':[], 'train': []}
partition_y={'validation':[], 'train': []}

rng = np.random.default_rng()

list_std=valid_mod[index_val]+'_ALL-BUT'+'_'+reference_period[0]+\
           '-'+reference_period[1]+'_'

for i in range(1,nb_mb[valid_mod[index_val]]+1):
    partition_X['validation'].append(valid_mod[index_val]+'_'+str(i)+'_'+
            reference_period_list[7][0]+'-'+reference_period_list[7][1]+'_')
    partition_y['validation'].append(valid_mod[index_val]+'_'+
            reference_period_list[7][0]+'-'+reference_period_list[7][1]+'_')           
     #      [valid_mod[index_val]+'_'
     #        +reference_period[0]+'-'+reference_period[1]+'_'],
     #     'train': []}

if (not(balanced)):        

  for ip in range(d_begin,d_end):
    all_models=["CESM2","CanESM5","MIROC-ES2L","MIROC6","MPI-ESM1-2-LR"]
    all_models.remove(valid_mod[index_val])
    for model in all_models:
        for i in range(1,nb_mb[model]+1):
           partition_X['train'].append(model+'_'+str(i)+'_'+
                reference_period_list[ip][0]+'-'+reference_period_list[ip][1]+'_')
           partition_y['train'].append(model+'_'+
                reference_period_list[ip][0]+'-'+reference_period_list[ip][1]+'_')


#!nvidia-smi
from torch.cuda import device_count
print('Torch detecting {0} GPUs compatible with CUDA'.format(device_count()))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Currently used device is :', device)

params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 8}

validation_set = Dataset(partition_X['validation'],partition_y['validation'],list_std)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

if (not(balanced)):
  training_set = Dataset(partition_X['train'],partition_y['train'],list_std)
  training_generator = torch.utils.data.DataLoader(training_set, **params)


# In[12]:


import torch.nn.functional as F

class conv_geo_lin(nn.Module):
    def __init__(self, in_ch, out_ch,k,km):
        super(conv_geo_lin, self).__init__()
        self.conv_nopad = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, [(k+1)*2+1,k*2+1,k*2+1], 
                    padding=[0,0,0], bias=False)
            )
        self.km = km
        self.k = k
        
    def forward(self, x):
        #print(self.k)
        k=self.k
        #print(self.km)
        x = F.pad(x,[k,k,0,0,0,0],mode='circular')
        x = F.pad(x,[0,0,k,k,0,0],mode='reflect')
        x = F.pad(x,[0,0,0,0,k+1-self.km,k+1-self.km],mode='replicate')
        #print('x shape=',x.shape)
        x = self.conv_nopad(x)
        return x

class conv_geo(nn.Module):
    def __init__(self, in_ch, out_ch,k,km):
        super(conv_geo, self).__init__()
        self.conv_nopad = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, [(k+1)*2+1,k*2+1,k*2+1], 
                    padding=[0,0,0], bias=False),
            nn.ReLU(inplace=True)
            )
        self.km = km
        self.k = k

    def forward(self, x):
        #print(self.k)
        k=self.k
        #print(self.km)
        x = F.pad(x,[k,k,0,0,0,0],mode='circular')
        x = F.pad(x,[0,0,k,k,0,0],mode='reflect')
        x = F.pad(x,[0,0,0,0,k+1-self.km,k+1-self.km],mode='replicate')
        #print('x shape=',x.shape)
        x = self.conv_nopad(x)
        return x

# k,j,i -> k,j-4,i-4
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch,dropout,k,km):
        super(double_conv, self).__init__()
        self.convg = conv_geo(in_ch, out_ch,k,km) 
        self.convg1 = conv_geo(out_ch, out_ch,k,km) 
        self.drop =  nn.Dropout(p=dropout) 

    def forward(self, x):
        x = self.convg(x)
        x = self.convg1(x)
        x = self.drop(x)
        return x

# k,j,i -> k*3,j*2,i*2
class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up_scale = nn.ConvTranspose3d(in_ch, out_ch, [5,3,3], stride=[3,2,2], bias=False)

    def forward(self, x1, x2):
        x2 = self.up_scale(x2)

        diffT = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        diffX = x1.size()[4] - x2.size()[4]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffT // 2, diffT - diffT // 2])

        x = torch.cat([x2, x1], dim=1)
        return x

# k,j,i -> k/2,j/2,i/2 -> k/2-4,j/2,i/2
class down_layer1(nn.Module):
    def __init__(self, in_ch, out_ch,drop,k):
        super(down_layer1, self).__init__()
        self.pool = nn.MaxPool3d([3,2,2], stride=[3,2,2], padding=0)
        self.conv = double_conv(in_ch, out_ch,drop,k,1)

    def forward(self, x):
        x = self.conv(self.pool(x))
        return x

# k,j,i -> k/3,j/2,i/2 -> k/3,j/2,i/2
class down_layer2(nn.Module):
    def __init__(self, in_ch, out_ch,drop,k):
        super(down_layer2, self).__init__()
        self.pool = nn.MaxPool3d([3,2,2], stride=[3,2,2], padding=0)
        self.conv = double_conv(in_ch, out_ch,drop,k,0)

    def forward(self, x):
        x = self.conv(self.pool(x))
        return x

# k,j,i -> k*3,j*2,i*2
class up_layer1(nn.Module):
    def __init__(self, in_ch, out_ch,drop,k):
        super(up_layer1, self).__init__()
        self.up = up(in_ch, out_ch)
        self.conv = double_conv(in_ch, out_ch,drop,k,0)

    def forward(self, x1, x2):
        a = self.up(x1, x2)
        x = self.conv(a)
        return x

# k,j,i -> k*3+4,j*2,i*2
class up_layer2(nn.Module):
    def __init__(self, in_ch, out_ch,drop,k):
        super(up_layer2, self).__init__()
        self.up = up(in_ch, out_ch)
        self.conv = double_conv(in_ch, out_ch,drop,k,-1)

    def forward(self, x1, x2):
        a = self.up(x1, x2)
        x = self.conv(a)
        return x

class UNet(nn.Module):
    def __init__(self,thick,dr,k):
        super(UNet, self).__init__()       # 876,76,148
        self.conv1 = double_conv(4, thick,dr,k,0)  # 876,72,144
        self.down1 = down_layer1(thick, thick*2,dr,k)    # 292,36,72 puis 288,36,72
        self.down2 = down_layer2(thick*2, thick*4,dr,k)   # 96,18,36
        self.down3 = down_layer2(thick*4, thick*8,dr,k)   # 32,9,18
        self.up1 = up_layer1(thick*8, thick*4,dr,k)       # 96,18,36
        self.up2 = up_layer2(thick*4, thick*2,dr,k)       # 292,36,72
        self.up3 = up_layer1(thick*2, thick,dr,k)        # 876,72,144  
        self.convg = conv_geo_lin(thick, 1,k,0)

    def forward(self, x):
        #print(np.shape(x))
        x1 = self.conv1(x)
        #print(np.shape(x1))
        x2 = self.down1(x1)
        #print(np.shape(x2))
        x3 = self.down2(x2)
        #print(np.shape(x3))
        x4 = self.down3(x3)
        #print(np.shape(x4))
        x1_up = self.up1(x3, x4)
        #print(np.shape(x1_up))
        x2_up = self.up2(x2, x1_up)
        #print(np.shape(x2_up))
        x3_up = self.up3(x1, x2_up)
        #print(np.shape(x3_up))
        output = self.convg(x3_up)
        #print(np.shape(output))
        return output

# thickness -> 8 
# dropout -> 0.1
# k=1
model=UNet(8,0.5,1).to(device)
#model=UNet()
model.parameters

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("nb de paramètres = ",pytorch_total_params)

stdX=torch.load(outdir+'X_std_tensor_'+list_std+ncvar+'_v3.pt')
std_y_ano=torch.load(outdir+'y-ANO_std_tensor_'+list_std+ncvar+'_v3.pt')
std_y_em=torch.load(outdir+'y-EM_std_tensor_'+list_std+ncvar+'_v3.pt')
stdX=stdX.to(device)
std_y_ano=std_y_ano.to(device)
std_y_em=std_y_em.to(device)

class MSE_areaweighted_Loss(nn.Module):
    def __init__(self, weights):
        super(MSE_areaweighted_Loss, self).__init__()
        self.weights = weights
        
    def forward(self, inputs, y_hat, targets):
        error = torch.square(y_hat - targets)
        weighted_error = error * self.weights
        loss = torch.mean(weighted_error) 
        return loss
    
#Define a Loss function and optimizer

import torch.optim as optim

#X_std_tas_c=X_std_tas.to(device)
wgts=torch.tensor(wgts).to(device)
criterion = MSE_areaweighted_Loss(wgts)
#criterion =  nn.CrossEntropyLoss(torch.tensor([1/3,1/3,1/3]))
optimizer = optim.SGD(model.parameters(), lr=0.01)
#optimizer = optim.Adam(model.parameters(), lr=0.01)

#Define a Loss function and optimizer

#Train the network

continue_train_k=CKCK

loss_train=np.zeros((4,(NB_epoch)*(continue_train_k+1)))
loss_test=np.zeros((4,(NB_epoch)*(continue_train_k+1)))

epoch_prev=0
if (continue_train_k>0):
    epoch_prev=90*continue_train_k
    model=torch.load(model,outdir+'UNet_'+ncvar+'_valid_'+valid_mod[index_val]+\
            '_BalBBBB_KernelKKK_DropOutDDD_ThickTTT_InstIII_epoch_'+str(epoch_prev)+'_'+version)
    loss_train[:,:90]=torch.load(outdir+'loss_train_'+ncvar+'_valid_'+valid_mod[index_val]+\
            '_BalBBBB_KernelKKK_DropOutDDD_ThickTTT_InstIII_epoch_'+str(epoch_prev)+'_'+version)
    loss_test[:,:90]=torch.load(model,outdir+'loss_test_'+ncvar+'_valid_'+valid_mod[index_val]+\
            '_BalBBBB_KernelKKK_DropOutDDD_ThickTTT_InstIII_epoch_'+str(epoch_prev)+'_'+version)

for epoch in range(epoch_prev,NB_epoch+epoch_prev+1):  # loop over the dataset multiple times
     
    if (balanced):
      partition_X['train']=[]
      partition_y['train']=[]
      for ip in range(d_begin,d_end):
        all_models=["CESM2","CanESM5","MIROC-ES2L","MIROC6","MPI-ESM1-2-LR"]
        all_models.remove(valid_mod[index_val])
        for mod in all_models:
            permuted_ind=rng.permutation(nb_mb[mod])
            permuted_ind=permuted_ind[:nb_mb_t[mod]]+1
            for i in permuted_ind : # range(1,nb_mb[model]+1):
              partition_X['train'].append(mod+'_'+str(i)+'_'+
                reference_period_list[ip][0]+'-'+reference_period_list[ip][1]+'_')
              partition_y['train'].append(mod+'_'+
                reference_period_list[ip][0]+'-'+reference_period_list[ip][1]+'_')
      
      #print(partition_X['train'])
      training_set = Dataset(partition_X['train'],partition_y['train'],list_std)
      training_generator = torch.utils.data.DataLoader(training_set, **params)     
    
    print("Epoch=",epoch)
    
    running_loss=0
    running_rmse_ex=0
    running_rmse_in=0
    running_rmse_nadd=0
    for i, data in enumerate(training_generator, 0):
        # get the inputs; data is a list of [inputs, labels]
        #print("i=",i)
        #print("inputs=",inputs)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device) 
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward 
        outputs = model(inputs)
        # gradient optimize
        loss = criterion(inputs,outputs,labels)
        # backward optimization
        loss.backward()
        # Update weigths
        optimizer.step()
        # cumulate loss to print statistics
        running_loss += loss.item()
    
    # test
    correct, total = 0, 0
    test_running_loss = 0.0

    # no need to calculate gradients during inference
    with torch.no_grad():
      for j, data in enumerate(validation_generator, 0):
         inputs, labels = data
         inputs, labels = inputs.to(device), labels.to(device) 
         # forward
         outputs = model(inputs)
         # get the predictions  
         t_loss = criterion(inputs,
                                                    outputs, labels) 
         test_running_loss += t_loss

    loss_test[0,epoch]=test_running_loss / (j+1)
    loss_train[0,epoch]=running_loss / (i+1)
    
    print('['+str(epoch+1)+'] train loss:', loss_train[:,epoch])
    print('['+str(epoch+1)+'] validation loss:', loss_test[:,epoch])

    if ( epoch == ((epoch//5)*5) ) :
        torch.save(model,outdir+'UNet_'+ncvar+'_valid_'+valid_mod[index_val]+\
            '_BalBBBB_KernelKKK_DropOutDDD_ThickTTT_InstIII_epoch_'+str(epoch)+'_'+version)
        torch.save(loss_train,outdir+'loss_train_'+ncvar+'_valid_'+valid_mod[index_val]\
           +'_BalBBBB_KernelKKK_DropOutDDD_ThickTTT_InstIII_epoch_'+str(epoch)+'_'+version+'.pt')
        torch.save(loss_test,outdir+'loss_test_'+ncvar+'_valid_'+valid_mod[index_val]\
           +'_BalBBBB_KernelKKK_DropOutDDD_ThickTTT_InstIII_epoch_'+str(epoch)+'_'+version+'.pt')
    
print('Finished Training')


# In[ ]:

