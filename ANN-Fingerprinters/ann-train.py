from models.annencoder import ANNEncoder

from data.dataset_old import MonthDataset2D
from torch import nn
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, ConcatDataset
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

lr_monitor = LearningRateMonitor(logging_interval='epoch')
import xarray as xa

import argparse

seed_everything(42, workers=True)
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent', type=int)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--channel', type=int, default=128)
    parser.add_argument('--variable', type=str, default='tas')
    parser.add_argument('--res', type=int, default=0)
    parser.add_argument('--month_encode', type=int, default=0)
    parser.add_argument('--test_model', type=int, default=0)
    parser.add_argument('--layer', type=int, default=2)
    args = vars(parser.parse_args())
    return args

if __name__ == '__main__':
    args = get_args()
    variable = args['variable']
    lr = args['lr']
    channel = args['channel']
    latent = args['latent']
    res = args['res']
    month_encode = args['month_encode']
    test_model = args['test_model']
    layer = args['layer']
    save_dir = "Production/ANNcoder/"+str(layer)+"-layer-latent-"+str(latent)+"/testModel-"+str(test_model)+variable+'-month_encode-'+str(month_encode)
    
    res = res>0
    month_encode = month_encode>0
    print('variable: ', variable, ' LR: ', lr)
    print('kernel_size ', channel, ' channel, encode month: ', month_encode)
    
    ds_train_all = []
    ds_test_all = []
    models = ['CESM2', 'MIROC6', 'MPI-ESM1-2-LR', 'MIROC-ES2L', 'CanESM5']
    testing_models = [models[test_model]]
    training_models = models.copy()
    training_models.remove(models[test_model])
    print(training_models)
    print(testing_models)
    for i in range(1, 13):
        for model in training_models:
            ds = MonthDataset2D(model=model, var=variable, month=i)
            ds_train_all.append(ds)
    
    for i in range(1, 13):
        for model in testing_models:
            ds = MonthDataset2D(model=model, var=variable, month=i)
            ds_test_all.append(ds)
    
    train_ds = ConcatDataset(ds_train_all)
    test_ds = ConcatDataset(ds_test_all)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True, num_workers=2)
    val_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    hiddens1 = []
    k = channel
    for i in range(layer):
        hiddens1.append(k)
        k = k//2
    hiddens2 = hiddens1[::-1]
    print(hiddens1)
    print(hiddens2)
    # model = UNetLightning(in_channels=months, out_channels=months, channels_config=[32, 64, 128, 256], kernel_size=3)
    model = ANNEncoder(encoder_hiddens=hiddens1,
                        decoder_hiddens=hiddens1,
                        loss='l1', res=res, 
                        latent_dim=latent, lr=lr, drop=0.4, month_encode=month_encode
                        )
    
    # Create a PyTorch Lightning Trainer
    # tb_logger = pl_loggers.TensorBoardLogger(save_dir='YearAnomaly-ResUNet3d-signal-0-unforced-1/monmaxpr-years-136-ExtwDrop/lightning_logs/version_0/')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=save_dir)
    checkpoint_callback = ModelCheckpoint(# dirpath='YearAnomaly-ResUNet3d-signal-0-unforced-1/monmaxpr-years-136-ExtwDrop/lightning_logs/version_0/checkpoints/',
                                          every_n_epochs=5,
                                          save_on_train_epoch_end=True,
                                          filename=None,
                                          save_top_k=-1)
    trainer = pl.Trainer(max_epochs=300, 
                         gpus=[0, 1, 2, 3],
                         # gpus=1,
                         check_val_every_n_epoch=3, gradient_clip_val=2.,
                         logger=tb_logger,
                         callbacks=[lr_monitor],
                         plugins=DDPPlugin(find_unused_parameters=True),
                        )
    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=val_loader)
    