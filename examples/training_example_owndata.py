# This is an example file showing how to train a model
import os
import torch
import albumentations as A

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from treecrowndelineation import TreeCrownDelineationModel
from treecrowndelineation.dataloading.in_memory_datamodule import InMemoryDataModule


###################################
#      file paths and settings    #
###################################

# path to data-directory, with folder structure as mentioned in README.md (subdirectories: tiles, masks, outlines, dist_trafo)
path_to_datadir = 'path/to/data'
logdir = ''
model_save_path = ''
experiment_name = ''
# number of channels includes NDVI if set True below
# thus for rgb-nir-tifs in_channel=5
# for rgb-nir-ndom-tifs in_channel=6 
in_channels = 5  

################################

#grab last characters of the file name to sort them equally in all lists
def last_chars(x):
    return(x.split('_')[-1])

rasters_filename = os.listdir(os.path.join(path_to_datadir,'tiles'))
rasters = [os.path.join(path_to_datadir,'tiles',el) for el in rasters_filename]
rasters = sorted(rasters, key=last_chars)

masks_filename = os.listdir(os.path.join(path_to_datadir,'masks'))
masks = [os.path.join(path_to_datadir,'masks',el) for el in masks_filename]
masks = sorted(masks, key=last_chars)

outlines_filename = os.listdir(os.path.join(path_to_datadir,'outlines'))
outlines = [os.path.join(path_to_datadir,'outlines',el) for el in outlines_filename]
outlines = sorted(outlines, key=last_chars)

dist_filename = os.listdir(os.path.join(path_to_datadir,'dist_trafo'))
dist = [os.path.join(path_to_datadir,'dist_trafo',el) for el in dist_filename]
dist = sorted(dist, key=last_chars)

arch = "Unet-resnet18"
width = 256
batchsize = 16
cpus = 1  # 2
backend = "dp"
max_epochs = 30 + 60 - 1
lr = 3E-4

training_split = 0.8

model_name = "{}_epochs={}_lr={}_width={}_bs={}".format(arch,
                                                        max_epochs,
                                                        lr,
                                                        width,
                                                        batchsize)

#%%
###################################
#             training            #
###################################
logger = TensorBoardLogger(logdir,
                           name=experiment_name,
                           version=model_name,
                           default_hp_metric=False)

cp = ModelCheckpoint(os.path.abspath(model_save_path) + "/" + experiment_name,
                     model_name + "-{epoch}",
                     monitor="val/loss",
                     save_last=True,
                     save_top_k=2)

callbacks = [cp, LearningRateMonitor()]

train_augmentation = A.Compose([A.RandomCrop(width, width, always_apply=True),
                                A.RandomRotate90(),
                                A.VerticalFlip()
                                ])
val_augmentation = A.RandomCrop(width, width, always_apply=True)

data = InMemoryDataModule(rasters,
                          (masks, outlines, dist),
                          width=width,
                          batchsize=batchsize,
                          training_split=training_split,
                          train_augmentation=train_augmentation,
                          val_augmentation=val_augmentation,
                          concatenate_ndvi=True,
                          red=0,  # true_ind -1 (needed later for calling array indice)
                          nir=3,  # true_ind -1 (needed later for calling array indice)
                          dilate_second_target_band=2,
                          rescale_ndvi=True)

model = TreeCrownDelineationModel(in_channels=in_channels, lr=lr)

#%%
trainer = Trainer(#fast_dev_run=True,
                  accelerator='cpu',
                  devices=cpus,
                  # distributed_backend=backend, # removed in pytorch.lightning 1.5.0
                  logger=logger,
                  callbacks=callbacks,
                  # checkpoint_callback=False,  # set this to avoid logging into the working directory
                  max_epochs=max_epochs,
                  log_every_n_steps=20)
trainer.fit(model, data)

#%%
model.to("cpu")
t = torch.rand(1, in_channels, width, width, dtype=torch.float32)
model.to_torchscript(
    os.path.abspath(model_save_path) + "/" + experiment_name + '/' + model_name + "_jitted.pt",
    method="trace",
    example_inputs=t)
