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
rasters = []
masks = []
outlines = []
dist = []
for i in range(39):
    rasters.append(f"training_owndata/data_vs1/tiles/TOM_378000_5711000_tile_{i}.tif")
    masks.append(f"training_owndata/data_vs1/masks/mask_{i}.tif")
    outlines.append(f"training_owndata/data_vs1/outlines/outline_{i}.tif")
    dist.append(f"training_owndata/data_vs1/dist_trafo/dist_trafo_{i}.tif")

logdir = "training_owndata/vs1"
model_save_path = "training_owndata/vs1"
experiment_name = "vs1_geomorpho_training_data"

arch = "Unet-resnet18"
width = 256
batchsize = 16
in_channels = 5  # number of channels includes NDVI if set True below
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
                  max_epochs=max_epochs)
trainer.fit(model, data)

#%%
model.to("cpu")
t = torch.rand(1, in_channels, width, width, dtype=torch.float32)
model.to_torchscript(
    os.path.abspath(model_save_path) + "/" + experiment_name + '/' + model_name + "_jitted.pt",
    method="trace",
    example_inputs=t)
