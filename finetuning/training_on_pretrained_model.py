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

# pretrained model:
base_model = ""

rasters = ""
masks = ""
outlines = ""
dist = ""

logdir = ""
model_save_path = ""
experiment_name = ""

arch = "Unet-resnet18"
width = 256
batchsize = 16
in_channels = 8
gpus = 2
backend = "dp"
max_epochs = 30 + 60 - 1
lr = 3E-4

training_split = 0.8

model_name = ("{}_epochs={}_lr={}_width={}_bs={}"
              "_base_model={}").format(arch,
                                       max_epochs,
                                       lr,
                                       width,
                                       batchsize,
                                       base_model)


###################################
#            load model           #
###################################
print("Loading model")

model_names = args.model

if isinstance(model_names, str):
    # LK modified
    # model = torch.jit.load(args.model).to(args.device)
    model = torch.jit.load(args.model)
elif isinstance(model_names, list):
    # LK modified
    # models = [torch.jit.load(m).to(args.device) for m in model_names]
    models = [torch.jit.load(m) for m in model_names]
    model = AveragingModel(models)
else:
    print("Error during model loading.")
    sys.exit(1)

print("Model loaded")


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
                          red=3,
                          nir=4,
                          dilate_second_target_band=2,
                          rescale_ndvi=True)

model = TreeCrownDelineationModel(in_channels=in_channels, lr=lr)

#%%
trainer = Trainer(gpus=gpus,
                  distributed_backend=backend,
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
