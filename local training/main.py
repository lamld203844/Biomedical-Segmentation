import os
import random
import time
from watermark import watermark

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import torchmetrics
from torch.optim import AdamW
import lightning as L
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, EarlyStopping

import segmentation_models_pytorch as smp
from helper_utils import get_dataset_loaders

L.pytorch.seed_everything(123)

#### CHECK VERSION WITH WATERMARK
print(watermark(packages="torch,tensorboard,lightning,torchmetrics"))

"""Loss functions"""
class IoULoss(torch.nn.Module):
    def __init__(self, eps=1e-7):
        super(IoULoss, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        output = F.softmax(output, dim=1)
        target = F.one_hot(target, num_classes=output.shape[1]).permute(0, 3, 1, 2).float()

        intersection = torch.sum(output * target, dim=(2, 3))
        union = torch.sum(output, dim=(2, 3)) + torch.sum(target, dim=(2, 3)) - intersection

        iou = (intersection + self.eps) / (union + self.eps)
        loss = 1. - iou
        return loss.mean()

"""Lightning module"""
class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = model
        self.loss = IoULoss()

        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters(ignore=["model"])

        # Create metrics
        self.train_iou = torchmetrics.JaccardIndex(num_classes=3, average='macro', task='multiclass')
        self.train_acc = torchmetrics.Accuracy(num_classes=3, average='macro', task='multiclass')
        self.train_dice = torchmetrics.Dice(num_classes=3, average='macro')


        self.val_iou = torchmetrics.JaccardIndex(num_classes=3, average='macro', task='multiclass')
        self.val_acc = torchmetrics.Accuracy(num_classes=3, average='macro', task='multiclass')
        self.val_dice = torchmetrics.Dice(num_classes=3, average='macro')


        self.test_iou = torchmetrics.JaccardIndex(num_classes=3, average='macro', task='multiclass')
        self.test_acc = torchmetrics.Accuracy(num_classes=3, average='macro', task='multiclass')
        self.test_dice = torchmetrics.Dice(num_classes=3, average='macro')


    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)

        loss = self.loss(logits, true_labels) # pytorch cross entropy function takes input as logits
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss)
        # Log train accuracy after epoch is completed.
        self.train_acc(predicted_labels, true_labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False
        )
        # train iou
        self.train_iou(predicted_labels, true_labels) # Log train accuracy after epoch is completed.
        self.log(
            "train_iou", self.train_iou, prog_bar=True, on_epoch=True, on_step=False
        )
        # train dice
        self.train_dice(predicted_labels, true_labels)
        self.log(
            "train_dice", self.train_dice, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss  # this is passed to the optimizer for training

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        # Log validation loss
        self.log("val_loss", loss, prog_bar=True)
        # validation accuracy
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True)
        # validation iou
        self.val_iou(predicted_labels, true_labels)
        self.log("val_iou", self.val_iou, prog_bar=True)
        # validation iou
        self.val_dice(predicted_labels, true_labels)
        self.log("val_dice", self.val_dice, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        # intersect over union
        self.test_iou(predicted_labels, true_labels)
        self.log('iou', self.test_iou)
        # accuracy metric
        self.test_acc(predicted_labels, true_labels)
        self.log('accuracy', self.test_acc)
        # dice score/ dice-similarity coefficient (DSC)
        self.test_dice(predicted_labels, true_labels)
        self.log('dice score', self.test_dice)


    def configure_optimizers(self):
#         max_lr = 1e-3
#         epoch = 30
#         weight_decay = 1e-4

#         optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
#         sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
#                                                     steps_per_epoch=len(train_loader)

        optimizer = AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )

        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer

"""Training"""

## MODEL
pytorch_model = smp.Unet(
    encoder_name="resnet34", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet", # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1, # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=3, # model output channels (number of classes in your dataset)
)

class CustomeTimingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        self.start = time.time()
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        self.end = time.time()
        total_minutes = (self.end - self.start) / 60
        print(f"Training is ending took {total_minutes} minutes")

## Dataloader
dataset_path = '/content/dataset'
mask_dir = 'mask' # type of mask set: "mask" or "enhance mask"
train_loader, val_loader, test_loader = get_dataset_loaders(dataset_path, mask_dir)

## Lightning module
lightning_model = LightningModel(model=pytorch_model, learning_rate=0.05)

## callbacks
callbacks = [
    CustomeTimingCallback(),
    ModelCheckpoint(save_top_k=1, mode="max", monitor="train_dice", save_last=True),
    # EarlyStopping(monitor="train_iou", min_delta=0.00, patience=3, mode="max"),

]

## Trainer
trainer = L.Trainer(
    callbacks=callbacks,
    max_epochs=100,
    accelerator="auto",  # set to "auto" or "gpu" to use GPUs if available
    devices="auto",  # Uses all available GPUs if applicable
    logger=CSVLogger(save_dir="logs/", name="my-model"),
)

trainer.fit(
    model=lightning_model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)

PATH = "lightning.pt"
torch.save(pytorch_model.state_dict(), PATH)

# To load model:
# model = PyTorchMLP(num_features=784, num_classes=10)
# model.load_state_dict(torch.load(PATH))
# model.eval()

"""# Testing and evaluation"""



metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

aggreg_metrics = []
agg_col = "epoch"
for i, dfg in metrics.groupby(agg_col):
    agg = dict(dfg.mean())
    agg[agg_col] = i
    aggreg_metrics.append(agg)

df_metrics = pd.DataFrame(aggreg_metrics)
df_metrics[["train_loss", "val_loss"]].plot(
    grid=True, legend=True, xlabel="Epoch", ylabel="Loss"
)

df_metrics[["train_acc", "val_acc"]].plot(
    grid=True, legend=True, xlabel="Epoch", ylabel="ACC"
)

df_metrics[["train_iou", "val_iou"]].plot(
    grid=True, legend=True, xlabel="Epoch", ylabel="Intersect over Union (IoU)"
)

plt.show()

# Evaluate model based on test_step
test_metrics = trainer.test(dataloaders=test_loader, ckpt_path="best")
print(test_metrics)
print(trainer.test(dataloaders=test_loader, ckpt_path="last"))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def visualize_full(loader, model):
    # Get a batch of images and masks from the loader
    images, masks = next(iter(loader))
    images, masks = images.to(device), masks.to(device)
    model.to(device)

    # Get model predictions
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        preds = model(images)

    batch_size = images.size(0)

    # Plot the images, masks, and predictions individually
    fig, axes = plt.subplots(nrows=batch_size, ncols=3, figsize=(10, 10))  # Adjust the number of rows and columns here
    axes = axes.flatten()  # Flatten the 2D array of axes into a 1D array

    for i in range(batch_size):
        # Input Image
        axes[3*i].imshow(images[i].cpu().permute(1, 2, 0), cmap='gray')
        axes[3*i].set_title(f'Input Image {i+1}')
        axes[3*i].axis('off')  # Hide axis labels and ticks

        # Ground Truth
        axes[3*i+1].imshow(masks[i].cpu().squeeze(), cmap='viridis')
        axes[3*i+1].set_title(f'Ground Truth {i+1}')
        axes[3*i+1].axis('off')

        # Model Prediction
        axes[3*i+2].imshow(torch.argmax(preds[i], dim=0).cpu().squeeze(), cmap='viridis')
        axes[3*i+2].set_title(f'Prediction {i+1}')
        axes[3*i+2].axis('off')

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()

visualize_full(test_loader, pytorch_model)