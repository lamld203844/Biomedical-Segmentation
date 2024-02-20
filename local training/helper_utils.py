import os
import cv2
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

"""# Dataloader

"""
b_sz = 8 # batch size

# define SEGMENTATION dataset class
class SegmentationDataset(Dataset):
    def __init__(self, dataset_path, mask_dir, imgs, masks, transform=None):
        self.imgs = imgs
        self.masks = masks
        self.transform = transform
        self.dataset_path = dataset_path
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = f'{self.dataset_path}/image/{img_name}'
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        mask_name = self.masks[idx]
        mask_path = f'{self.dataset_path}/{self.mask_dir}/{mask_name}'
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # read mask in grayscale

        # loss func: crossentropy just accept: 0, 1, 2, ... like indices
        mask[mask == 7] = 1 # convert mask value 7(SP) -> 1
        mask[mask == 8] = 2 # convert mask value 8(head) -> 2

        mask = Image.fromarray(mask)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # convert to long tensor
        # and .squeeze() > removes redundant dimensions (1, rz, rz) > (rz, rz)
        # * 255 to avoid = 0 (due to long()), since being scaling to [0, 1] by .ToTensor()
        mask = (mask * 255).long().squeeze()

        return image, mask

t_test = T.Compose([
        T.CenterCrop((1024, 1024)),
        T.Resize(size=(128, 128), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
])

t_train = T.Compose([
        T.CenterCrop((1024, 1024)),
        T.Resize(size=(128, 128), interpolation=T.InterpolationMode.NEAREST),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=20),
        T.ToTensor(),
        T.RandomErasing(p=0.2),
])


def get_dataset_loaders(dataset_path, mask_dir):
    '''
    input:
        - dataset_path,
        - mask_dir: subdir in dataset_path i.e 'mask' or 'enhance_mask'
    output:
        - train_loader, val_loader, test_loader
    '''
    #### FETCH DATASET FROM KAGGLE

    # API token
    username = "ryanlliu"
    api_token = "ffaea405d69ad6362ec708866991bc29"

    # Create `~/.kaggle` directory if it doesn't exist
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)

    # Save username and API token to `kaggle.json` securely
    with open(os.path.join(os.path.expanduser("~/.kaggle"), "kaggle.json"), "w") as f:
        f.write(f'{{"username": "{username}", "key": "{api_token}"}}')

    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

    os.system("kaggle datasets download -d ryanlliu/jnu-ifm-for-segment-pubic-symphysis-fetal-head")
    os.system("unzip -q /content/jnu-ifm-for-segment-pubic-symphysis-fetal-head.zip")


    ## GET IMAGE AND MASK PATHS
    img_paths = os.listdir(f'{dataset_path}/image')
    mask_paths = os.listdir(f'{dataset_path}/{mask_dir}')

    imgs = [img.replace("_mask", "") for img in mask_paths] # Mapping 1-1 between mask and image

    ## TEST-TRAIN SPLITTING ratio 9(9-1)-1
    train_img, test_img, train_mask, test_mask = train_test_split(imgs,
                                                                  mask_paths,
                                                                  test_size = 0.1,
                                                                  random_state=123)
    train_img, val_img, train_mask, val_mask = train_test_split(train_img,
                                                                train_mask,
                                                                test_size = 0.1,
                                                                random_state=123)

    ## DEFINE DATASET
    train_set = SegmentationDataset(
        dataset_path,
        mask_dir,
        train_img,
        train_mask,
        transform=t_train)
    val_set = SegmentationDataset(
        dataset_path,
        mask_dir,
        val_img,
        val_mask,
        transform=t_test)
    test_set = SegmentationDataset(
        dataset_path,
        mask_dir,
        test_img,
        test_mask,
        transform=t_test)

    ## DEFINE DATALOADER
    train_loader = DataLoader(dataset=train_set, batch_size=b_sz, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=val_set, batch_size=b_sz, shuffle=False, num_workers=2)
    test_loader = DataLoader(dataset=test_set, batch_size=b_sz, shuffle=True, num_workers=2)

    return train_loader, val_loader, test_loader

def test_dataloader(
        dataset_path = '/content/dataset'
        mask_dir = 'mask' # type of mask set: "mask" or "enhance mask"
):
    train_loader, val_loader, test_loader = get_dataset_loaders(dataset_path, mask_dir)

    def visualize(loader, n=b_sz):
        # Get a batch of images and masks from the loader
        images, masks = next(iter(loader))

        # Plot the images and masks individually
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 10))
        axes = axes.flatten()  # Flatten the 2D array of axes into a 1D array

        for i in range(n):
            # Input Image
            axes[i].imshow(images[i].permute(1, 2, 0), cmap='gray')
            axes[i].set_title(f'Input Image {i+1}')
            axes[i].axis('off')  # Hide axis labels and ticks

            # Ground Truth
            axes[i+n].imshow(masks[i].squeeze(), cmap='viridis')
            axes[i+n].set_title(f'Ground Truth {i+1}')
            axes[i+n].axis('off')

        plt.tight_layout()  # Adjust spacing between subplots
        plt.show()

    visualize(val_loader, b_sz)