# Dataset

### Simple version of JNU-IFM dataset

Simple version of JNU-IFM dataset consists of **78 video, 6224 frames (img = mask = enhance_mask), 51 patients**

### Structure of my dataset version

- `./image`: screenshot extracted from the video, naming rule: "video name" + "_frame sequence number".png
- `./mask`: the mask of the corresponding image in the image folder, naming rules: "video name" + "_frame number" + "_mask".png)
- `./mask_enhance`: mask after image processing i.e. red = SP (Symphysis pubic), green = Head
- `framelabel.csv`: each entry is the type of label (`frame_label`) corresponding to the frame number (`frame_id`) of specific video (`video_name`)

### Note

1. The images in the **image** folder have undergone basic preprocessing. After cropping and overwriting operations, the original interface toolbar and text information in the image are removed. 
    - The processed size is **1295*1026**
    - In order to prevent information loss, the downsampling process is not performed
    - At the same time, it has been converted into a grayscale image, which can be directly read in grayscale format when using it.
2. The image in the **mask** folder may look completely black due to the low label value (pixel value 7-SP, 8-Head).
3. Frame_label frame label 3-None, 4-OnlySP, 5-OnlyHead, 6-SP+Head in framelabel.csv.

### Links to original dataset, challenge:

- challenge: https://ps-fh-aop-2023.grand-challenge.org/
- dataset description paper: https://www.sciencedirect.com/science/article/pii/S2352340922001160
- dataset: https://figshare.com/articles/dataset/JNU-IFM/14371652?file=34761769

# GPU accelerating

### **Nvidia Installation**

- **NVIDIA:**
    - **[CUDA](https://developer.nvidia.com/cuda-toolkit)**
        - Check installed: `nvcc --version` (preferable) or `nvidia-smi`
    - [**cuDNN**](https://developer.nvidia.com/cudnn)
        - **Windows**: copy the contents of the **`bin`**, **`include`**, and **`lib`** folders from the cuDNN package to the corresponding folders in the CUDA installation directory (usually **`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y`**, where **`X.Y`** is your CUDA version).
- [**PyTorch**](https://pytorch.org/get-started/locally/)
    - Using **`pip`** corresponding command
- **Checking**
    
    ```python
    import torch
    print(torch.cuda.is_available())
    ```
    
    True means accelerated GPU
    

### PyTorch with GPU

- **Switching between CPU and GPU**
    
    ```python
    import torch
    print(torch.cuda.is_available())
    ```
    
    True
    
    ```python
    my_tensor = torch.tensor([1., 2., 3.])
    my_tensor
    ```
    
    tensor([1., 2., 3.])
    
    ```python
    my_tensor.to('cuda')
    ```
    
    tensor([1., 2., 3.], device='cuda:0')
    
    ```python
    my_tensor.to('cpu')
    ```
    
    tensor([1., 2., 3.])