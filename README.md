English | [简体中文](README_zh_CN.md)

# Technical Document

Medical image segmentation algorithms can extract key information from automatically generated images of specific tissues, eliminating the enormous amount of time spent manually drawing medical images in clinical settings, and thus becoming a hot research topic for scholars. However, the problem with evaluating segmentation performance of existing medical image segmentation algorithms is that they require high-precision annotated cell datasets as support, but manually annotated cells inevitably contain errors, which is not conducive to model generalization. To address this issue, this method proposes a cell generation method based on conditional generative adversarial networks.

Firstly, the StyleGAN3 network is used to train the cell's mask to obtain a mask image that can control style information. Then, features are trained using the Pix2PixHD network, and the mask image obtained in the previous step is used as input to obtain cell images that fully conform to semantic information. Finally, this method inputs the generated cell images into existing medical image segmentation algorithms to measure the performance of the model, truly reflecting the strengths and weaknesses of various algorithms.

In the process of image generation, StyleGAN3 and Pix2PixHD models are used, and the specific technical details are as follows:

## Requirements

- Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
- 1–8 high-end NVIDIA GPUs with at least 12 GB of memory. 
- 64-bit Python 3.8 and PyTorch 1.12.0 (or later). See https://pytorch.org for PyTorch install instructions. CUDA toolkit 11.6 or later.
- GCC 7 or later (Linux) or Visual Studio (Windows) compilers. Recommended GCC version depends on CUDA version, see for example [CUDA 11.6 system requirements](https://docs.nvidia.com/cuda/archive/11.6.0/cuda-installation-guide-linux/index.html#system-requirements).
- Python libraries: see `environment.yml `for exact library dependencies. You can use the following commands with Miniconda3/Anaconda3 to create and activate your cell Python environment:
  - `conda env create -f environment.yml`
  - `conda activate cell`

## 1.Stylegan3

### 1.1 Preparing datasets

You can put your images in a folder to create your custom dataset; use `python dataset_tool.py --help` for more detailed information. Additionally, the folder can be used as a dataset directly without running `dataset_tool.py` first, but this may lead to suboptimal results.

```.bash
# Original 1024x1024 resolution.
python dataset_tool.py --source=./your/path/images1024x1024 --dest=./datasets/images-1024x1024.zip

# Scaled down 256x256 resolution.
python dataset_tool.py --source=./your/path/images1024x1024 --dest=~/datasets/images-256x256.zip \
--resolution=256x256
```

Please note that the above command will create a single combined dataset using all images from all categories in the folder, matching the setting used in the StyleGAN3 paper. Additionally, you can also create a separate dataset for each category.

```python
python dataset_tool.py --source=./your/path/cell1024x1024 --dest=~/datasets/cell-1024x1024.zip
python dataset_tool.py --source=./your/path/nucleus1024x1024 --dest=~/datasets/Nucleus-1024x1024.zip
```

### 1.2 Training

You can use `train.py` to train a new neural network, for example:

```.bash
# Train StyleGAN3-T for Dataset using 8 GPUs.
python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=~/datasets/images-1024x1024.zip \
    --gpus=8 --batch=32 --gamma=8.2 --mirror=1

# Fine-tune StyleGAN3-R for Dataset using 1 GPU, starting from the pre-trained FFHQ-U pickle.
python train.py --outdir=~/training-runs --cfg=stylegan3-r --data=~/datasets/images-1024x1024.zip \
    --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \
    --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl

# Train StyleGAN2 for Dataset at 1024x1024 resolution using 8 GPUs.
python train.py --outdir=~/training-runs --cfg=stylegan2 --data=~/datasets/images-1024x1024.zip \
    --gpus=8 --batch=32 --gamma=10 --mirror=1 --aug=noaug
```

Note that the result quality and training time depend heavily on the exact set of options. The most important ones (`--gpus`, `--batch`, and `--gamma`) must be specified explicitly, and they should be selected with care. See [`python train.py --help`](./docs/train-help.txt) for the full list of options and [Training configurations](./docs/configs.md) for general guidelines &amp; recommendations, along with the expected training speed &amp; memory usage in different scenarios.

The results of each training run are saved to a newly created directory, for example `~/training-runs/00000-stylegan3-t-afhqv2-512x512-gpus8-batch32-gamma8.2`. The training loop exports network pickles (`network-snapshot-<KIMG>.pkl`) and random image grids (`fakes<KIMG>.png`) at regular intervals (controlled by `--snap`). For each exported pickle, it evaluates FID (controlled by `--metrics`) and logs the result in `metric-fid50k_full.jsonl`. It also records various statistics in `training_stats.jsonl`, as well as `*.tfevents` if TensorBoard is installed.

### 1.3 Testing

Trained networks are stored as `*.pkl` files that can be referenced using local filenames :

```.bash
# Generate an image using a model .
python gen_images.py --outdir=~/out --trunc=1 --seeds=2 \
    --network=~/training-runs/00000-stylegan3-r-labels1024x1024-gpus1-batch2-gamma8.8/network-snapshot-000220.pkl
    # Generate multiple images using a model .
python gen_images.py --outdir=~/out --trunc=1 --seeds=1-100 \
    --network=~/training-runs/00000-stylegan3-r-labels1024x1024-gpus1-batch2-gamma8.8/network-snapshot-000220.pkl
```

### 1.5 Image Converter

为确保StyleGAN生成的图片为二值化的图片，您可以使用`Tools/Threshold.py`将生成的图片转化为标准的黑白二值图像。

## 2.Pix2pixHD

### 2.1 Training

```python
# Traina model at 1024 x 1024 resolution
python train.py --name cell--label_nc 0 --dataroot ./your/data/path/ --no_instance
```

#### Training with your own dataset

- If you want to train with your own dataset, please generate label maps which are one-channel whose pixel values correspond to the object labels (i.e. 0,1,...,N-1, where N is the number of labels). This is because we need to generate one-hot vectors from the label maps. Please also specity `--label_nc N` during both training and testing.
- If your input is not a label map, please just specify `--label_nc 0` which will directly use the RGB colors as input. The folders should then be named `train_A`, `train_B` instead of `train_label`, `train_img`, where the goal is to translate images from A to B.
- If you don't have instance maps or don't want to use them, please specify `--no_instance`.
- The default setting for preprocessing is `scale_width`, which will scale the width of all training images to `opt.loadSize` (1024) while keeping the aspect ratio. If you want a different setting, please change it by using the `--resize_or_crop` option. For example, `scale_width_and_crop` first resizes the image to have width `opt.loadSize` and then does random cropping of size `(opt.fineSize, opt.fineSize)`. `crop` skips the resizing step and only performs random cropping. If you don't want any preprocessing, please specify `none`, which will do nothing other than making sure the image is divisible by 32.

### 2.2 Testing

```python
# Test the model
python test.py --name cell --label_nc 0 --no_instance --which_epoch 80 --how_many 10
```

### 2.3 Details

在Pix2pixHD中你可以同时训练您的细胞图像以及荧光点图像，荧光点的Mask可以通过二维高斯分布生成，我们在`Tools/Points.py`中提供了此功能。

### 2.4 More Training/Test Details

- Flags: see `options/train_options.py` and `options/base_options.py` for all the training flags; see `options/test_options.py` and `options/base_options.py` for all the test flags.
- Instance map: we take in both label maps and instance maps as input. If you don't want to use instance maps, please specify the flag `--no_instance`.



## 3.Image Segmentation

在这一步，我们使用生成的数据集来比较各分割算法的性能。

### 3.1 Cellpose

传统的分水岭方法对于有明确边界的对象能够取得较好的分割效果，因为该算法能形成一个个小的“盆地”，这些盆地就代表一个个对象。但是，大多数情形下，不同的对象形成不同“深度”（强度）的盆地，很难统一进行分割。因此，创建一个关于对象的中间表达Intermediate Representation，来形成统一的拓扑盆地，就是一个很好的方法。Cellpose做的就是对Mask进行模拟扩散Simulated Diffusion，形成一个矢量场的拓扑映射。

#### Run Cellpose in GUI

```
# Install cellpose and the GUI dependencies from your base environment using the command
python -m pip install cellpose[gui]

# The quickest way to start is to open the GUI from a command line terminal.
python -m cellpose
```

- 在GUI中加载图像（拖入图像或从File菜单中加载）；
- 设置模型：Cellpose中有两个模型，cytoplasm和nuclei，即细胞质模型和细胞核模型。
- 设置通道：选择要分割的图像通道，如果想分割细胞质，即选择green通道；如果想分割细胞核，则选择red/blue通道。如果是分割细胞质，且图中还有细胞核，则将chan设置为细胞质所在通道，而chan2通道设置为细胞核所在通道；如果分割细胞质但里面没有细胞核，则只设置chan即可，chan2设为None；
- 点击calibrate按钮来预估图中物体的尺寸；也可以手动输入cell diameter来设置。该预估的尺寸会通过左下方的红色圆盘体现；
- 点击run segmentation来运行模型。可以通过是否勾选MASKS ON来调节是否显示分割后的掩膜。

#### Run Cellpose in Terminal

上面GUI界面中的参数输入同样可以通过命令行模式来实现：

```python
python -m cellpose --dir ~/images_cyto/test/ --pretrained_model cyto --chan 2 --chan2 3 --save_png
```

所有的参数可以通过help参数来查看：

```python
python -m cellpose -h
```

#### Run Cellpose in Code

与上面两种方式类似，也可以在Python代码中直接调用Cellpose进行编程：

```python
from cellpose import models
import skimage.io

model = models.Cellpose(gpu=False, model_type='cyto')

files = ['img0.tif', 'img1.tif']

imgs = [skimage.io.imread(f) for f in files]

masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=[0,0],
                                         threshold=0.4, do_3D=False)
```

### 3.2 Cellpofiler

CellProfier是由哈佛和MIT的Broad Institute开发的一款免费软件，旨在让生物学家无需计算机视觉或编程方面的培训，即可自动定量测量数千张图像的表型。

### 3.3 Deepcell

研究人员开发了一种深度学习的分割算法，Mesmer（由一个 ResNet50 主干和一个特征金字塔网络组成），其实现了关键细胞特征的自动提取，如蛋白质信号的亚细胞定位，达到了人类水平的性能。

#### Run Deepcell

访问[deepcell.org](https://deepcell.org/)上预训练的深度学习模型。该网站允许你轻松上传示例图像，在可用模型上运行，并下载结果，不需要任何本地安装。

### 3.4 Mask R-CNN

#### Introduction

Mask R-CNN算法流程如下：

- 将图片输入到一个预训练好的神经网络中（ResNeXt等）获得对应的feature map；
- 对Feature Map中的每一点设定预定个的ROI，从而获得多个候选ROI；
- 将候选的ROI送入RPN网络进行二值分类（前景或背景）和BB回归，过滤掉一部分候选的ROI；
- 对剩下的ROI进行ROIAlign操作（即先将原图和Feature Map的Pixel对应起来，然后将Feature Map和固定的Feature对应起来）；
- 对ROI进行分类（N类别分类）、BB回归和MASK生成（在每一个ROI里面进行FCN操作）。

### 3.5 U2-Net

U2 -Net 的架构是一个两级嵌套的 U 结构。

- 提出残差 U 块 (RSU) 中混合了不同大小的感受野，它能够从不同的尺度捕获更多的上下文信息。
- 这些 RSU 块中使用了池化操作，它增加了整个架构的深度，而不会显著增加计算成本。

## 4.Tips

如果您使用Windows运行代码，您需要首先安装Visual Studio，安装C++桌面开发的库。并且配置VS的环境变量Path、LIB和INCLUDE，它们的内容可能是：

![visual_studio_1](D:\Project\Cell_generation_pipeline\Docs\imgs\visual_studio_1.png)

- Path=C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64;

- LIB=C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\lib\x64;
- INCLUDE=C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\include;
