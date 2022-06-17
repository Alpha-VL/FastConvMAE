## Pretraining Fast ConvMAE
## Usage

### Install
- Clone this repo:

```bash
git clone https://github.com/Alpha-VL/FastConvMAE
cd FastConvMAE
```

- Create a conda environment and activate it:
```bash
conda create -n fastconvmae python=3.7
conda activate fastconvmae
```

- Install `Pytorch==1.8.0` and `torchvision==0.9.0` with `CUDA==11.1`

```bash
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

- Install `timm==0.3.2`

```bash
pip install timm==0.3.2
```

### Data preparation

You can download the ImageNet-1K [here](https://image-net.org) and prepare the ImageNet-1K follow this format:
```tree data
imagenet
  ├── train
      ├── class1
      │   ├── img1.jpeg
      │   ├── img2.jpeg
      │   └── ...
      ├── class2
      │   ├── img3.jpeg
      │   └── ...
      └── ...
```

### Training
To pretrain FastConvMAE-Base with **multi-node distributed training**, run the following on 1 node with 8 GPUs each (only mask 75% is supported):

```bash
python submitit_pretrain.py \
    --job_dir ${JOB_DIR} \
    --nodes 1 \
    --batch_size 64 \
    --model fastconvmae_convvit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 50 \
    --warmup_epochs 10 \
    --blr 6.0e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR}
```
