This is the official repository of the paper "Sparse Diffusion Models for Multi-annotator Medical Image Segmentation".

## Data preparations
```
data/ # the root of the data folders
    brain_growth/
    brain_tumor/
    kidney/
    prostate/
```

All datasets were taken from [Quantification of Uncertainties in Biomedical Image Quantification Challenge (QUBIQ) 2021](https://qubiq.grand-challenge.org/).
Download the datasets from the following [link](https://qubiq21.grand-challenge.org/participation/).

The datasets should have the following format
```
<dataset_name>/
    train/*
    val/*
```

## Train and Evaluate

Environment:
torch-1.12.1+cu113
torchvision-0.13.1+cu113

Install sp_avg:

```
cd improved_diffusion/sige_avg
pip install -e .
```

Training script example:

```
CUDA_VISIBLE_DEVICES=0 python image_train.py --dataname "brain_tumor" \
    --save_interval 5000 --batch_size 4 --lr 0.00002 --diffusion_steps 100 --consensus_training True \
    --n_gen 25 --log_interval 100 --predict_xstart True --learn_sigma False --use_fp16 False \
    --data_dir "./data/" --out_dir "./logs/"
```

Evaluation script example:
```
CUDA_VISIBLE_DEVICES=0 python image_sample.py --use_ddim False --dataname "brain_tumor" --model_path <path-for-model-weights> --data_dir "./data/" --use_sparse True --use_bg True --cal_bgnum True --overlap_w 0.1 --cut_padding 2 --n_gen 25
```