# CoopVAEBM:  Cooperative Training of Energy-based Model and Variational Auto-Encoder

Cooperative learning of energy-based model and variational auto-encoder

This repository contains a pytorch implementation for the paper "[Learning energy-based model with variational auto-encoder as amortized sampler](https://arxiv.org/pdf/2012.14936.pdf)" AAAI 2021.


## Installation


```bash
conda create --name coopvaebm python=2.7
conda activate coopvaebm
conda install tensorflow-gpu==1.12.0
conda install Pillow    
```

Download the cifar10 checkpoint from [here](http://www.stat.ucla.edu/~jxie/CoopVAEBM/coopvaebm_file/code/checkpoints.zip)

## Exp 1: Training


(1) Cifar10 dataset

```bash
python main.py --test False
```

(2) MNIST dataset

```bash
python main.py --net_type 'mnist' --category 'mnist' --image_size 28 --num_channel 1 --batch_size 400 --nTileRow 20 --nTileCol 20 --des_step_size 0.001 --des_sample_steps 50 --vae_lr 0.0001 --weight_latent_loss 3
```

(3) MNIST-fashion dataset

```bash
python main.py --net_type 'mnist' --category 'mnist-fashion' --image_size 28 --num_channel 1 --batch_size 400 --nTileRow 20 --nTileCol 20 --des_step_size 0.001 --des_sample_steps 50 --vae_lr 0.0001 --weight_latent_loss 3
```

## Exp 2: Synthesis with a pretrained model



```bash
python main.py --test True --test_type 'syn' --ckpt 'pretrained/checkpoints/cifar/model.ckpt-3000'
```
<p align="center"><img src="/demo/syn.png" width="700px"/></p>

## Exp 3: Interpolation with a pretrained model


```bash
python main.py --test True --test_type 'inter' --ckpt 'pretrained/checkpoints/cifar/model.ckpt-3000'
```

<p align="center"><img src="/demo/interp.png" width="700px"/></p>

## References
    @inproceedings{xie2021learning,
        title={Learning energybased model with variational auto-encoder as amortized sampler},
        author={Xie, Jianwen and Zheng, Zilong and Li, Ping},
        booktitle={The Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI)},
        year={2021}
    }
