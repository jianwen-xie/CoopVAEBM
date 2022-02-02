# CoopVAEBM
Cooperative learning of energy-based model and variational auto-encoder

This repository contains a pytorch implementation for the paper "[Learning energybased model with variational auto-encoder as amortized sampler](https://arxiv.org/pdf/2012.14936.pdf)" AAAI 2021.


## Installation


```bash
conda create --name coopvaebm python=2.7
conda activate coopvaebm
conda install tensorflow-gpu==1.12.0
conda install Pillow    
```

Download the checkpoint from [here](www.stat.ucla.edu/~jxie/CoopVAEBM/coopvaebm_file/code/checkpoints.zip)

## Exp 1: Training


```bash
python main.py --test False
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
