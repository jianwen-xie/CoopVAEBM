# CoopVAEBM
Cooperative learning of energy-based model and variational auto-encoder

## Installation


```bash
conda create --name coopvaebm python=2.7
conda activate coopvaebm
conda install tensorflow-gpu==1.12.0
conda install Pillow    
```
    
## Exp 1: Training


```bash
python main.py --test False
```

## Exp 2: Synthesis with a pretrained model


```bash
python main.py --test True --test_type 'syn' --ckpt 'pretrained/checkpoints/cifar/model.ckpt-3000'
```

## Exp 3: Interpolation with a pretrained model


```bash
python main.py --test True --test_type 'inter' --ckpt 'pretrained/checkpoints/cifar/model.ckpt-3000'
```

