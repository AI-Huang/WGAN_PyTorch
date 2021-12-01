# WGAN_PyTorch

WGAN, iWGAN (WGAN-GP), and DCGAN implementations with PyTorch.

## Dataset

1. Use train set of CIFAR-10 for our GAN experiment.
2. Do data preprocessing using transforms as in `dataloader.py` which is different with [1].

```python
`dataloader.py`
transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             ])
```

## Acknowledgement

Thanks to

- [1] https://github.com/eriklindernoren/PyTorch-GAN
