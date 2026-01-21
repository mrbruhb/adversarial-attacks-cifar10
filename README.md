# Adversarial Attacks on Deep Neural Networks

A collection of adversarial attack implementations targeting CIFAR-10 image classifiers. Demonstrates grey-box attacks, universal perturbations, and adaptive attacks against randomised defences.

## Overview

| Attack | Type | Access Level | Target Model | Use Case |
|--------|------|--------------|--------------|----------|
| **Grey-box PGD** | Targeted | Grey-box | VGG11-BN | Adversarial examples without direct gradient access |
| **Universal Perturbation** | Image-agnostic | White-box | VGG11-BN | Single perturbation fooling multiple inputs |
| **Adaptive Attack** | Defence-aware | White-box | VGG11-BN + RandomCrop defence | Breaking stochastic/randomized defences |

## Target Model

All attacks target a **VGG11 with Batch Normalization** (`vgg11_bn`) trained on CIFAR-10:
- Clean accuracy: **87.22%** on test set
- Input: 32×32×3 images, normalized with CIFAR-10 statistics
- Classes: aeroplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Attack Implementations

### Grey-box PGD Attack (`greybox_attack.py`)

Targeted attack where you **do not have direct access** to compute gradients on the target model. Uses transfer-based techniques with surrogate models accessed via stack frame inspection.

**Constraints:** L∞ = 0.04 on 100 test images

**Key features:**
- 10 random restarts for robustness
- Momentum-accelerated gradient updates (μ = 0.90)
- Logit averaging over final iterations
- Best-sample selection based on target class confidence

```python
adv_x = generate_attack(
    x_arr=images,              # [100, 3, 32, 32] unnormalized
    y_arr=true_labels,         # Original labels
    adv_target_arr=(y_arr+1)%10,  # Target: next class
    adv_l_inf=0.04,
    exp_cfg=config
)
```

### Universal Adversarial Perturbations (`universal_attack.py`)

Generates a **single perturbation** that causes all 100 images to be classified as "cat" (class 3).

**Constraints:** L∞ = 0.06, target class = 3 (cat)

**Key features:**
- Momentum-based gradient accumulation (μ = 0.75)
- Iterative refinement focusing on non-fooled samples
- Step size decay for convergence
- Early stopping when all images are fooled

```python
uap = generate_UAPs(
    target_model=model,
    x_arr=images,           # [100, 3, 32, 32]
    y_arr=labels,
    UAP_target=3,           # Target: cat
    UAP_l_inf=0.06,
    exp_cfg=config
)
# Apply: adv_images = torch.clamp(images + uap, 0, 1)
```

### Adaptive Attack (`adaptive_attack.py`)

Breaks a **randomised defence** that uses random cropping and majority voting. The defence (`RobustModel`) applies `RandomResizedCrop(scale=0.20-0.50)` with 21 votes.

**Constraints:** L∞ = 0.04 on 50 test images

**Key features:**
- Expectation over Transformation (EOT) with 20 samples
- Gradients averaged across stochastic defence passes
- Momentum-based PGD updates
- Simulates defence's random crop during attack

```python
class RobustModel(nn.Module):
    """Defence: random crop + resize + majority vote (21 votes)"""
    def __init__(self, model):
        self.model = model
        self.transform = transforms.RandomResizedCrop(32, scale=(0.20, 0.50))
        self.num_votes = 21

adv_x = generate_attack(
    target_model=model,     # Raw model (not RobustModel)
    x_arr=images[:50],
    y_arr=labels[:50],
    adv_target_arr=targets[:50],
    adv_l_inf=0.04,
    exp_cfg=config
)
```

## Experimental Setup

**Environment:** Google Colab with CUDA GPU

**Dataset:** CIFAR-10 test set
- 100 correctly-classified images selected (seed=375)
- Unnormalized [0,1] for perturbation, normalized for inference

**Normalization:**
```python
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]
```

## Technical Details

### Threat Model

All attacks use the **L∞ threat model**:
- Grey-box & Adaptive: ε = 0.04 (~10/255)
- Universal: ε = 0.06 (~15/255)

Perturbations are applied to unnormalized images [0,1], then normalised before inference.

### Hyperparameters

| Attack | Iterations | Step Size | Momentum | Other |
|--------|------------|-----------|----------|-------|
| Grey-box | 50 × 10 restarts | 0.02 | 0.90 | Logit window: 5 |
| UAP | 500 | ε/4 (decaying) | 0.75 | Batch: 20 |
| Adaptive | 150 | ε/12 | 0.75 | EOT samples: 20 |

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy
```

**Codebase dependencies:**
- `codebase.utils.normalize()` — CIFAR-10 normalization
- `codebase.setup.CIFAR10_MEAN/STD` — Dataset statistics
- `codebase.classifiers.vgg` — VGG model definitions
- `codebase.model_trainer` — Checkpoint loading

## Project Structure

```
├── greybox_attack.py           # Grey-box PGD attack
├── universal_attack.py         # UAP generation
├── adaptive_attack.py          # Adaptive attack (EOT)
├── codebase/                   # Utilities
│   ├── utils.py
│   ├── setup.py
│   ├── model_trainer.py
│   └── classifiers/
│       └── vgg.py
└── out/
    └── target_model/           # Pretrained VGG11-BN weights
```

## References

- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) — Goodfellow et al., 2015 (FGSM)
- [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083) — Madry et al., 2018 (PGD)
- [Universal Adversarial Perturbations](https://arxiv.org/abs/1610.08401) — Moosavi-Dezfooli et al., 2017
- [Obfuscated Gradients Give a False Sense of Security](https://arxiv.org/abs/1802.00420) — Athalye et al., 2018 (EOT)

## License

MIT
