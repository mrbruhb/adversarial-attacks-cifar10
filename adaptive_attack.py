import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from codebase import utils, setup

#robust wrapper for defence
class RobustModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.transform = transforms.RandomResizedCrop(size=32, scale=(0.20, 0.50), antialias=True)
        self.num_votes = 21

    def forward(self, x):
        rlt_arr = []
        for _ in range(self.num_votes):
            trans_x = self.transform(x)
            logits = self.model(trans_x)
            preds = torch.argmax(logits, dim=1)
            rlt_arr.append(preds)

        rlt_arr = torch.stack(rlt_arr).transpose(0, 1)
        mode_val, mode_idx = torch.mode(rlt_arr, dim=1)
        final_logits = F.one_hot(mode_val, num_classes=10).float()
        return final_logits

#normalise with cifar10 stats
def preprocess(x, mean_tensor, std_tensor):
    return utils.normalize(x, mean_tensor, std_tensor)

#main attack uses pgd with momentum + eot
def generate_attack(
        target_model: nn.Module,
        x_arr: torch.Tensor,
        y_arr: torch.Tensor,
        adv_target_arr: torch.Tensor,
        adv_l_inf: float,
        exp_cfg,
) -> torch.Tensor:
    device = exp_cfg.device
    model = target_model.to(device)
    model.eval()

    #wrapper to access transform()
    defence = RobustModel(model)

    #prepare data and targets
    x = x_arr.clone().detach().to(device).to(torch.float32)
    targets = adv_target_arr.to(device)

    #attack params
    eps = adv_l_inf
    alpha = eps / 12
    max_iters = 150
    eot_samples = 20
    momentum = 0.75

    #normalisation tensors
    mean = torch.tensor(setup.CIFAR10_MEAN, dtype=torch.float32).reshape([1, 3, 1, 1]).to(device)
    std = torch.tensor(setup.CIFAR10_STD, dtype=torch.float32).reshape([1, 3, 1, 1]).to(device)

    #init adv examples
    x_adv = x + (torch.rand_like(x) * 2 - 1) * eps
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    velocity = torch.zeros_like(x_adv)

    for _ in range(max_iters):
        x_adv.requires_grad = True
        grad_accum = torch.zeros_like(x_adv)

        #eot loop
        for _ in range(eot_samples):
            normed = preprocess(x_adv, mean, std)
            transformed = defence.transform(normed)
            logits = model(transformed)
            loss = F.cross_entropy(logits, targets)

            model.zero_grad()
            loss.backward()
            grad_accum += x_adv.grad.data
            x_adv.grad.zero_()

        #momentum update
        grad = grad_accum / eot_samples
        grad = grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-8)
        velocity = momentum * velocity + grad

        x_adv = x_adv - alpha * velocity.sign()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    return x_adv
