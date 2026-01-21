import torch
import torch.nn.functional as F
from codebase import utils, setup

def generate_UAPs(
        target_model: torch.nn.Module,
        x_arr: torch.Tensor,
        y_arr: torch.Tensor,
        UAP_target: int,
        UAP_l_inf: float,
        exp_cfg,
) -> torch.Tensor:
    device = exp_cfg.device
    model = target_model.to(device)
    model.eval()

    #make sure input is float32
    x_arr = x_arr.clone().detach().to(device).to(torch.float32)
    y_arr = y_arr.to(device)

    uap = torch.zeros_like(x_arr[0]).to(torch.float32).to(device)
    momentum_vec = torch.zeros_like(uap)

    #cifar10 normalisation
    mean = torch.tensor(setup.CIFAR10_MEAN, dtype=torch.float32).reshape([1, 3, 1, 1]).to(device)
    std = torch.tensor(setup.CIFAR10_STD, dtype=torch.float32).reshape([1, 3, 1, 1]).to(device)
    normalise = lambda x: utils.normalize(x, mean, std)

    max_iters = 500
    step_size = UAP_l_inf / 4
    batch_size = 20
    momentum = 0.75

    for step in range(max_iters):
        grad_total = torch.zeros_like(uap)

        #check which inputs are not fooled yet
        with torch.no_grad():
            x_perturbed = torch.clamp(x_arr + uap, 0.0, 1.0)
            preds = model(normalise(x_perturbed)).argmax(dim=1)
            not_fooled = (preds != UAP_target).nonzero(as_tuple=True)[0]

        if len(not_fooled) == 0:
            break

        #go through in batches
        for i in range(0, len(not_fooled), batch_size):
            batch_idx = not_fooled[i:i + batch_size]
            batch_x = x_arr[batch_idx]
            x_adv = torch.clamp(batch_x + uap, 0.0, 1.0).detach().requires_grad_(True)

            logits = model(normalise(x_adv))
            target = torch.full((len(batch_idx),), UAP_target, device=device)
            loss = F.cross_entropy(logits, target)

            model.zero_grad()
            loss.backward()

            grad = x_adv.grad.data.mean(dim=0)
            grad = grad / (grad.abs().mean() + 1e-8)
            grad_total += grad

        #update with momentum
        momentum_vec = momentum * momentum_vec + grad_total
        uap = uap - step_size * momentum_vec.sign()
        uap = torch.clamp(uap, -UAP_l_inf, UAP_l_inf)

        #decay step size every 100 steps
        if step % 100 == 0 and step > 0:
            step_size *= 0.9

    return uap.detach()
