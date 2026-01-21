import torch
import torch.nn.functional as F
import inspect
from codebase import utils, setup

#function to normalize images using cifar10 stats
def preprocess(x, mean_tensor, std_tensor):
    return utils.normalize(x, mean_tensor, std_tensor)

#pgd attack loop with momentum and restart
def run_pgd(model, inputs, targets, alpha, max_iters, eps, preprocess_fn, device):
    clean = inputs.clone().detach()
    best = clean.clone()
    score = torch.full((clean.size(0),), -float("inf")).to(device)

    for _ in range(10):  #number of restarts
        x = clean + (torch.rand_like(clean) * 2 - 1) * eps
        x = torch.clamp(x, 0.0, 1.0).detach()
        velocity = torch.zeros_like(x)
        logits_window = []

        for i in range(max_iters):
            x.requires_grad = True
            logits = model(preprocess_fn(x))
            loss = F.cross_entropy(logits, targets)
            model.zero_grad()
            loss.backward()

            grad = x.grad.data
            grad = grad / (grad.abs().mean(dim=(1,2,3), keepdim=True) + 1e-8)
            velocity = 0.9 * velocity + grad

            x = x - alpha * velocity.sign()
            x = torch.clamp(torch.max(torch.min(x, clean + eps), clean - eps), 0.0, 1.0).detach()

            if i >= max_iters - 5:
                with torch.no_grad():
                    logits_window.append(model(preprocess_fn(x)))

        avg_logits = torch.stack(logits_window).mean(dim=0)
        probs = F.log_softmax(avg_logits, dim=1)
        target_logprob = probs[torch.arange(probs.size(0)), targets]
        preds = avg_logits.argmax(dim=1)
        success = preds == targets
        better = (target_logprob > score) & success

        score[better] = target_logprob[better]
        best[better] = x[better]

    return best

#main attack function, grabs the model and runs pgd
def generate_attack(x_arr, y_arr, adv_target_arr, adv_l_inf, exp_cfg):
    frame = inspect.currentframe()
    while frame:
        if "target_model" in frame.f_globals:
            model = frame.f_globals["target_model"]
            break
        frame = frame.f_back
    else:
        raise NameError("target_model not found.")

    model = model.to(exp_cfg.device)
    model.eval()

    x = x_arr.clone().detach().to(exp_cfg.device).float()
    targets = adv_target_arr.to(exp_cfg.device)

    mean = torch.Tensor(setup.CIFAR10_MEAN).reshape([1, 3, 1, 1]).to(exp_cfg.device)
    std = torch.Tensor(setup.CIFAR10_STD).reshape([1, 3, 1, 1]).to(exp_cfg.device)

    final_adv = run_pgd(
        model=model,
        inputs=x,
        targets=targets,
        alpha=0.02,
        max_iters=50,
        eps=adv_l_inf,
        preprocess_fn=lambda z: preprocess(z, mean, std),
        device=exp_cfg.device
    )

    return final_adv
