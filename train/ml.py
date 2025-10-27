import torch
import inspect


def load_fast_weights(net, fast_parameters):
    for weight, fast in zip(
        net.parameters(), fast_parameters or [None] * len(list(net.parameters()))
    ):
        weight.fast = fast


def compute_gradient(net, fast_parameters, input, label, criterion, ovaloss, weight):
    out_c, out_b = net.c_forward(x=input)
    out_b = out_b.view(out_b.size(0), 2, -1)

    loss = criterion(out_c, label) + ovaloss(out_b, label)
    loss *= weight

    grad = torch.autograd.grad(loss, fast_parameters, create_graph=False, allow_unused=True)
    grad = [g.detach() if g is not None else g for g in grad]

    return grad


def update_fast_weights_reptile(net, grad, meta_lr):
    fast_parameters = []
    for k, weight in enumerate(net.parameters()):
        if grad[k] is not None:
            if weight.fast is None:
                weight.fast = weight - meta_lr * grad[k]
            else:
                weight.fast = weight.fast - meta_lr * grad[k]

        if weight.fast is None:
            fast_parameters.append(weight)
        else:
            fast_parameters.append(weight.fast)

    return fast_parameters


def update_fast_weights_sam(net, grad, meta_lr):
    fast_parameters = []
    for k, weight in enumerate(net.parameters()):
        if grad[k] is not None:
            if weight.fast is None:
                weight.fast = weight + meta_lr * grad[k]
            else:
                weight.fast = weight.fast + meta_lr * grad[k]

        if weight.fast is None:
            fast_parameters.append(weight)
        else:
            fast_parameters.append(weight.fast)

    return fast_parameters


update_methods = {
    "reptile": update_fast_weights_reptile,
    "sam": update_fast_weights_sam,
}


def update_fast_weights(method_name, **kwargs):
    if method_name not in update_methods:
        raise ValueError(f"Unknown method: {method_name}")

    method = update_methods[method_name]

    sig = inspect.signature(method)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    return method(**filtered_kwargs)


def accumulate_meta_grads_reptile(net, meta_lr):
    for weight in net.parameters():
        if weight.fast is not None:
            new_grad = (weight - weight.fast) / meta_lr
            if weight.grad is None:
                weight.grad = new_grad
            else:
                weight.grad += new_grad


def accumulate_meta_grads_arith(net, grad, meta_lr, eta):
    scale = eta / meta_lr
    for weight, g in zip(net.parameters(), grad):
        if g is not None:
            if weight.grad is None:
                weight.grad = g * scale
            else:
                weight.grad += g * scale


def accumulate_meta_grads_noise(net, grad, meta_lr, eta):
    scale = eta / meta_lr
    for weight, g in zip(net.parameters(), grad):
        if g is not None:
            if weight.grad is None:
                weight.grad = g * scale + torch.randn(g.shape, device="cuda")
            else:
                weight.grad += g * scale + torch.randn(g.shape, device="cuda")


def _flatten_grad_list(grad_list):
    buf = []
    for g in grad_list:
        if g is not None:
            buf.append(g.reshape(-1))
    return torch.cat(buf) if buf else torch.zeros(0, device="cpu")


def _normalize_weights(ws, target_sum=1.0, eps=1e-12):
    if isinstance(ws, torch.Tensor):
        w = ws.detach()
        if w.dtype != torch.float32:
            w = w.to(dtype=torch.float32)
    else:
        if len(ws) == 0:
            return torch.empty(0, dtype=torch.float32)

        if isinstance(ws[0], torch.Tensor):
            device = ws[0].device
            w = torch.stack([x.detach().to(device=device, dtype=torch.float32) for x in ws], dim=0)
        else:
            w = torch.tensor(ws, dtype=torch.float32)

    s = w.sum().clamp_min(eps)
    return (w / s) * target_sum


class DomainBayesWeighter:
    def __init__(self, num_domains, ema=0.9, init_var=1.0, device="cpu"):
        self.ema = ema
        self.mean = torch.zeros(num_domains, dtype=torch.float32, device=device)
        self.var = torch.full((num_domains,), float(init_var), dtype=torch.float32, device=device)
        self.num_domains = num_domains
        self.log_domain_scale = None

    def attach_learnable_scale(self, device="cpu", init_log=0.0):
        self.log_domain_scale = torch.nn.Parameter(
            torch.full((self.num_domains,), float(init_log), dtype=torch.float32, device=device)
        )
        return self.log_domain_scale

    def update_and_get_tau_per_step(self, grads_history, step_domains):
        flat = [_flatten_grad_list(g) for g in grads_history]
        device = flat[0].device if flat and flat[0].numel() else "cpu"

        norms = torch.stack(
            [f.norm() if f.numel() > 0 else torch.tensor(0.0, device=device) for f in flat]
        )
        n_steps = len(step_domains)

        with torch.no_grad():
            for s in range(n_steps):
                ds = step_domains[s]
                if len(ds) == 0:
                    continue
                v = norms[s]
                idx = torch.tensor(ds, dtype=torch.long, device=device)

                # Welford-EMA
                delta = v - self.mean.index_select(0, idx)
                new_mean = self.mean.index_select(0, idx) + (1 - self.ema) * delta
                new_var = self.var.index_select(0, idx) * self.ema + (1 - self.ema) * delta * (
                    v - new_mean
                )

                self.mean.index_copy_(0, idx, new_mean)
                self.var.index_copy_(0, idx, new_var.clamp_min(1e-12))

        tau_d = 1.0 / self.var.clamp_min(1e-12)

        if self.log_domain_scale is not None:
            tau_d = tau_d * torch.exp(self.log_domain_scale)

        tau_step = []
        for ds in step_domains:
            if len(ds) == 0:
                tau_step.append(torch.tensor(1.0, device=device))
            else:
                tau_step.append(tau_d[torch.tensor(ds, dtype=torch.long, device=device)].mean())
        tau_step = torch.stack(tau_step).to(device)
        return tau_step


def _compute_probabilistic_arith_weights(
    grads_history,
    beta=0.5,
    _weight_noise_state=None,
    target_sum=1.0,
    bayes_weighter=None,
    device=None,
    step_domains=None,
):
    n = len(grads_history)
    if n == 0:
        return torch.ones(0, device=device)

    base = torch.arange(n, 0, -1, dtype=torch.float32, device=device)
    w_base = base / (base.sum().clamp_min(1e-12)) * target_sum

    tau = bayes_weighter.update_and_get_tau_per_step(grads_history, step_domains)
    w = w_base * (tau.clamp_min(1e-12) ** beta)

    return _normalize_weights(w, target_sum=target_sum).to(device)


def accumulate_meta_grads_arith_prob(
    net,
    grads_history,
    meta_lr,
    scaled_factor=3.0,
    bayes_ema=0.9,
    beta=0.5,
    _bayes_state=dict(),
    step_domains=None,
    domain_count=None,
):
    if not grads_history:
        return

    device = next(net.parameters()).device
    target_sum = 1.0 * scaled_factor

    if "bw" not in _bayes_state:
        _bayes_state["bw"] = DomainBayesWeighter(domain_count, ema=bayes_ema, device=device)
    bw = _bayes_state["bw"]

    w = _compute_probabilistic_arith_weights(
        grads_history,
        beta=beta,
        target_sum=target_sum,
        bayes_weighter=bw,
        step_domains=step_domains,
        device=device,
    )

    for step_idx, grad in enumerate(grads_history):
        scale = w[step_idx] / meta_lr
        for weight, g in zip(net.parameters(), grad):
            if g is None:
                continue
            if weight.grad is None:
                weight.grad = g * scale
            else:
                weight.grad += g * scale


def accumulate_meta_grads_ours(
    net,
    grads_history,
    meta_lr,
    scaled_factor=3.0,
    bayes_ema=0.9,
    beta=0.5,
    _bayes_state=dict(),
    step_domains=None,
    domain_count=None,
):
    if not grads_history:
        return

    device = next(net.parameters()).device
    target_sum = 1.0 * scaled_factor

    if "bw" not in _bayes_state:
        _bayes_state["bw"] = DomainBayesWeighter(domain_count, ema=bayes_ema, device=device)
    bw = _bayes_state["bw"]

    w = _compute_probabilistic_arith_weights(
        grads_history,
        beta=beta,
        target_sum=target_sum,
        bayes_weighter=bw,
        step_domains=step_domains,
        device=device,
    )

    for step_idx, grad in enumerate(grads_history):
        scale = w[step_idx] / meta_lr
        for weight, g in zip(net.parameters(), grad):
            if g is None:
                continue
            if weight.grad is None:
                weight.grad = g * scale + torch.randn(g.shape, device="cuda")
            else:
                weight.grad += g * scale + torch.randn(g.shape, device="cuda")


accumulate_methods = {
    "reptile": accumulate_meta_grads_reptile,
    "arith": accumulate_meta_grads_arith,
    "noise": accumulate_meta_grads_noise,
    "arith_prob": accumulate_meta_grads_arith_prob,
    "ours": accumulate_meta_grads_ours,
}


def accumulate_meta_grads(method_name, **kwargs):
    if method_name not in accumulate_methods:
        raise ValueError(f"Unknown method: {method_name}")

    method = accumulate_methods[method_name]

    sig = inspect.signature(method)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    return method(**filtered_kwargs)
