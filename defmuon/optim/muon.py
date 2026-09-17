## Muon code from Moonlight
## https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py

# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
# via GPT-opt: https://github.com/modichirag/GPT-opt (polar branch).
# Added here: the batched deflation front-end integration (_muon_step_batched)
# and the construction-time compile warm-up (_warm_polar_compile).
import torch
import math
import os
from defmuon.optim.polar_express import PolarExpress
from defmuon.optim.deflation_batched import (_DeflatedPolarBase,
                                      _GATE_ACC, _TIMING_ON, _PENDING)

@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) >= 2
    a, b, c = (3.4445, -4.7750, 2.0315) 
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Distributed use: every rank holds the full parameter set and computes the
    identical update from the DDP-synchronized gradients (no optimizer-state
    sharding).  The deflated path's only randomness is seeded from the shared
    step counter, so replicas stay bit-identical without extra communication.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        polar_args: A dictionary of additional arguments to pass to the polar factorization method.
    """
    def __init__(self,
                 named_params,
                 lr=1e-3,
                 weight_decay=0,
                 momentum=0.95,
                 nesterov=True,
                 ns_steps=5,
                 rms_scaling=True,
                 nuclear_scaling=False,
                 polar_method="ns5",
                 adamw_betas=(0.95, 0.95),
                 adamw_eps=1e-8,
                 polar_args={},
                 batched_deflation=False,
                ):
        """
        Arguments:
            polar_method: The polar factorization method to use: "ns5" (standard
                Muon Newton-Schulz quintic) or "polarexpress".  The early-stage
                deflation method (deflated_ns_cut) is installed by
                defmuon/optim/deflation_cutoff.py via the run_hydra_cutoff.py
                entry point.
        """
        defaults = dict(
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                rms_scaling=rms_scaling,
                nuclear_scaling=nuclear_scaling,
                adamw_betas=adamw_betas,
                adamw_eps=adamw_eps,
        )

        muon_params, muon_params_names = [], []
        adamw_params, adamw_params_names = [], []
        for name, p in named_params:
            if p.ndim >= 2 and not any(excluded in name for excluded in ["embeddings", "embed_tokens", "wte", "lm_head", "wpe"]):
                muon_params.append(p)
                muon_params_names.append(name)
            else:
                adamw_params.append(p)
                adamw_params_names.append(name)
        params = list(muon_params)
        params.extend(adamw_params)
        super().__init__(params, defaults)
        
        # Sort parameters into those for which we will use Muon, and those for which we will not
        # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
        for p, p_name in zip(muon_params, muon_params_names):
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
            if p_name.endswith("attn.c_attn.weight"):
                self.state[p]["is_W_QKV"] = True   # processed as W_Q, W_K, W_V (see _qkv_split)

        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

        # Instantiate the polar factorization method
        self.polar_factorizer = self._initialize_polar_factorizer(polar_method, polar_args)

        # Deflated arms run their setup through the batched front-end: one call per
        # distinct tall shape rather than one rSVD per matrix.  The flag is derived
        # from the factorizer rather than configured; the `batched_deflation` kwarg is
        # kept only so existing configs keep loading.  deflation_cutoff flips the
        # attribute to False at its switch step (only when DEFMUON_DEFL_CUTOFF is
        # set), together with the factorizer.  The
        # iteration runs once per parameter (a W_QKV iterates its three blocks as
        # one batch).
        self.batched_deflation = isinstance(self.polar_factorizer, _DeflatedPolarBase)
        if self.batched_deflation:
            self._bd_step = 0

        # Pay any @torch.compile'd factorizer's per-shape compilation now, outside
        # the training loop's step_times: otherwise it lands in step 1 and, through
        # val_train_times, biases every time-based comparison against the compiled
        # arm.  DEFMUON_WARM_POLAR=0 disables it.
        if os.environ.get("DEFMUON_WARM_POLAR", "1") == "1":
            self._warm_polar_compile()

    def _qkv_split(self, p, g):
        """A W_QKV gradient (3d x d) is processed as its three square d x d blocks
        W_Q, W_K, W_V; everything else passes through unchanged."""
        if self.state[p].get("is_W_QKV", False):
            return g.reshape(3, g.shape[0] // 3, g.shape[1])
        return g

    def _warm_polar_compile(self):
        """Trigger the polar path's per-(shape, dtype, device) torch.compile at
        construction, so every arm pays its compile before the clock starts.  Zeros
        input: zeros are a fixed point of every iteration here and nothing touches
        the global RNG or any seed counter, so training is unaffected -- only the
        one-off compile latency moves ahead of step 1.

        Plain arms warm the factorizer itself on each muon param shape.  Deflated
        arms warm `_iterate` on each distinct tall bf16 shape, exactly what
        _muon_step_batched feeds it, and run one throwaway batched clip per
        tall-shape group at the true batch size, so the cuSOLVER/cuBLAS first-call
        initialisation is paid here too.  Shapes are taken from _qkv_split, so a
        W_QKV is warmed as its three blocks."""
        fac = self.polar_factorizer
        steps = self.defaults["ns_steps"]
        deflated = isinstance(fac, _DeflatedPolarBase)
        try:
            if deflated:
                # Tall-shape groups at their TRUE batch sizes -- exactly the stacks
                # _muon_step_batched builds every step (a W_QKV contributes its
                # three square blocks, so QKV and W_O share a group).
                groups = {}
                iter_shapes = set()
                dev = None
                for group in self.param_groups:
                    for p in group["params"]:
                        if self.state[p].get("use_muon"):
                            A = self._qkv_split(p, p)
                            nb = A.shape[0] if A.ndim == 3 else 1
                            n, m = max(A.shape[-2:]), min(A.shape[-2:])
                            groups[(n, m)] = groups.get((n, m), 0) + nb
                            iter_shapes.add((nb, n, m))
                            dev = p.device
                for nb, n, m in sorted(iter_shapes):
                    # the compiled iteration, per (batch, tall bf16 shape) as the
                    # step feeds it -- one call per parameter, batch = its block count
                    fac._iterate(torch.zeros((nb, n, m), dtype=torch.bfloat16,
                                             device=dev), steps)
                for (n, m), cnt in sorted(groups.items()):
                    # One clip per group at the true batch size, to pay the
                    # cuSOLVER/cuBLAS first-call init and the allocator's first big
                    # block before the clock.  randn from a dedicated generator: the
                    # global RNG stays untouched, and random (not zeros) input keeps
                    # the CholeskyQR Gram well-conditioned, exercising the same
                    # kernels the production step runs.
                    g_ = torch.Generator(device=dev)
                    g_.manual_seed(0)
                    A = torch.randn((cnt, n, m), dtype=torch.float32, device=dev,
                                    generator=g_)
                    fac.BATCHED_CLIP(A, generator=g_, pad=fac.PAD,
                                     out_dtype=torch.bfloat16)
                    del A
                if groups:
                    print(f"warm_polar: pre-compiled the deflated iteration and "
                          f"pre-warmed the clip for {len(groups)} tall-shape "
                          f"groups at construction")
            else:
                seen = set()
                for group in self.param_groups:
                    for p in group["params"]:
                        if not self.state[p].get("use_muon"):
                            continue
                        shape = tuple(self._qkv_split(p, p).shape)
                        key = (shape, p.dtype, p.device)
                        if key not in seen:
                            seen.add(key)
                            fac(torch.zeros(shape, dtype=p.dtype,
                                            device=p.device), steps)
                if seen:
                    print(f"warm_polar: pre-compiled the polar factorizer for "
                          f"{len(seen)} shapes at construction")
        except Exception as e:
            print(f"warm_polar: warm-up failed ({e}); step 1 pays the compile "
                  f"as before")

    def _initialize_polar_factorizer(self, polar_method, polar_args):
        """Initialize the polar factorization method based on the provided name and parameters."""
        if polar_method == "ns5":
            return zeropower_via_newtonschulz5  # Use the method directly
        elif polar_method == "polarexpress":
            return PolarExpress
        else:
            raise ValueError(f"Unknown polar method: {polar_method}")

    def _muon_step_batched(self, group, params, lr, weight_decay, momentum):
        """The Muon update with the deflation setup batched per tall shape.

        Phase 1 computes every param's Nesterov momentum exactly as the sequential
        loop does.  Phase 2 stacks them by tall shape, runs one batched deflation
        front-end call per shape, ticks the gate counters from the returned
        per-matrix k on device, then runs the factorizer's per-matrix iteration.
        The entry is safe by construction, so there is no per-matrix recovery
        fallback; see deflation_batched's module docstring.  Deterministic across
        data-parallel replicas: param order, group order and the sketch seed are
        functions of the step counter.
        """
        fac = self.polar_factorizer
        ns_steps = group["ns_steps"]
        entries = []                    # one per parameter; QKV carries 3 blocks
        for p in params:
            if p.grad is None:
                continue
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p.grad)
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(p.grad)
            g = p.grad.add(buf, alpha=momentum) if group["nesterov"] else buf
            A = self._qkv_split(p, g).float()
            if A.ndim == 2:
                A = A[None]
            tr = A.shape[-2] < A.shape[-1]
            if tr:
                A = A.mT
            entries.append({"p": p, "g": g, "A": A, "tr": tr, "nb": A.shape[0]})

        shapes = {}
        for e in entries:
            shapes.setdefault(tuple(e["A"].shape[-2:]), []).append(e)

        self._bd_step += 1
        for gi, (shape, es) in enumerate(sorted(shapes.items())):
            A = torch.cat([e["A"] for e in es], dim=0)
            gen = torch.Generator(device=A.device)
            gen.manual_seed((self._bd_step * 1009 + gi * 9176867) % (2 ** 62))
            timed = _TIMING_ON[0] and A.is_cuda
            if timed:
                e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
                e0.record()
            # bf16 output straight from the clip: every iteration consumes bf16,
            # so this is output-identical and skips a full fp32 round-trip
            X0, kvec, _ = fac.BATCHED_CLIP(A, generator=gen, pad=fac.PAD,
                                           out_dtype=torch.bfloat16)
            if timed:
                e1.record()
            # Gate stats accumulate on device; polar_stats() reads them back at val
            # steps, so diagnostics never stall the timed path.
            if _GATE_ACC[0] is None or _GATE_ACC[0].device != kvec.device:
                _GATE_ACC[0] = torch.zeros(3, dtype=torch.int64, device=kvec.device)
            _GATE_ACC[0] += torch.stack([
                torch.tensor(A.shape[0], dtype=torch.int64, device=kvec.device),
                (kvec > 0).sum(), kvec.sum().to(torch.int64)])
            off = 0
            for e in es:
                e["u"] = fac._iterate(X0[off:off + e["nb"]], ns_steps)
                off += e["nb"]
            if timed:
                e2.record()
                _PENDING.append((e0, e1, e2))
            for e in es:
                p, g = e["p"], e["g"]
                u = (e["u"].mT if e["tr"] else e["u"]).reshape(p.shape).bfloat16()
                adjusted_lr = self.adjust_lr_for_muon(
                    lr, group["rms_scaling"], group["nuclear_scaling"],
                    p.shape, g.bfloat16(), u)
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(u, alpha=-adjusted_lr)

    def adjust_lr_for_muon(self, lr, rms_scaling, nuclear_scaling, param_shape, grad, grad_sign):
        scale = 1.0
        if rms_scaling:
            fan_out, fan_in = param_shape[:2]
            scale *= math.sqrt(fan_out / fan_in)
        if nuclear_scaling:
            scale *= torch.trace(grad.T @ grad_sign)
        return lr * scale

    def step(self, closure=None):
        """Perform a single optimization step.
            Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
"""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                        
        for group in self.param_groups:
            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            if self.batched_deflation and params:
                self._muon_step_batched(group, params, lr, weight_decay, momentum)
                params = []

            # generate weight updates in distributed fashion
            for p in params:
                g = p.grad
                if g is None:
                    continue

                assert g is not None
                
                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                old_shape = g.shape
                g = self._qkv_split(p, g)

                # Use the selected polar factorization method
                u = self.polar_factorizer(g, group["ns_steps"])

                g = g.reshape(old_shape)
                u = u.reshape(old_shape)

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(
                    lr,
                    group["rms_scaling"],
                    group["nuclear_scaling"],
                    p.shape,
                    g.bfloat16(),  # convert to bfloat16 to be compatible with u
                    u
                )
                
                # apply weight decay
                p.data.mul_(1 - lr * weight_decay)
                
                # apply update
                p.data.add_(u, alpha=-adjusted_lr)
                
            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["weight_decay"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)
                    
        return loss



