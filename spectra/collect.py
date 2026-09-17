"""Momentum-spectrum collection for the Section 2.1 study.

Patches Muon so that, at chosen optimizer steps, the exact post-Nesterov momentum
matrices the polar factorizer consumes (A = grad + beta*(beta*buf + grad), the
matrix NS Frobenius-normalizes) are saved to disk for the mid-depth layer
(h.{n_layer//2}), and singular-value quantiles of those same matrices are tracked
every few steps for the whole run (the quantile-stabilization view of
arXiv:2606.04058, Sec. 3.1).

Collected per layer: attn.c_attn.weight (dumped whole; its three row blocks
[0:E]=Q, [E:2E]=K, [2E:3E]=V are the matrices Q, K, V that Muon orthogonalizes,
each with its own Frobenius divisor, and the plot scripts take them apart),
attn.c_proj.weight (O), mlp.c_fc.weight, mlp.c_proj.weight.

Read-only by construction: computes A out-of-place from grad and the momentum
buffer BEFORE the real step mutates it, touches no RNG, writes on rank 0 only.
The training trajectory is bit-identical to an unpatched run.

Environment variables:
  DEFMUON_SPECTRA_DIR     output root            (default: spectra/dumps)
  DEFMUON_SPECTRA_TAG     run subfolder          (default: inferred from n_embd)
  DEFMUON_SPECTRA_STEPS   optimizer steps to dump (default: "1,500,1000,1500";
                          step 1 has buf=0, so A = (1+beta)*grad, i.e. the pure
                          gradient direction -- the "step 0")
  DEFMUON_SPECTRA_QSTEP   quantile cadence in optimizer steps, 0 = off (default 10)

Activated through spectra/run_hydra_spectra.py.
"""

import json
import os
import re

import torch

SPECTRA_DIR = os.environ.get("DEFMUON_SPECTRA_DIR", "spectra/dumps")
DUMP_STEPS = {int(s) for s in
              os.environ.get("DEFMUON_SPECTRA_STEPS", "1,500,1000,1500").split(",")}
QSTEP = int(os.environ.get("DEFMUON_SPECTRA_QSTEP", "10"))
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)

# n_embd -> human tag, used when DEFMUON_SPECTRA_TAG is not set
_EMBD_TAG = {1280: "gpt-large"}

_INSTALLED = [False]


def _master():
    return int(os.environ.get("RANK", "0") or 0) <= 0


def install():
    if _INSTALLED[0]:
        return
    _INSTALLED[0] = True
    from defmuon.optim.muon import Muon

    orig_init = Muon.__init__
    orig_step = Muon.step

    def init_with_capture(self, named_params, *args, **kwargs):
        named = list(named_params)          # tee: the original init consumes it too
        orig_init(self, named, *args, **kwargs)
        layers = [int(m.group(1)) for name, _ in named
                  for m in [re.search(r"\.h\.(\d+)\.", name)] if m]
        if not layers:
            self._spectra_targets = []
            return
        mid = (max(layers) + 1) // 2
        suffixes = {f"h.{mid}.attn.c_attn.weight": "attn_qkv",
                    f"h.{mid}.attn.c_proj.weight": "attn_o",
                    f"h.{mid}.mlp.c_fc.weight": "mlp_fc",
                    f"h.{mid}.mlp.c_proj.weight": "mlp_proj"}
        self._spectra_targets = [(lbl, p) for name, p in named
                                 for sfx, lbl in suffixes.items()
                                 if name.endswith(sfx)]
        n_embd = min(min(p.shape) for _, p in self._spectra_targets)
        tag = os.environ.get("DEFMUON_SPECTRA_TAG") or \
            _EMBD_TAG.get(n_embd, f"nembd{n_embd}")
        self._spectra_out = os.path.join(SPECTRA_DIR, tag)
        self._spectra_qrecs = []
        if _master():
            os.makedirs(self._spectra_out, exist_ok=True)
            print(f"[spectra] mid layer h.{mid}: tracking "
                  f"{[lbl for lbl, _ in self._spectra_targets]} -> "
                  f"{self._spectra_out} (dump at {sorted(DUMP_STEPS)}, "
                  f"quantiles every {QSTEP} steps)")

    def _momentum_A(self, p):
        """The post-Nesterov matrix the factorizer will consume this step,
        computed out-of-place (the real step's in-place buf update happens later)."""
        group = next(g for g in self.param_groups if any(q is p for q in g["params"]))
        mom = group["momentum"]
        g32 = p.grad.float()
        buf = self.state[p].get("momentum_buffer")
        newbuf = g32 if buf is None else buf.float().mul(mom).add_(g32)
        return g32 + mom * newbuf if group["nesterov"] else newbuf

    def step_with_spectra(self, closure=None):
        n = getattr(self, "_spectra_n", 0) + 1      # this call = optimizer step n
        self._spectra_n = n
        targets = getattr(self, "_spectra_targets", None)
        if _master() and targets:
            if n in DUMP_STEPS:
                for lbl, p in targets:
                    if p.grad is None:
                        continue
                    A = _momentum_A(self, p)
                    torch.save({"step": n, "label": lbl, "A": A.cpu(),
                                "momentum": self.defaults["momentum"],
                                "nesterov": self.defaults["nesterov"]},
                               os.path.join(self._spectra_out,
                                            f"step{n:04d}_{lbl}.pt"))
                print(f"[spectra] dumped {len(targets)} matrices at step {n}")
            if QSTEP and n % QSTEP == 0:
                for lbl, p in targets:
                    if p.grad is None:
                        continue
                    A = _momentum_A(self, p)
                    # Q/K/V are separate matrices, so quantiles are recorded per
                    # block, each with its own norm
                    if lbl == "attn_qkv":
                        E = A.shape[1]
                        blocks = [("attn_q", A[:E]), ("attn_k", A[E:2 * E]),
                                  ("attn_v", A[2 * E:])]
                    else:
                        blocks = [(lbl, A)]
                    for bl, B in blocks:
                        fro = B.norm().item()
                        s = torch.linalg.svdvals(B)      # descending
                        r = s.numel()
                        qv = {str(q): (s[min(int(-(-q * r // 1)), r) - 1]
                                       / (fro + 1e-30)).item() for q in QUANTILES}
                        self._spectra_qrecs.append(
                            {"step": n, "label": bl, "fro": fro,
                             "sigma_max": s[0].item(), "q": qv})
                with open(os.path.join(self._spectra_out, "quantiles.json"),
                          "w") as f:
                    json.dump(self._spectra_qrecs, f)
        return orig_step(self, closure)

    Muon.__init__ = init_with_capture
    Muon.step = step_with_spectra
