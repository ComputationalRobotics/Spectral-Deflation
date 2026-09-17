"""Deflated Muon arms with an OPTIONAL early-stage cutoff.

Default (DEFMUON_DEFL_CUTOFF unset): deflation runs for the WHOLE run -- the
benchmark protocol.  With DEFMUON_DEFL_CUTOFF=N set, deflation
runs for the first N optimizer steps and then hard-switches to the plain path
(used only by experiments/cutoff_ablation.sbatch).

After a switch the factorizer IS zeropower_via_newtonschulz5 / PolarExpress
(the same compiled function the plain arm dispatches to) and batched_deflation
is disabled, so the late step is byte-for-byte the baseline step.  The switch
triggers on a per-process optimizer-step counter, identical across DDP ranks,
so all replicas flip at the same step; momentum buffers pass through untouched.

Activated only through run_hydra_cutoff.py.  The "_cut" suffix of the arm
names marks the arms that CAN take a cutoff; with the default they never
switch.

    polar methods   "deflated_ns_cut" (plain path ns5), "deflated_pe_cut"
                    (plain path polarexpress, pad 1.1 via DEFMUON_DEFL_PE_PAD)
    cutoff          DEFMUON_DEFL_CUTOFF optimizer steps; unset = never switch
"""

import os

import torch

from defmuon.optim.deflation_batched import DeflatedNSPolar, DeflatedPEPolar

_env = os.environ.get("DEFMUON_DEFL_CUTOFF")
CUTOFF_STEPS = int(_env) if _env else None   # None = full-run deflation
_INSTALLED = [False]


class CutoffDeflatedNSPolar(DeflatedNSPolar):
    """Plain deflated_ns; the class only marks the arm for the step-counter switch."""
    IS_CUTOFF_ARM = True
    POST = "ns5"


class CutoffDeflatedPEPolar(DeflatedPEPolar):
    """Deflated Polar Express; hard-switches to the plain PolarExpress path."""
    IS_CUTOFF_ARM = True
    POST = "polarexpress"


def install():
    if _INSTALLED[0]:
        return
    _INSTALLED[0] = True
    from defmuon.optim.muon import Muon, zeropower_via_newtonschulz5
    from defmuon.optim.polar_express import PolarExpress

    post_fns = {"ns5": zeropower_via_newtonschulz5,
                "polarexpress": PolarExpress}

    orig_init = Muon._initialize_polar_factorizer

    def warm_post(muon, post):
        """Trigger the post-cutoff factorizer's per-shape torch.compile at
        construction, so the step-time series stays spike-free at CUTOFF_STEPS.

        Zeros input on purpose: torch.randn would consume the GLOBAL RNG and
        desynchronise this arm's random stream from the other arms'.
        """
        fn = post_fns[post]
        steps = muon.defaults["ns_steps"]
        seen = set()
        for group in muon.param_groups:
            for p in group["params"]:
                if not muon.state[p].get("use_muon"):
                    continue
                # warm the shape the post-cutoff plain loop will actually feed
                # (a W_QKV iterates as (3, d, d))
                shape = tuple(muon._qkv_split(p, p).shape)
                key = (shape, p.dtype, p.device)
                if key not in seen:
                    seen.add(key)
                    fn(torch.zeros(shape, dtype=p.dtype, device=p.device), steps)
        print(f"[deflation_cutoff] warmed {post} compile for {len(seen)} shapes")

    def init_with_cutoff(self, polar_method, polar_args):
        arm = {"deflated_ns_cut": CutoffDeflatedNSPolar,
               "deflated_pe_cut": CutoffDeflatedPEPolar}.get(polar_method)
        if arm is None:
            return orig_init(self, polar_method, polar_args)
        if CUTOFF_STEPS is None:
            print("[deflation_cutoff] no cutoff set: deflation on for the whole run")
            return arm()                 # nothing to switch to -> no warm-up
        try:
            warm_post(self, arm.POST)
        except Exception as e:
            print(f"[deflation_cutoff] {arm.POST} warm-up failed ({e}); "
                  f"the switch step will pay the compile instead")
        return arm()

    Muon._initialize_polar_factorizer = init_with_cutoff

    orig_step = Muon.step

    def step_with_cutoff(self, closure=None):
        n = getattr(self, "_cut_steps", 0)
        if (CUTOFF_STEPS is not None and n == CUTOFF_STEPS
                and getattr(self.polar_factorizer, "IS_CUTOFF_ARM", False)):
            post = self.polar_factorizer.POST
            self.batched_deflation = False
            self.polar_factorizer = post_fns[post]
            print(f"[deflation_cutoff] optimizer step {n}: deflation OFF -> plain {post}")
        out = orig_step(self, closure)
        self._cut_steps = n + 1
        return out

    Muon.step = step_with_cutoff
