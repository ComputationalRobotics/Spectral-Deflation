"""Entry point for the Section 2.1 spectral-collection runs.

Installs spectra/collect.py (dumps the mid-layer post-Nesterov momentum matrices
at chosen steps and tracks singular-value quantiles), then executes run_hydra.py
UNMODIFIED via runpy, same pattern as run_hydra_cutoff.py.  Meant for the plain
ns5 arm; the patch is read-only, so the trajectory equals an unpatched run.
"""

import os
import runpy
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from spectra.collect import install

install()

if __name__ == "__main__":
    runpy.run_path(os.path.join(_ROOT, "run_hydra.py"), run_name="__main__")
