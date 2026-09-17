"""Entry point for the early-stage-only deflation experiments.

Installs deflation_cutoff (adds the deflated_ns_cut / deflated_pe_cut polar methods:
deflated Muon for the whole run by default, or for the first DEFMUON_DEFL_CUTOFF
optimizer steps and the plain path afterwards when that variable is set),
then executes run_hydra.py UNMODIFIED via runpy (hydra's config_path resolution
requires run_hydra.py to load as the main script; the patches live in sys.modules
and survive).
"""

import os
import runpy

from defmuon.optim.deflation_cutoff import install

install()

if __name__ == "__main__":
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "run_hydra.py"), run_name="__main__")
