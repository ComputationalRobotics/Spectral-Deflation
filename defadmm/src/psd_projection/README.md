# psd_projection (reference filter)

The GPU PSD-projection library of Kang et al. (arXiv:2507.09165), copied unchanged from https://github.com/ComputationalRobotics/psd_projection:
`check.h utils.{h,cu}` (error macros, helpers), `lanczos.{h,cu}` (`approximate_two_norm`,
the scaling bound), `composite_FP16.{h,cu}` (the 7-stage FP16 composite filter = the
factorization-free projection, baseline16), `composite_FP32.{h,cu}` (baseline32),
`lobpcg.{h,cu}` (used internally by composite_FP16.cu).  Included as
`#include "psd_projection/<file>.h"` with `-I src`.
