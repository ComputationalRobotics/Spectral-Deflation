// pm_head_test.cu -- single-projection accuracy of the deflated projection on a synthetic symmetric
// matrix with a NEGATIVE head, a POSITIVE head and a dense bulk, FP16 composite of the parent paper.
// Truth is exact by construction: M = Q diag(l) Q^T, Pi_+(M) = Q diag(max(l,0)) Q^T.
// Arms:
//   base : the paper's composite_FP16_auto_scale on M (no deflation)
//   defl : deflated projection -- both heads removed to zero, alpha = Lanczos bound of the remainder, restore at
//          sign(l)/1.01, composite; recovery 0.5*S0*(I+X_T) with the head-ZERO S0; positive head
//          added back exactly in FP64.
// The head directions are taken EXACTLY from Q (Ritz residual 0), so the comparison isolates the
// recovery variant.  Muon's per-sign gate is evaluated on an emulated block (the ceil(0.05 n) largest
// |l|, s_kw := block min when a side has fewer than kw entries) and printed.
// usage: pm_head_test n n_neg n_pos bulk_max neg_lo neg_hi pos_lo pos_hi seed
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_fp16.h>
#include "psd_projection/composite_FP16.h"
#include "psd_projection/lanczos.h"
#include "psd_projection/check.h"
#include "psd_projection/utils.h"
#include "composite_FP16_rec.cuh"

static double dnrm(cublasHandle_t cb, const double* A, size_t nn) {
    double r = 0.0; CHECK_CUBLAS(cublasDnrm2(cb, (int)nn, A, 1, &r)); return r;
}
// S += sign * W diag(d) W^T     (W: n x k on device; d: host, length k; T: n x k scratch; dD: k scratch)
static void rank_k(cublasHandle_t cb, double* S, int n, const double* W, int k,
                   const std::vector<double>& d, double* dD, double* T, double sign) {
    if (k <= 0) return;
    CHECK_CUDA(cudaMemcpy(dD, d.data(), (size_t)k * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUBLAS(cublasDdgmm(cb, CUBLAS_SIDE_RIGHT, n, k, W, n, dD, 1, T, n));
    const double one = 1.0;
    CHECK_CUBLAS(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k, &sign, T, n, W, n, &one, S, n));
}
// ||W^T D W||_F for a block W (n x k)
static double block_norm(cublasHandle_t cb, const double* D, int n, const double* W, int k, double* T, double* P) {
    if (k <= 0) return 0.0;
    const double one = 1.0, zero = 0.0;
    CHECK_CUBLAS(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n, &one, D, n, W, n, &zero, T, n));   // T = D W
    CHECK_CUBLAS(cublasDgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, k, k, n, &one, W, n, T, n, &zero, P, k));   // P = W^T T
    return dnrm(cb, P, (size_t)k * k);
}
static double lambda_min(cusolverDnHandle_t sv, const double* A, int n, double* scratch) {
    const size_t nn = (size_t)n * n;
    CHECK_CUDA(cudaMemcpy(scratch, A, nn * sizeof(double), cudaMemcpyDeviceToDevice));
    double* dE; CHECK_CUDA(cudaMalloc(&dE, (size_t)n * sizeof(double)));
    int lw = 0;
    CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(sv, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
                                               n, scratch, n, dE, &lw));
    double* work; CHECK_CUDA(cudaMalloc(&work, (size_t)lw * sizeof(double)));
    int* info; CHECK_CUDA(cudaMalloc(&info, sizeof(int)));
    CHECK_CUSOLVER(cusolverDnDsyevd(sv, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
                                    n, scratch, n, dE, work, lw, info));
    double lmin = 0.0; CHECK_CUDA(cudaMemcpy(&lmin, dE, sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(dE); cudaFree(work); cudaFree(info);
    return lmin;
}

int main(int argc, char** argv) {
    if (argc < 10) {
        fprintf(stderr, "usage: %s n n_neg n_pos bulk_max neg_lo neg_hi pos_lo pos_hi seed\n", argv[0]);
        return 2;
    }
    const int n = atoi(argv[1]), n_neg = atoi(argv[2]), n_pos = atoi(argv[3]);
    const double bulk_max = atof(argv[4]), neg_lo = atof(argv[5]), neg_hi = atof(argv[6]);
    const double pos_lo = atof(argv[7]), pos_hi = atof(argv[8]);
    const unsigned long seed = strtoul(argv[9], nullptr, 10);
    const size_t nn = (size_t)n * n;
    const int k = n_neg + n_pos;
    const double PAD = 1.01, TAU = 0.10, WIN = 0.025, KFRAC = 0.05;

    cublasHandle_t cb; cusolverDnHandle_t sv;
    CHECK_CUBLAS(cublasCreate(&cb)); CHECK_CUSOLVER(cusolverDnCreate(&sv));

    // ---- spectrum: [0,n_neg) negative head, [n_neg,k) positive head, [k,n) bulk ----------
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<double> lam(n), lam_pos(n);
    for (int i = 0; i < n_neg; ++i) lam[i] = -(neg_lo + (neg_hi - neg_lo) * U(rng));
    for (int i = n_neg; i < k; ++i) lam[i] = pos_lo + (pos_hi - pos_lo) * U(rng);
    for (int i = k; i < n; ++i)     lam[i] = bulk_max * (2.0 * U(rng) - 1.0);
    for (int i = 0; i < n; ++i)     lam_pos[i] = std::max(lam[i], 0.0);

    // ---- Q = orthogonal factor of a Gaussian matrix -------------------------------------
    double* Q; CHECK_CUDA(cudaMalloc(&Q, nn * sizeof(double)));
    {   std::vector<double> hG(nn); std::normal_distribution<double> N01(0.0, 1.0);
        for (auto& x : hG) x = N01(rng);
        CHECK_CUDA(cudaMemcpy(Q, hG.data(), nn * sizeof(double), cudaMemcpyHostToDevice));
        double* tau; CHECK_CUDA(cudaMalloc(&tau, (size_t)n * sizeof(double)));
        int lw1 = 0, lw2 = 0; int* info; CHECK_CUDA(cudaMalloc(&info, sizeof(int)));
        CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(sv, n, n, Q, n, &lw1));
        CHECK_CUSOLVER(cusolverDnDorgqr_bufferSize(sv, n, n, n, Q, n, tau, &lw2));
        double* work; CHECK_CUDA(cudaMalloc(&work, (size_t)std::max(lw1, lw2) * sizeof(double)));
        CHECK_CUSOLVER(cusolverDnDgeqrf(sv, n, n, Q, n, tau, work, lw1, info));
        CHECK_CUSOLVER(cusolverDnDorgqr(sv, n, n, n, Q, n, tau, work, lw2, info));
        cudaFree(tau); cudaFree(work); cudaFree(info); }

    // ---- M, exact projection EX, buffers --------------------------------------------------
    double *M, *EX, *S, *S0, *T, *D, *dD, *P;
    CHECK_CUDA(cudaMalloc(&M,  nn * sizeof(double))); CHECK_CUDA(cudaMalloc(&EX, nn * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&S,  nn * sizeof(double))); CHECK_CUDA(cudaMalloc(&S0, nn * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&T,  nn * sizeof(double))); CHECK_CUDA(cudaMalloc(&D,  nn * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dD, (size_t)n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&P,  (size_t)std::max(k, 1) * std::max(k, 1) * sizeof(double)));
    {   const double one = 1.0, zero = 0.0;
        CHECK_CUDA(cudaMemcpy(dD, lam.data(), (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUBLAS(cublasDdgmm(cb, CUBLAS_SIDE_RIGHT, n, n, Q, n, dD, 1, T, n));
        CHECK_CUBLAS(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &one, T, n, Q, n, &zero, M, n));
        CHECK_CUDA(cudaMemcpy(dD, lam_pos.data(), (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUBLAS(cublasDdgmm(cb, CUBLAS_SIDE_RIGHT, n, n, Q, n, dD, 1, T, n));
        CHECK_CUBLAS(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &one, T, n, Q, n, &zero, EX, n)); }
    const double exn = dnrm(cb, EX, nn);
    const size_t stride = ((nn + 3) / 4) * 4;
    float* wsF; CHECK_CUDA(cudaMalloc(&wsF, 3 * stride * sizeof(float)));
    __half* wsH; CHECK_CUDA(cudaMalloc(&wsH, 3 * stride * sizeof(__half)));

    // ---- Muon's per-sign gate on the emulated block (ceil(0.05 n) largest |l|) -------------
    printf("# pm_head_test n=%d n_neg=%d n_pos=%d bulk<=%.3g neg in [%.3g,%.3g] pos in [%.3g,%.3g] seed=%lu\n",
           n, n_neg, n_pos, bulk_max, neg_lo, neg_hi, pos_lo, pos_hi, seed);
    {   const int K = (int)std::ceil(KFRAC * n), kw = (int)std::ceil(WIN * n);
        std::vector<int> idx(n); for (int i = 0; i < n; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return std::fabs(lam[a]) > std::fabs(lam[b]); });
        const double bmin = std::fabs(lam[idx[std::min(K, n) - 1]]);
        for (int sd = 0; sd < 2; ++sd) {
            std::vector<double> v;
            for (int j = 0; j < std::min(K, n); ++j) { const double l = lam[idx[j]]; if (sd == 0 ? l < 0.0 : l > 0.0) v.push_back(std::fabs(l)); }
            const double s1 = v.empty() ? 0.0 : v[0];
            const double s_kw = v.empty() ? 0.0 : (((int)v.size() >= kw) ? v[kw - 1] : bmin);
            const bool fired = !v.empty() && (s_kw < TAU * s1);
            int kr = 0; for (double x : v) if (x > TAU * s1) ++kr;
            printf("# gate %s side: in block %zu, s1=%.4g s_kw/s1=%.4g -> fired=%d, clip kr=%d (head has %d)\n",
                   sd == 0 ? "neg" : "pos", v.size(), s1, s1 > 0 ? s_kw / s1 : 1.0, (int)fired, kr, sd == 0 ? n_neg : n_pos);
        } }

    // ---- arms -----------------------------------------------------------------------------
    std::vector<double> lam_head(lam.begin(), lam.begin() + k);                 // both heads (exact)
    std::vector<double> lam_ph(lam.begin() + n_neg, lam.begin() + k);           // positive head
    std::vector<double> sgn(k); for (int i = 0; i < k; ++i) sgn[i] = (lam[i] > 0.0 ? 1.0 : -1.0) / PAD;
    const double* Wall = Q;                 // columns [0,k)
    const double* Wneg = Q;                 // columns [0,n_neg)
    const double* Wpos = Q + (size_t)n * n_neg;   // columns [n_neg,k)
    double lam_ph_norm = 0.0; for (double x : lam_ph) lam_ph_norm += x * x; lam_ph_norm = std::sqrt(lam_ph_norm);

    printf("%-6s %-14s %-16s %-16s %-16s %-14s\n", "arm", "relF_err", "negHead_blk", "posHead_blk_rel", "bulk_rest_err", "lambda_min");
    for (int arm = 0; arm < 2; ++arm) {
        const double one = 1.0, minus_one = -1.0;
        CHECK_CUDA(cudaMemcpy(S, M, nn * sizeof(double), cudaMemcpyDeviceToDevice));
        double alpha = 0.0;
        if (arm == 0) {
            composite_FP16_auto_scale(cb, sv, S, n, wsF, wsH);
        } else {
            rank_k(cb, S, n, Wall, k, lam_head, dD, T, -1.0);                    // both heads -> 0
            double lo = 0.0, up = 0.0; approximate_two_norm(cb, sv, S, (size_t)n, &lo, &up);
            alpha = up > 0.0 ? up : 1.0;
            const double inv_a = 1.0 / alpha;
            CHECK_CUBLAS(cublasDscal(cb, (int)nn, &inv_a, S, 1));
            CHECK_CUDA(cudaMemcpy(S0, S, nn * sizeof(double), cudaMemcpyDeviceToDevice));   // head-zero, scaled
            rank_k(cb, S, n, Wall, k, sgn, dD, T, +1.0);                           // restore at sign/pad
            composite_FP16_rec(cb, S, S0, n, wsF, wsH);
            CHECK_CUBLAS(cublasDscal(cb, (int)nn, &alpha, S, 1));
            rank_k(cb, S, n, Wpos, n_pos, lam_ph, dD, T, +1.0);                    // exact positive add-back
        }
        // errors
        CHECK_CUDA(cudaMemcpy(D, S, nn * sizeof(double), cudaMemcpyDeviceToDevice));
        CHECK_CUBLAS(cublasDaxpy(cb, (int)nn, &minus_one, EX, 1, D, 1));          // D = out - EX
        const double relF = dnrm(cb, D, nn) / exn;
        const double nb = block_norm(cb, D, n, Wneg, n_neg, T, P);
        const double pb = block_norm(cb, D, n, Wpos, n_pos, T, P);
        // error outside the head blocks: || D - W (W^T D W) W^T ||_F approximated by sqrt(||D||^2 - ||W^T D W||^2 - 2||D W||^2 + 2||W^T D W||^2)
        // = sqrt(||D||^2 - 2||DW||^2 + ||W^T D W||^2); with T = D W (n x k) from a fresh product
        double rest = 0.0;
        {   const double zero = 0.0;
            CHECK_CUBLAS(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n, &one, D, n, Wall, n, &zero, T, n));
            const double dw = dnrm(cb, T, (size_t)n * k);
            const double wdw = block_norm(cb, D, n, Wall, k, T, P);
            const double dn = dnrm(cb, D, nn);
            rest = std::sqrt(std::max(0.0, dn * dn - 2.0 * dw * dw + wdw * wdw)); }
        const double lmin = lambda_min(sv, S, n, D);
        printf("%-6s %-14.4e %-16.4e %-16.4e %-16.4e %-14.4e\n",
               arm == 0 ? "base" : "defl", relF, nb,
               lam_ph_norm > 0 ? pb / lam_ph_norm : 0.0, rest / exn, lmin);
        if (arm > 0) printf("#   alpha=%.4g  (largest |l| of M ~ %.4g)\n", alpha,
                            std::max(std::fabs(lam[0]), n_pos > 0 ? std::fabs(lam[n_neg]) : 0.0));
    }
    printf("# relF_err: ||out - Pi_+(M)||_F / ||Pi_+(M)||_F.  negHead_blk = ||W-^T (out-Pi) W-||_F (should be ~0);\n"
           "# posHead_blk_rel = ||W+^T (out-Pi) W+||_F / ||diag(l+)||_F;  bulk_rest_err = error outside the head blocks / ||Pi||_F.\n");
    return 0;
}
