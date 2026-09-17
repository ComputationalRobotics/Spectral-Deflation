// Tracking-accuracy study copy of ../src/admm_mc.cu (Appendix D of the paper).  Identical
// to the production solver except for the checkpoint block marked [TRACK]: from iteration
// START on, every DEFADMM_TRACK_EVERY (default 100) iterations the current ADMM matrix M is
// eigendecomposed exactly in FP64 and the Ritz values that the gate deflates,
// I = { i : |lambda_hat_i| > tau * max|lambda_hat| }, are compared with the exact eigenvalues,
// matched by sign and magnitude order:
//     eig_err = max_{i in I} |lambda_hat_i - lambda_i| / |lambda_i|
// Checkpoint lines are printed as "# track it=...".  Checkpoint time is measured separately
// and excluded from the projection time.  The ADMM trajectory is unchanged: the checkpoint
// only reads M and the Ritz values.
//
// Deflated ADMM -- three-step ADMM (Wen, Goldfarb, Yin; the scheme of Kang et al.,
// arXiv:2507.09165 Sec. 4.2) for the Gset SDPs of Mittelmann's sparse SDP collection, with
// the PSD-cone projection done by the FP16 polynomial filter of Kang et al., optionally
// behind Muon-style spectral deflation.
//
//   min <C,X>  s.t. A(X) = b, X >= 0
//     mc (max-cut):        A(X) = diag(X),               b = 1            (m = n)
//     mb (min-bisection):  A(X) = (diag(X), <J,X>),      b = (1, 0)       (m = n+1, J = 11^T)
//   C = -F0 where F0 is the objective matrix of the SDPA file (SDPA's dual maximises <F0,Y>).
//
// ADMM (eq. 24 of the paper):
//   y   = (AA*)^-1 [ sigma^-1 b - A(sigma^-1 X + S - C) ]
//   M   = C - A* y - sigma^-1 X
//   S   = Pi_+(M)                      <- the only O(n^3) step
//   X   = sigma (S - M)              [ = X + sigma (S + A* y - C) ]
// AA* = I for mc; for mb AA* = [I 1; 1^T n^2], inverted in closed form.
//
// Modes
//   baseline32 | baseline16   the reference composite filter (Kang et al.), FP32 / FP16,
//                             unmodified: Lanczos scale, 7-stage composite, recovery.
//   deflated16                the same FP16 filter behind DEFLATION:
//                             a fixed n x k orthonormal block (k = ceil(0.05 n), Muon's
//                             sketch size) is advanced by ONE power step per ADMM
//                             iteration; Muon's gate + clip (window w, threshold tau = 0.1)
//                             are evaluated every iteration from iteration 100 on the
//                             |Ritz values| of the block (the singular values); the clipped
//                             head is removed to zero, the remainder is scaled by the
//                             Lanczos bound, each head direction re-enters at sign/1.1
//                             (Muon's 1/gamma), the ORIGINAL composite runs, the recovery
//                             product 0.5 * S0 (I + X_T) uses the head-ZERO matrix S0, and
//                             the positive head is added back exactly in FP64.
//
// Usage: admm_mc <problem.sdp> <mode> <maxiter> <sigma> [eta_tol=1e-4] [log_every=25]
//                [window=0.025] [gamma=1.1] [symmetrize=1]
//   problem    compact SDP file (data/*.sdp, from scripts/sdpa_to_compact.py): line 1 "n m mc|mb",
//              then the upper-triangle nonzeros "i j val" of F0 (1-based)
//   window     Muon's gate window w (fraction of n); deflated16 only
//
// SYMMETRIZATION.  The reconstruction  P = H_+ + 0.5 * R (I + g(X_0))  is symmetric only up to
// rounding (the recovery product is symmetrized in FP32 inside composite_FP16_rec, the FP64
// rank-k add-back of the positive head rounds (i,j) and (j,i) independently), so the projection
// output P (= S) is replaced by (P + P^T)/2 before the X update (symmetrize=1, the default).
// The relative defect eps_sym = ||P - P^T||_F / ||P||_F is measured on the logged iterates
// before (eps_sym_raw) and after (eps_sym_used) and summarised at the end of the run.
//
// Per-iteration log: pinf / dinf / gap / eta (the eigenvalue-free KKT residual of the
// paper), objectives, projection ms, cumulative projection s, deflated directions,
// eps_sym_raw, eps_sym_used.
// End of run: lambda_min(X), lambda_min(S), rank(X), full eta (FINAL line).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_fp16.h>

#include "psd_projection/composite_FP32.h"
#include "psd_projection/composite_FP16.h"
#include "psd_projection/lanczos.h"
#include "psd_projection/check.h"
#include "composite_FP16_rec.cuh"

__global__ void y_update_kernel(double* y, const double* X, const double* S,
                                const double* C, const double* b,
                                double inv_sigma, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        size_t d = (size_t)i * n + i;
        y[i] = inv_sigma * (b[i] - X[d]) - (S[d] - C[d]);
    }
}
// M = C - Diag(y) - y2 * J - inv_sigma * X      (y2 = 0 for mc)
__global__ void m_build_kernel(double* M, const double* C, const double* X,
                               const double* y, double y2, double inv_sigma, int n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t nn = (size_t)n * n;
    if (idx < nn) {
        double v = C[idx] - inv_sigma * X[idx] - y2;
        if (idx % (n + 1) == 0) v -= y[idx / (n + 1)];
        M[idx] = v;
    }
}
// X = sigma * (S - M)
__global__ void x_update_kernel(double* X, const double* S, const double* M,
                                double sigma, size_t nn) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nn) X[idx] = sigma * (S[idx] - M[idx]);
}
// D = S - C + Diag(y) + y2 * J   (dual-feasibility residual matrix A* y + S - C)
__global__ void dres_kernel(double* D, const double* S, const double* C,
                            const double* y, double y2, int n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t nn = (size_t)n * n;
    if (idx < nn) {
        double v = S[idx] - C[idx] + y2;
        if (idx % (n + 1) == 0) v += y[idx / (n + 1)];
        D[idx] = v;
    }
}
__global__ void diag_res_kernel(double* r, const double* X, const double* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) r[i] = X[(size_t)i * n + i] - b[i];
}
// ---- symmetry defect and symmetrization of the projection output P ----------------------
// acc += sum_{i<j} (P_ij - P_ji)^2, so that ||P - P^T||_F^2 = 2 * acc.  Column-major:
// (i,j) at i + j n.  blockDim.x must be 256.
__global__ void antisym_sq_kernel(const double* P, int n, double* acc) {
    __shared__ double sh[256];
    const size_t nn = (size_t)n * n;
    double s = 0.0;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < nn;
         idx += (size_t)gridDim.x * blockDim.x) {
        const int i = (int)(idx % n), j = (int)(idx / n);
        if (i < j) { const double d = P[idx] - P[(size_t)j + (size_t)i * n]; s += d * d; }
    }
    sh[threadIdx.x] = s; __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(acc, sh[0]);
}
// P <- (P + P^T)/2 in place; one thread per pair i<j writes both entries (race-free), the
// diagonal is untouched.  0.5*(a+b) is computed once, so the result is exactly symmetric.
__global__ void symmetrize_kernel(double* P, int n) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nn = (size_t)n * n;
    if (idx < nn) {
        const int i = (int)(idx % n), j = (int)(idx / n);
        if (i < j) {
            const size_t t = (size_t)j + (size_t)i * n;
            const double v = 0.5 * (P[idx] + P[t]);
            P[idx] = v; P[t] = v;
        }
    }
}

// eigenvalues only (ascending), FP64, for the a-posteriori check
static void sym_eigs(cusolverDnHandle_t sv, const double* dA, int n, std::vector<double>& w) {
    size_t nn = (size_t)n * n;
    double* dW; CHECK_CUDA(cudaMalloc(&dW, nn * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(dW, dA, nn * sizeof(double), cudaMemcpyDeviceToDevice));
    double* dE; CHECK_CUDA(cudaMalloc(&dE, (size_t)n * sizeof(double)));
    int lwork = 0; int* dInfo; CHECK_CUDA(cudaMalloc(&dInfo, sizeof(int)));
    CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(sv, CUSOLVER_EIG_MODE_NOVECTOR,
        CUBLAS_FILL_MODE_LOWER, n, dW, n, dE, &lwork));
    double* work; CHECK_CUDA(cudaMalloc(&work, (size_t)lwork * sizeof(double)));
    CHECK_CUSOLVER(cusolverDnDsyevd(sv, CUSOLVER_EIG_MODE_NOVECTOR,
        CUBLAS_FILL_MODE_LOWER, n, dW, n, dE, work, lwork, dInfo));
    w.resize(n);
    CHECK_CUDA(cudaMemcpy(w.data(), dE, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(dW); cudaFree(dE); cudaFree(work); cudaFree(dInfo);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s problem.sdp baseline32|baseline16|deflated16 maxiter sigma "
                        "[eta_tol=1e-4] [log_every=25] [window=0.025] [gamma=1.1] [symmetrize=1]\n", argv[0]);
        return 2;
    }
    const char* ppath = argv[1];
    const char* mode = argv[2];
    const int maxiter = atoi(argv[3]);
    const double sigma = atof(argv[4]);
    const double eta_tol = (argc > 5) ? atof(argv[5]) : 1e-4;
    const int log_every = (argc > 6) ? atoi(argv[6]) : 25;
    const double window = (argc > 7) ? atof(argv[7]) : 0.025;
    const double gamma_pad = (argc > 8) ? atof(argv[8]) : 1.1;    // restore height 1/gamma (padding)
    const int symmetrize = (argc > 9) ? atoi(argv[9]) : 1;       // P <- (P + P^T)/2 after the projection
    const double inv_sigma = 1.0 / sigma;

    const bool deflate = (strcmp(mode, "deflated16") == 0);
    const bool fp16 = deflate || (strcmp(mode, "baseline16") == 0);
    if (!deflate && !fp16 && strcmp(mode, "baseline32") != 0) {
        fprintf(stderr, "unknown mode '%s' (baseline32 | baseline16 | deflated16)\n", mode);
        return 2;
    }

    // ---- read the compact SDP: "n m mc|mb", then F0's upper-triangle nonzeros; C = -F0 ----
    FILE* fp = fopen(ppath, "r");
    if (!fp) { fprintf(stderr, "cannot open %s\n", ppath); return 2; }
    int n = 0, m = 0; char ptype[8] = {0};
    if (fscanf(fp, "%d %d %7s", &n, &m, ptype) != 3 || n <= 0) { fprintf(stderr, "bad header in %s\n", ppath); return 2; }
    const bool mb = (strcmp(ptype, "mb") == 0);
    if (!mb && strcmp(ptype, "mc") != 0) { fprintf(stderr, "unknown problem type '%s'\n", ptype); return 2; }
    if (m != n + (mb ? 1 : 0)) { fprintf(stderr, "m=%d inconsistent with type %s, n=%d\n", m, ptype, n); return 2; }
    const size_t nn = (size_t)n * n;
    std::vector<double> hC(nn, 0.0);
    long nz = 0;
    {   int i, j; double v;
        while (fscanf(fp, "%d %d %lf", &i, &j, &v) == 3) {
            --i; --j; ++nz;
            hC[(size_t)i * n + j] -= v;                       // C = -F0
            if (i != j) hC[(size_t)j * n + i] -= v;
        } }
    fclose(fp);

    cublasHandle_t cb; cusolverDnHandle_t sv;
    CHECK_CUBLAS(cublasCreate(&cb)); CHECK_CUSOLVER(cusolverDnCreate(&sv));
    double *C, *X, *S, *M, *y, *b, *rdiag;
    CHECK_CUDA(cudaMalloc(&C, nn * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&X, nn * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&S, nn * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&M, nn * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y, (size_t)n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&b, (size_t)n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&rdiag, (size_t)n * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(C, hC.data(), nn * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(X, 0, nn * sizeof(double)));
    CHECK_CUDA(cudaMemset(S, 0, nn * sizeof(double)));
    CHECK_CUDA(cudaMemset(y, 0, (size_t)n * sizeof(double)));
    {   std::vector<double> hb(n, 1.0);
        CHECK_CUDA(cudaMemcpy(b, hb.data(), (size_t)n * sizeof(double), cudaMemcpyHostToDevice)); }
    const double one_c = 1.0, zero_c = 0.0, minus_one_c = -1.0;
    // mb: the extra constraint <J,X> = 0.  ones = 1 (n-vector), tmp = work; 1^T Z 1 via gemv + dot.
    double *ones = nullptr, *tmp = nullptr; double sumC = 0.0, y2 = 0.0;
    auto total_sum = [&](const double* Z) {
        CHECK_CUBLAS(cublasDgemv(cb, CUBLAS_OP_N, n, n, &one_c, Z, n, ones, 1, &zero_c, tmp, 1));
        double sres = 0.0; CHECK_CUBLAS(cublasDdot(cb, n, tmp, 1, ones, 1, &sres)); return sres;
    };
    if (mb) {
        CHECK_CUDA(cudaMalloc(&ones, (size_t)n * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&tmp,  (size_t)n * sizeof(double)));
        std::vector<double> h1(n, 1.0);
        CHECK_CUDA(cudaMemcpy(ones, h1.data(), (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
        sumC = total_sum(C);
    }

    // persistent workspaces for the reference filter (its auto_scale entry points
    // otherwise cudaMalloc per call)
    const size_t stride = ((nn + 3) / 4) * 4;
    float* wsF; CHECK_CUDA(cudaMalloc(&wsF, 3 * stride * sizeof(float)));
    __half* wsH = nullptr;
    if (fp16) CHECK_CUDA(cudaMalloc(&wsH, 3 * stride * sizeof(__half)));

    // =====================================================================
    // Deflation state (deflated16).  Constants are Muon's: tau = 0.1 for the gate and the
    // clip, gamma = 1.1 for the restore height, k = ceil(0.05 n) for the tracked block
    // (Muon's oversampled sketch size), window w from the command line.  The only
    // ADMM-side setting is START: the random block needs a few power steps to converge,
    // and ADMM's head structure grows with the iteration (Muon's is strongest early).
    // =====================================================================
    const double TAU = 0.10, GAMMA = gamma_pad;
    const int    START = 100;
    const int    K = deflate ? (int)std::ceil(0.05 * n) : 0;
    double *dV = nullptr, *dZ = nullptr, *dT = nullptr, *dW = nullptr, *dB = nullptr, *dG = nullptr,
           *dSel = nullptr, *dEig = nullptr, *dS0 = nullptr, *dQRwork = nullptr, *dSyWork = nullptr;
    int *dInfoQ = nullptr; int dQRlw = 0, dSyLw = 0;
    std::vector<double> h_lam;                       // Ritz values of the block (this iteration)
    double theta = 1e300;                            // clip threshold on |lambda| (this iteration)
    int    kr[2] = {0, 0};                           // clipped directions: [0] negative, [1] positive (logging)
    bool   fired_any = false; int n_clipped = 0;
    long   st_evals = 0, st_fired = 0, st_fired_side[2] = {0, 0}, st_restore = 0;
    int    win_cnt = 0, win_fired = 0; double alpha_last = 0.0;
    auto clipped = [&](double lam) { return std::fabs(lam) > theta; };

    // [TRACK] checkpoint state
    const int TRACK_EVERY = getenv("DEFADMM_TRACK_EVERY") ? atoi(getenv("DEFADMM_TRACK_EVERY")) : 100;
    double *dME = nullptr, *dEvals = nullptr, *dEwork = nullptr;
    int *dInfoE = nullptr; int dElw = 0;
    double ckpt_ms_iter = 0.0, ckpt_ms_total = 0.0; long n_ckpt = 0;
    cudaEvent_t c0, c1; cudaEventCreate(&c0); cudaEventCreate(&c1);
    if (deflate) {
        CHECK_CUDA(cudaMalloc(&dME,    nn * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&dEvals, (size_t)n * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&dInfoE, sizeof(int)));
        CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(sv, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
                                                   n, dME, n, dEvals, &dElw));
        CHECK_CUDA(cudaMalloc(&dEwork, (size_t)dElw * sizeof(double)));
    }
    auto is_ckpt = [&](int t) { return deflate && t >= START && (t - START) % TRACK_EVERY == 0; };
    // Called inside deflate_step after the Rayleigh-Ritz step (h_lam valid, M intact).
    auto track_checkpoint = [&](int it) {
        cudaEventRecord(c0);
        // exact FP64 eigenvalues of M (ascending)
        CHECK_CUDA(cudaMemcpy(dME, M, nn * sizeof(double), cudaMemcpyDeviceToDevice));
        CHECK_CUSOLVER(cusolverDnDsyevd(sv, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
                                        n, dME, n, dEvals, dEwork, dElw, dInfoE));
        std::vector<double> w(n);
        CHECK_CUDA(cudaMemcpy(w.data(), dEvals, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
        // exact spectrum: magnitude order, threshold, gate
        std::vector<int> ord(n); for (int i = 0; i < n; ++i) ord[i] = i;
        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return std::fabs(w[a]) > std::fabs(w[b]); });
        const double s1_ex = std::fabs(w[ord[0]]);
        const int kw = std::min(std::max(1, (int)std::ceil(window * n)), K);
        int k_ex = 0, k_ex_pos = 0, k_ex_neg = 0;
        for (int i = 0; i < n; ++i) if (std::fabs(w[i]) > TAU * s1_ex) { ++k_ex; if (w[i] > 0) ++k_ex_pos; else ++k_ex_neg; }
        const bool fired_ex = std::fabs(w[ord[kw - 1]]) < TAU * s1_ex;
        // tracked set I = { |lambda_hat| > tau * max|lambda_hat| } (what the gate deflates if it fires)
        double s1_r = 0.0; for (int i = 0; i < K; ++i) s1_r = std::max(s1_r, std::fabs(h_lam[i]));
        std::vector<double> rpos, rneg;
        for (int i = 0; i < K; ++i) if (std::fabs(h_lam[i]) > TAU * s1_r) (h_lam[i] > 0 ? rpos : rneg).push_back(h_lam[i]);
        const int p = (int)(rpos.size() + rneg.size());
        std::sort(rpos.begin(), rpos.end(), std::greater<double>());
        std::sort(rneg.begin(), rneg.end());                               // most negative first
        // sign-matched eigenvalue error: largest positives vs largest positives, most negative vs most negative
        std::vector<double> epos, eneg;
        for (int i = n - 1; i >= 0 && (int)epos.size() < (int)rpos.size(); --i) if (w[i] > 0) epos.push_back(w[i]);
        for (int i = 0; i < n && (int)eneg.size() < (int)rneg.size(); ++i) if (w[i] < 0) eneg.push_back(w[i]);
        double eig_err = 0.0; int n_pairs = 0;
        for (size_t i = 0; i < std::min(rpos.size(), epos.size()); ++i) { eig_err = std::max(eig_err, std::fabs(rpos[i] - epos[i]) / std::fabs(epos[i])); ++n_pairs; }
        for (size_t i = 0; i < std::min(rneg.size(), eneg.size()); ++i) { eig_err = std::max(eig_err, std::fabs(rneg[i] - eneg[i]) / std::fabs(eneg[i])); ++n_pairs; }
        cudaEventRecord(c1); cudaEventSynchronize(c1);
        float cms = 0.f; cudaEventElapsedTime(&cms, c0, c1);
        ckpt_ms_iter += cms; ckpt_ms_total += cms; ++n_ckpt;
        printf("# track it=%d fired=%d fired_exact=%d k_ritz=%d k_ritz_pos=%d k_ritz_neg=%d k_exact=%d k_exact_pos=%d k_exact_neg=%d"
               " pairs=%d eig_err=%.3e s1_ritz=%.6e s1_exact=%.6e ckpt_ms=%.1f\n",
               it, (int)fired_any, (int)fired_ex, p, (int)rpos.size(), (int)rneg.size(), k_ex, k_ex_pos, k_ex_neg,
               n_pairs, eig_err, s1_r, s1_ex, cms);
        fflush(stdout);
    };
    // -------------------------------------------------------------------------------------

    if (deflate) {
        CHECK_CUDA(cudaMalloc(&dV,  (size_t)n * K * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&dZ,  (size_t)n * K * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&dT,  (size_t)n * K * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&dW,  (size_t)n * K * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&dB,  (size_t)K * K * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&dG,  (size_t)K * K * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&dSel, (size_t)K * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&dEig, (size_t)K * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&dS0, nn * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&dInfoQ, sizeof(int)));
        CHECK_CUSOLVER(cusolverDnDpotrf_bufferSize(sv, CUBLAS_FILL_MODE_LOWER, K, dG, K, &dQRlw));
        CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(sv, CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER, K, dB, K, dEig, &dSyLw));
        CHECK_CUDA(cudaMalloc(&dQRwork, (size_t)dQRlw * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&dSyWork, (size_t)dSyLw * sizeof(double)));
        std::vector<double> hR((size_t)n * K);       // random seed block, orthonormalised at it 1
        std::mt19937_64 rng(20260826ULL);
        std::normal_distribution<double> N01(0.0, 1.0);
        for (auto& x : hR) x = N01(rng);
        CHECK_CUDA(cudaMemcpy(dV, hR.data(), hR.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    // CholeskyQR2 on Y (n x K) -> orthonormal columns in place; false on Cholesky breakdown.
    auto cholqr2 = [&](double* Y) -> bool {
        for (int rep = 0; rep < 2; ++rep) {
            CHECK_CUBLAS(cublasDsyrk(cb, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, K, n,
                                     &one_c, Y, n, &zero_c, dG, K));
            CHECK_CUSOLVER(cusolverDnDpotrf(sv, CUBLAS_FILL_MODE_LOWER, K, dG, K, dQRwork, dQRlw, dInfoQ));
            int hi = 0; CHECK_CUDA(cudaMemcpy(&hi, dInfoQ, sizeof(int), cudaMemcpyDeviceToHost));
            if (hi != 0) return false;
            CHECK_CUBLAS(cublasDtrsm(cb, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                     CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n, K, &one_c, dG, K, Y, n));
        }
        return true;
    };
    auto reseed = [&](unsigned long long seed) {
        std::vector<double> hR((size_t)n * K);
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> N01(0.0, 1.0);
        for (auto& x : hR) x = N01(rng);
        CHECK_CUDA(cudaMemcpy(dV, hR.data(), hR.size() * sizeof(double), cudaMemcpyHostToDevice));
        cholqr2(dV);
    };

    // ---- ACQUIRE + GATE + CLIP + REMOVE (S currently holds a copy of M) -------------
    auto deflate_step = [&](int it) {
        if (it == 1) cholqr2(dV);                                              // orthonormalise the seed
        // one warm power step + Rayleigh-Ritz:  Z = M V,  B = V^T Z = U diag(lam) U^T,  W = V U
        CHECK_CUBLAS(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, n, K, n, &one_c, M, n, dV, n, &zero_c, dZ, n));
        CHECK_CUBLAS(cublasDgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, K, K, n, &one_c, dV, n, dZ, n, &zero_c, dB, K));
        CHECK_CUSOLVER(cusolverDnDsyevd(sv, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                                        K, dB, K, dEig, dSyWork, dSyLw, dInfoQ));
        h_lam.resize(K);
        CHECK_CUDA(cudaMemcpy(h_lam.data(), dEig, (size_t)K * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUBLAS(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, n, K, K, &one_c, dV, n, dB, K, &zero_c, dW, n));

        // Muon's gate + clip (deflation_batched.py, batched_rmfro_clip), every iteration,
        // stateless, on the SINGULAR VALUES of the symmetric input, i.e. the sorted |lambda_i|
        // of the whole block:  s_1 >= s_2 >= ... , kw = ceil(w n),
        //   fired iff s_kw < tau * s_1,   clip I = { |lambda_i| > tau * s_1 }   (so 1 <= |I| <= kw-1).
        // The block always holds K >= kw values, so no missing-value convention is needed, and a
        // positive direction is clipped only if it is a head relative to the whole spectrum.
        // (A per-sign variant with a block-minimum convention for a side with fewer than kw
        // entries fired on unconverged bulk-edge Ritz values and diverged on G59mc/G60mc.)
        std::vector<double> mags(K);
        for (int i = 0; i < K; ++i) mags[i] = std::fabs(h_lam[i]);
        std::sort(mags.begin(), mags.end(), std::greater<double>());
        const int kw = std::min(std::max(1, (int)std::ceil(window * n)), K);
        const double s_1 = mags[0], s_kw = mags[kw - 1];
        fired_any = (it >= START) && (s_kw < TAU * s_1);
        theta = fired_any ? TAU * s_1 : 1e300;
        kr[0] = kr[1] = 0;
        for (int i = 0; i < K; ++i) if (std::fabs(h_lam[i]) > theta) ++kr[h_lam[i] > 0.0 ? 1 : 0];
        if (it >= START) {
            ++st_evals; st_fired += fired_any; st_fired_side[0] += (kr[0] > 0); st_fired_side[1] += (kr[1] > 0);
            ++win_cnt; win_fired += fired_any;
        }
        if (it % 250 == 0) {
            printf("# gate it=%d k=%d kw=%d s1=%.3e s_kw/s1=%.3e fired=%d clipped_neg=%d clipped_pos=%d fired_frac_last250=%.3f\n",
                   it, K, kw, s_1, s_1 > 0.0 ? s_kw / s_1 : 1.0, (int)fired_any, kr[0], kr[1],
                   win_cnt ? (double)win_fired / win_cnt : 0.0);
            win_cnt = 0; win_fired = 0;
        }
        if (is_ckpt(it)) track_checkpoint(it);                                    // [TRACK]
        // REMOVE: S <- M - sum_{i in I} lambda_i w_i w_i^T   (the clipped head to zero)
        n_clipped = 0;
        if (fired_any) {
            std::vector<double> hsel(K, 0.0);
            for (int i = 0; i < K; ++i) if (clipped(h_lam[i])) { hsel[i] = h_lam[i]; ++n_clipped; }
            CHECK_CUDA(cudaMemcpy(dSel, hsel.data(), (size_t)K * sizeof(double), cudaMemcpyHostToDevice));
            CHECK_CUBLAS(cublasDdgmm(cb, CUBLAS_SIDE_RIGHT, n, K, dW, n, dSel, 1, dT, n));
            CHECK_CUBLAS(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_T, n, n, K, &minus_one_c, dT, n, dW, n, &one_c, S, n));
        }
        // next iteration's block: V <- orth(Z)  (one power step ahead)
        CHECK_CUDA(cudaMemcpy(dV, dZ, (size_t)n * K * sizeof(double), cudaMemcpyDeviceToDevice));
        if (!cholqr2(dV)) reseed(97ULL * it);                                   // breakdown: fresh random block
    };

    double normC = 0.0, normb = std::sqrt((double)n);                 // ||b||_2 = sqrt(n) (the mb entry is 0)
    CHECK_CUBLAS(cublasDnrm2(cb, (int)nn, C, 1, &normC));
    const int TPB = 256;
    const int gridN = (n + TPB - 1) / TPB;
    const size_t gridNN = (nn + TPB - 1) / TPB;

    // symmetrization state: eps_sym = ||P - P^T||_F / ||P||_F of the projection output
    double* dAcc; CHECK_CUDA(cudaMalloc(&dAcc, sizeof(double)));
    auto sym_defect = [&](const double* P) -> double {
        CHECK_CUDA(cudaMemset(dAcc, 0, sizeof(double)));
        antisym_sq_kernel<<<1024, TPB>>>(P, n, dAcc);
        double a = 0.0, nrm = 0.0;
        CHECK_CUDA(cudaMemcpy(&a, dAcc, sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUBLAS(cublasDnrm2(cb, (int)nn, P, 1, &nrm));
        return nrm > 0.0 ? std::sqrt(2.0 * a) / nrm : 0.0;
    };
    std::vector<double> rec_raw, rec_used; std::vector<char> rec_defl;   // per logged iterate
    double sym_total_ms = 0.0;
    cudaEvent_t s0, s1; cudaEventCreate(&s0); cudaEventCreate(&s1);

    cudaEvent_t e0, e1, w0, w1;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&w0); cudaEventCreate(&w1);
    double proj_total_ms = 0.0, wall_total_ms = 0.0;

    printf("# %s SDP  problem=%s n=%d m=%d F0_nnz=%ld  mode=%s sigma=%.4g eta_tol=%g window=%.3g gamma=%.4g symmetrize=%d  ||C||_F=%.6e\n",
           mb ? "min-bisection" : "max-cut", ppath, n, m, nz, mode, sigma, eta_tol, window, gamma_pad, symmetrize, normC);
    printf("# iter  pinf        dinf        gap         eta         obj_primal      obj_dual        proj_ms  cum_proj_s  deflated  eps_sym_raw  eps_sym_used\n");

    int it = 0;
    double pinf = 1e300, dinf = 1e300, gap = 1e300, eta = 1e300, objp = 0, objd = 0;
    for (it = 1; it <= maxiter; ++it) {
        cudaEventRecord(w0);
        // y = (AA*)^-1 r,  r = sigma^-1 b - A(sigma^-1 X + S - C)
        y_update_kernel<<<gridN, TPB>>>(y, X, S, C, b, inv_sigma, n);             // r1 (diagonal part)
        y2 = 0.0;
        if (mb) {
            // r2 = sigma^-1 * 0 - 1^T (sigma^-1 X + S - C) 1 ;  AA* = [I 1; 1^T n^2]:
            //   y2 = (r2 - sum(r1)) / (n^2 - n),   y1 = r1 - y2 * 1
            const double r2 = -(inv_sigma * total_sum(X) + total_sum(S) - sumC);
            double sum_r1 = 0.0; CHECK_CUBLAS(cublasDdot(cb, n, y, 1, ones, 1, &sum_r1));
            y2 = (r2 - sum_r1) / ((double)n * n - (double)n);
            const double neg_y2 = -y2;
            CHECK_CUBLAS(cublasDaxpy(cb, n, &neg_y2, ones, 1, y, 1));
        }
        m_build_kernel<<<(unsigned)gridNN, TPB>>>(M, C, X, y, y2, inv_sigma, n);
        CHECK_CUDA(cudaMemcpy(S, M, nn * sizeof(double), cudaMemcpyDeviceToDevice));   // S = Pi_+(M) ...
        cudaEventRecord(e0);

        if (deflate) {
            deflate_step(it);
            if (fired_any && n_clipped > 0) {
                // SCALE: alpha = Lanczos bound of the head-free remainder (set by the bulk, not the head)
                double lo_r = 0.0, up_r = 0.0;
                approximate_two_norm(cb, sv, S, (size_t)n, &lo_r, &up_r);
                const double alpha = up_r > 0.0 ? up_r : 1.0, inv_a = 1.0 / alpha;
                CHECK_CUBLAS(cublasDscal(cb, (int)nn, &inv_a, S, 1));
                CHECK_CUDA(cudaMemcpy(dS0, S, nn * sizeof(double), cudaMemcpyDeviceToDevice));   // head-zero S0
                // RESTORE: each clipped direction re-enters at sign(lambda)/gamma (Muon's 1/pad)
                std::vector<double> hrb(K, 0.0), hpos(K, 0.0); int n_pos = 0;
                for (int i = 0; i < K; ++i)
                    if (clipped(h_lam[i])) {
                        hrb[i] = (h_lam[i] > 0.0 ? 1.0 : -1.0) / GAMMA;
                        if (h_lam[i] > 0.0) { hpos[i] = h_lam[i]; ++n_pos; }
                    }
                CHECK_CUDA(cudaMemcpy(dSel, hrb.data(), (size_t)K * sizeof(double), cudaMemcpyHostToDevice));
                CHECK_CUBLAS(cublasDdgmm(cb, CUBLAS_SIDE_RIGHT, n, K, dW, n, dSel, 1, dT, n));
                CHECK_CUBLAS(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_T, n, n, K, &one_c, dT, n, dW, n, &one_c, S, n));
                // ITERATE + RECOVER: the reference composite on the restored matrix, recovery
                // product 0.5 * S0 (I + X_T) with the head-zero S0 -> head output exactly 0
                composite_FP16_rec(cb, S, dS0, n, wsF, wsH);
                CHECK_CUBLAS(cublasDscal(cb, (int)nn, &alpha, S, 1));
                // ADD BACK the positive head exactly (Pi_+ is the identity there); nothing
                // for the negative head (Pi_+ = 0 there)
                if (n_pos > 0) {
                    CHECK_CUDA(cudaMemcpy(dSel, hpos.data(), (size_t)K * sizeof(double), cudaMemcpyHostToDevice));
                    CHECK_CUBLAS(cublasDdgmm(cb, CUBLAS_SIDE_RIGHT, n, K, dW, n, dSel, 1, dT, n));
                    CHECK_CUBLAS(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_T, n, n, K, &one_c, dT, n, dW, n, &one_c, S, n));
                }
                alpha_last = alpha; ++st_restore;
                if (it % 250 == 0)
                    printf("# deflate it=%d alpha=%.3e theta=%.3e clipped_neg=%d clipped_pos=%d\n",
                           it, alpha, theta, n_clipped - n_pos, n_pos);
            } else {
                composite_FP16_auto_scale(cb, sv, S, n, wsF, wsH);              // gate closed: reference filter
            }
        } else if (fp16) {
            composite_FP16_auto_scale(cb, sv, S, n, wsF, wsH);
        } else {
            composite_FP32_auto_scale(cb, sv, S, n, wsF);
        }
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float pms = 0.f; cudaEventElapsedTime(&pms, e0, e1);
        pms -= (float)ckpt_ms_iter; ckpt_ms_iter = 0.0;                        // [TRACK] untimed
        proj_total_ms += pms;

        // ---- SYMMETRIZATION: S now holds the projection output P.  Measure eps_sym on the
        // logged iterates (untimed), then P <- (P + P^T)/2 (timed, counted as projection
        // cost) before X consumes it.  eps_sym_used is the defect of the P used downstream. ----
        const bool measure = (it <= 20 || it % log_every == 0);
        double eps_raw = -1.0, eps_used = -1.0;
        if (measure) eps_raw = sym_defect(S);
        if (symmetrize) {
            cudaEventRecord(s0);
            symmetrize_kernel<<<(unsigned)gridNN, TPB>>>(S, n);
            cudaEventRecord(s1); cudaEventSynchronize(s1);
            float sms = 0.f; cudaEventElapsedTime(&sms, s0, s1);
            sym_total_ms += sms; pms += sms; proj_total_ms += sms;
            if (measure) eps_used = sym_defect(S);
        } else if (measure) eps_used = eps_raw;
        if (measure) {
            rec_raw.push_back(eps_raw); rec_used.push_back(eps_used);
            rec_defl.push_back((char)(deflate && fired_any && n_clipped > 0));
        }

        x_update_kernel<<<(unsigned)gridNN, TPB>>>(X, S, M, sigma, nn);

        // ---- KKT residuals (the eigenvalue-free eta of the paper) --------------
        diag_res_kernel<<<gridN, TPB>>>(rdiag, X, b, n);
        double rp = 0.0; CHECK_CUBLAS(cublasDnrm2(cb, n, rdiag, 1, &rp));
        if (mb) { const double sX = total_sum(X); rp = std::sqrt(rp * rp + sX * sX); }   // (<J,X> - 0)^2
        pinf = rp / (1.0 + normb);
        dres_kernel<<<(unsigned)gridNN, TPB>>>(M, S, C, y, y2, n);   // M is free now
        double rd = 0.0; CHECK_CUBLAS(cublasDnrm2(cb, (int)nn, M, 1, &rd));
        dinf = rd / (1.0 + normC);
        CHECK_CUBLAS(cublasDdot(cb, (int)nn, C, 1, X, 1, &objp));
        CHECK_CUBLAS(cublasDdot(cb, n, b, 1, y, 1, &objd));   // b^T y; the mb component has b = 0
        gap = std::fabs(objp - objd) / (1.0 + std::fabs(objp) + std::fabs(objd));
        eta = std::max(pinf, std::max(dinf, gap));
        cudaEventRecord(w1); cudaEventSynchronize(w1);
        float wms = 0.f; cudaEventElapsedTime(&wms, w0, w1);
        wall_total_ms += wms;

        if (it <= 20 || it % log_every == 0 || eta < eta_tol) {
            printf("%6d  %.4e  %.4e  %.4e  %.4e  %+.8e  %+.8e  %8.2f  %9.3f  %d  %.3e  %.3e\n",
                   it, pinf, dinf, gap, eta, objp, objd, pms, proj_total_ms / 1e3,
                   deflate ? n_clipped : 0, eps_raw, eps_used);
            fflush(stdout);
        }
        if (eta < eta_tol) break;
    }
    if (it > maxiter) it = maxiter;

    if (deflate)
        printf("# track summary: %ld checkpoints every %d its from it=%d, %.1f s total (excluded from proj time)\n",
               n_ckpt, TRACK_EVERY, START, ckpt_ms_total / 1e3);
    if (deflate)
        printf("# deflation summary: gate evaluated %ld its (it>=%d), fired %ld (%.1f%%) [clipped a negative head %ld times, a positive one %ld];"
               " deflated projections %ld; last alpha=%.3e\n",
               st_evals, START, st_fired, st_evals ? 100.0 * st_fired / st_evals : 0.0,
               st_fired_side[0], st_fired_side[1], st_restore, alpha_last);

    // ---- symmetry-defect summary: mean / median / max of eps_sym over the logged iterates ----
    auto stats = [](std::vector<double> v, double* mean, double* med, double* mx) {
        if (v.empty()) { *mean = *med = *mx = 0.0; return; }
        std::sort(v.begin(), v.end());
        double s = 0.0; for (double x : v) s += x;
        *mean = s / v.size(); *med = v[v.size() / 2]; *mx = v.back();
    };
    {
        std::vector<double> raw_d, used_d;
        for (size_t i = 0; i < rec_raw.size(); ++i) if (rec_defl[i]) { raw_d.push_back(rec_raw[i]); used_d.push_back(rec_used[i]); }
        double m1, d1, x1, m2, d2, x2;
        printf("# symmetry defect eps_sym = ||P-P^T||_F/||P||_F on %zu logged iterates (%zu on the deflated path), symmetrize=%d:\n",
               rec_raw.size(), raw_d.size(), symmetrize);
        stats(rec_raw, &m1, &d1, &x1);  stats(raw_d, &m2, &d2, &x2);
        printf("#   before symmetrization: all mean %.3e median %.3e max %.3e | deflated-path mean %.3e median %.3e max %.3e\n", m1, d1, x1, m2, d2, x2);
        stats(rec_used, &m1, &d1, &x1); stats(used_d, &m2, &d2, &x2);
        printf("#   after  symmetrization: all mean %.3e median %.3e max %.3e | deflated-path mean %.3e median %.3e max %.3e\n", m1, d1, x1, m2, d2, x2);
        printf("# symmetrize=%d: %.1f ms total (%.3f ms/it), included in proj time\n", symmetrize, sym_total_ms, sym_total_ms / it);
    }

    // ---- a posteriori: lambda_min of the final iterates, rank(X), full eta ----
    std::vector<double> wX, wS;
    sym_eigs(sv, X, n, wX); sym_eigs(sv, S, n, wS);
    const double lminX = wX.front(), lmaxX = wX.back(), lminS = wS.front();
    int rankX = 0, rankX3 = 0, rankX4 = 0;
    for (double v : wX) {
        if (v > 1e-6 * std::max(lmaxX, 1e-300)) ++rankX;      // the reference threshold (counts FP16 noise as rank)
        if (v > 1e-4 * std::max(lmaxX, 1e-300)) ++rankX4;
        if (v > 1e-3 * std::max(lmaxX, 1e-300)) ++rankX3;
    }
    printf("# rank(X) at relative thresholds 1e-6 / 1e-4 / 1e-3 of lambda_max: %d / %d / %d\n", rankX, rankX4, rankX3);
    const double eta_full = std::max(eta, std::max(std::max(0.0, -lminX) / (1.0 + normb),
                                                   std::max(0.0, -lminS) / (1.0 + normC)));
    printf("\nFINAL problem=%s mode=%s sigma=%.4g iters=%d | pinf %.3e dinf %.3e gap %.3e eta %.3e"
           " | lmin(X) %.3e lmin(S) %.3e eta_full %.3e | rank(X) %d/%d"
           " | obj %.8e | proj %.1f s (%.1f ms/it) wall %.1f s | symmetrize %d eps_raw_max %.3e eps_used_max %.3e\n",
           ppath, mode, sigma, it, pinf, dinf, gap, eta, lminX, lminS, eta_full, rankX, n, objp,
           proj_total_ms / 1e3, proj_total_ms / it, wall_total_ms / 1e3, symmetrize,
           rec_raw.empty() ? 0.0 : *std::max_element(rec_raw.begin(), rec_raw.end()),
           rec_used.empty() ? 0.0 : *std::max_element(rec_used.begin(), rec_used.end()));
    CHECK_CUBLAS(cublasDestroy(cb)); CHECK_CUSOLVER(cusolverDnDestroy(sv));
    return 0;
}
