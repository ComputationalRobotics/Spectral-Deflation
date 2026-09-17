#!/usr/bin/env python3
"""Convert a Mittelmann Gset SDP (SDPA sparse, gz) into the compact format read by admm_mc:
   line 1:  n m type        (type: mc = diag(X)=1 ;  mb = diag(X)=1 and <J,X>=0)
   line 2+: i j val         upper-triangle nonzeros of F0 (1-based).  The solver minimises <C,X>
                            with C = -F0 (SDPA's dual maximises <F0,Y>).
Verifies the constraint structure while streaming."""
import gzip, sys
src, dst = sys.argv[1], sys.argv[2]
with gzip.open(src, "rt") as f:
    m = int(f.readline().split()[0]); nblk = int(f.readline().split()[0]); n = int(f.readline().split()[0])
    assert nblk == 1, nblk
    b = [float(x) for x in f.readline().split()]
    assert len(b) == m
    C = []; diag_ok = [False] * (n + 1); last_cnt = 0; other = 0
    for line in f:
        p = line.split()
        if len(p) < 5: continue
        k, blk, i, j, v = int(p[0]), int(p[1]), int(p[2]), int(p[3]), float(p[4])
        if k == 0: C.append((i, j, v))
        elif k <= n:
            if i == j == k and v == 1.0: diag_ok[k] = True
            else: other += 1
        else:
            last_cnt += 1
            if v != 1.0: other += 1
assert all(diag_ok[1:]), "constraints 1..n are not diag(X)=1"
assert other == 0, f"unexpected constraint entries: {other}"
if m == n:
    typ = "mc"; assert all(x == 1.0 for x in b)
elif m == n + 1:
    typ = "mb"; assert all(x == 1.0 for x in b[:n]) and b[n] == 0.0 and last_cnt == n * (n + 1) // 2, (b[n], last_cnt)
else:
    raise SystemExit(f"unexpected m={m} for n={n}")
with open(dst, "w") as g:
    g.write(f"{n} {m} {typ}\n")
    for i, j, v in C: g.write(f"{i} {j} {v:.17g}\n")
offd = sum(1 for i, j, _ in C if i != j); dg = sum(1 for i, j, _ in C if i == j)
print(f"{src}: n={n} m={m} type={typ} F0 nonzeros={len(C)} (diag {dg}, offdiag {offd}) -> {dst}")
