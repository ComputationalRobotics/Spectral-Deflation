#!/usr/bin/env python3
"""Summarise results/<instance>_<mode>.txt: best/final eta, lambda_min, ms/it,
first firing iteration and firing fraction.   usage: scripts/report.py [G55mc ...]"""
import glob, os, re, statistics as st, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES = os.path.join(ROOT, "results")
def load(fn):
    d = {"eta": [], "it": [], "final": None, "first_fire": None, "fired_frac": None, "ms": None}
    for l in open(fn):
        if re.match(r"^\s+\d+\s", l):
            p = l.split()
            try:
                d["it"].append(int(p[0])); d["eta"].append(float(p[4]))
                if d["first_fire"] is None and len(p) > 9 and p[9].isdigit() and int(p[9]) > 0: d["first_fire"] = int(p[0])
            except ValueError: pass
        elif l.startswith("FINAL"):
            d["final"] = l.strip()
            m = re.search(r"\((\d+\.\d+) ms/it\)", l); d["ms"] = float(m.group(1)) if m else None
        elif "summary" in l and "fired" in l:
            m = re.search(r"fired (\d+) \(([\d.]+)%\)", l); d["fired_frac"] = float(m.group(2)) if m else None
    return d
def field(final, key):
    m = re.search(key + r" (-?[\d.]+e[-+]\d+)", final or ""); return float(m.group(1)) if m else float("nan")
INST = ("G55mc", "G59mc", "G60mc", "G60_mb")
graphs = sys.argv[1:] or [g for g in INST if glob.glob(f"{RES}/{g}_*.txt")]
for g in graphs:
    print(f"\n== {g} ==")
    print(f"{'mode':<16}{'best eta':>10}{'final eta':>11}{'lmin(S)':>11}{'lmin(X)':>11}{'ms/it':>7}{'1st fire':>9}{'fired%':>8}")
    modes = ["baseline32", "baseline16", "deflated16"] + sorted(os.path.basename(f)[len(g)+1:-4] for f in glob.glob(f"{RES}/{g}_deflated16_g*.txt"))
    for m in modes:
        fn = f"{RES}/{g}_{m}.txt"
        if not os.path.exists(fn): continue
        d = load(fn)
        if not d["eta"]: continue
        lminS = field(d['final'], r'lmin\(S\)'); lminX = field(d['final'], r'lmin\(X\)')
        ms = d['ms'] or float('nan')
        ff = d['fired_frac'] if d['fired_frac'] is not None else float('nan'); first = str(d['first_fire'] or '-')
        print(f"{m:<16}{min(d['eta']):>10.3e}{d['eta'][-1]:>11.3e}{lminS:>11.2e}{lminX:>11.2e}{ms:>7.1f}{first:>9}{ff:>8.1f}")
print("\nms/it from FINAL (includes the first-iteration warm-up).")
