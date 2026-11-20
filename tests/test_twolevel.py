#!/usr/bin/env python3
"""Two-level study (twolevel/, scripts/twolevel.py): identity with the original code + tiny end-to-end run.

1. Identity. tests/twolevel_reference.json holds, for 2 small matrices per problem (n ~ 3k, test seed
   1000), the matrix hash and, per method, the sha1 of the strength matrix C, the PCG iteration count,
   the coarse size and the coarse LU nnz, computed on the reference cluster (Linux x86_64) with the
   ORIGINAL prototype scripts of the three sub-studies ("bar", "coarse", "reward") for the
   classical rules, the paper GAT policy (FastPolicy and reference path, results/paper_original models),
   the committed lin / node policies and the hand rule, with both splu settings. The consolidated
   harness must reproduce all of them exactly on the reference platform; elsewhere the generated
   matrices can differ in the last bits (see tests/smoke.sh and docs/REPRODUCE.md). The same check at
   50k on the real test matrices is summarised in results/twolevel_2026-10/README.md, "Verification".
2. Pipeline. `scripts/twolevel.py all --local --small` (data, tuning, bar, paper rows, cost model,
   training, selection, final comparisons, headroom, report) at n ~ 3k into outputs_smoke/twolevel.

Run from the repository root:  python tests/test_twolevel.py [--identity-only | --pipeline-only] [--reference F]
Takes a few minutes on one node.
"""
import json
import os
import shutil
import subprocess
import sys
import hashlib
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def A_sha1(A):
    import numpy as np
    h = hashlib.sha1()
    for arr in (A.indptr.astype(np.int64), A.indices.astype(np.int64), A.data.astype(np.float64)):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def identity(ref_file):
    ref = json.loads(Path(ref_file).read_text())
    tmp = tempfile.mkdtemp(prefix="tl_identity_")
    os.environ["TWOLEVEL_DATA"] = tmp
    os.chdir(ROOT)
    from twolevel import harness as H
    from twolevel.common import gen_set, load_A
    bad = 0
    cache = {}
    for r in ref:
        key = (r["problem"], r["n_target"], r["seed"], r["path"])
        if key not in cache:
            i = int(r["path"].rsplit("_", 1)[1].split(".")[0])
            path = gen_set(r["problem"], r["n_target"], r["seed"], i + 1, verbose=False)[i]
            A, fseed = load_A(path)
            if A_sha1(A) != r["A_sha1"] or fseed != r["fseed"]:
                print(f"MATRIX MISMATCH {key}")
                bad += 1
            cache[key] = (A, fseed)
        A, fseed = cache[key]
        got = H.fingerprint(A, fseed, r["method"])
        diff = [k for k in ("C_sha1", "iters", "coarse_n", "lu_nnz") if got[k] != r[k]]
        status = "ok" if not diff else "MISMATCH " + ", ".join(f"{k} {got[k]} != {r[k]}" for k in diff)
        bad += bool(diff)
        print(f"{r['problem']:12s} {r['path']:22s} {r['origin']:6s} {r['method'][-58:]:58s} it={got['iters']:3d} {status}")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"identity: {len(ref) - bad}/{len(ref)} fingerprints identical to the original prototype code")
    return bad == 0


def pipeline():
    env = dict(os.environ, LSC_OUTPUTS="outputs_smoke")
    env.pop("TWOLEVEL_OUT", None)
    env.pop("TWOLEVEL_DATA", None)
    out = ROOT / "outputs_smoke" / "twolevel"
    shutil.rmtree(out, ignore_errors=True)
    subprocess.run([sys.executable, "scripts/twolevel.py", "all", "--local", "--small"], cwd=ROOT, env=env, check=True)
    need = ["tune/def_50000_anisotropy.json", "bar/def_50000_perforated_l.json", "bar/sym_500000_lognormal.json",
            "paper/v2_50000_anisotropy.json", "costmodel/cost_model.json", "costmodel/cost_model_sym.json",
            "policies/node/perforated_l_noderow_costtgt/train.json", "policies/lin/lin_perforated_l_50k_theta.json",
            "select/sym_50000_lognormal.json", "final/def_50000_perforated_l.json", "final/sym_500000_perforated_l.json",
            "headroom/retime_lognormal.json", "report.md"]
    missing = [f for f in need if not (out / f).exists()]
    if missing:
        print("pipeline: missing outputs", missing)
        return False
    J = json.loads((out / "final" / "def_50000_perforated_l.json").read_text())
    meths = {r["method"] for r in J["rows"]}
    ok = any(m.startswith("node:") for m in meths) and any(m.startswith("lin:") for m in meths) and all(
        r["iters"] < 500 for r in J["rows"])
    print(f"pipeline: {'ok' if ok else 'FAILED'} ({len(J['rows'])} rows in final/def_50000_perforated_l.json)")
    return ok


def main():
    ok = True
    ref = ROOT / "tests" / "twolevel_reference.json"
    if "--reference" in sys.argv:          # e.g. the same fingerprints at 50k (verification on the reference cluster)
        ref = sys.argv[sys.argv.index("--reference") + 1]
    if "--pipeline-only" not in sys.argv:
        ok &= identity(ref)
    if "--identity-only" not in sys.argv:
        ok &= pipeline()
    print("TWOLEVEL TEST", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
