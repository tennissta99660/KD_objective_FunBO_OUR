#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kd_objective.py

Subprocess wrapper that runs student_kd.py with a given hyperparameter dict
and returns 1.0 - val_acc (so lower is better for BO), same interface as before.
"""

import tempfile, json, os, subprocess, sys, time, shutil

def kd_objective(hparams, script="student_kd.py", short=True, timeout=1800):
    td = tempfile.mkdtemp(prefix="kd_trial_")
    cfg_path = os.path.join(td, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(hparams, f)

    cmd = [sys.executable, script, "--config", cfg_path]
    if short:
        cmd.append("--short")

    try:
        start = time.time()
        # capture_output to reduce main console noise; use text mode
        proc = subprocess.run(cmd, check=True, timeout=timeout, capture_output=True, text=True)
        elapsed = time.time() - start

        res_path = os.path.join(td, "result.json")
        if os.path.exists(res_path):
            r = json.load(open(res_path))
            acc = r.get("val_acc", None)
            if acc is None:
                print(f"[kd_objective] result.json missing val_acc in {td}; treating as failure.")
                return 1.0
            return 1.0 - float(acc)
        else:
            # No result file means something went wrong
            print(f"[kd_objective] No result.json found in {td}. stdout/stderr:")
            print(proc.stdout)
            print(proc.stderr)
            return 1.0
    except subprocess.TimeoutExpired:
        print(f"[kd_objective] Timeout during KD run in {td}")
        return 1.0
    except subprocess.CalledProcessError as e:
        print(f"[kd_objective] KD run failed in {td}:")
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        return 1.0
    finally:
        if os.path.exists(td):
            try:
                shutil.rmtree(td)
            except Exception:
                pass
