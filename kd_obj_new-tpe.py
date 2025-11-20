#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kd_objective.py

Thin wrapper to call student_kd.py as a subprocess (keeps current setup).
Returns 1.0 - val_acc (so lower is better for optimization pipelines).
Safe: handles timeouts / errors and cleans up temporary dirs.
"""

import tempfile
import json
import os
import sys
import subprocess
import time
import shutil

def kd_objective(hparams, script="student_kd.py", short=True, timeout=1800):
    """
    Run student_kd.py with the given hparams -> returns 1 - val_acc (loss).
    - hparams: dict with keys alpha, weight_decay, temperature, learning_rate, batch_size, epochs, ...
    - short: if True, runs in short cheap mode.
    - timeout: seconds before killing process.
    """
    td = tempfile.mkdtemp(prefix="kd_trial_")
    cfg_path = os.path.join(td, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(hparams, f)

    cmd = [sys.executable, script, "--config", cfg_path]
    if short:
        cmd.append("--short")

    try:
        start = time.time()
        # capture output to avoid console spam; if you want logs, set capture_output=False
        subprocess.run(cmd, check=True, timeout=timeout, capture_output=True, text=True)
        elapsed = time.time() - start

        res_path = os.path.join(td, "result.json")
        if os.path.exists(res_path):
            r = json.load(open(res_path))
            acc = r.get("val_acc", 0.0) or 0.0
            return 1.0 - acc
        else:
            # missing file -> failure (return worst possible)
            print(f"kd_objective: no result.json found in {td}")
            return 1.0
    except subprocess.TimeoutExpired:
        print(f"kd_objective: Timeout during KD run in {td}")
        return 1.0
    except subprocess.CalledProcessError as e:
        print(f"kd_objective: KD run failed in {td}: {getattr(e, 'stderr', str(e))}")
        return 1.0
    finally:
        try:
            if os.path.exists(td):
                shutil.rmtree(td)
        except Exception:
            pass
