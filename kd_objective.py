import tempfile, json, os, subprocess, sys, time
import shutil # FIX: Import shutil for directory cleanup

def kd_objective(hparams, script="student_kd.py", short=True, timeout=900):
    td = tempfile.mkdtemp(prefix="kd_trial_")
    cfg_path = os.path.join(td, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(hparams, f)
    
    cmd = [sys.executable, script, "--config", cfg_path]
    if short:
        cmd.append("--short")
    
    try:
        start = time.time()
        # FIX: Added capture_output=True to hide subprocess stdout/stderr
        # unless an error occurs, cleaning up the main console.
        subprocess.run(cmd, check=True, timeout=timeout, capture_output=True, text=True)
        elapsed = time.time() - start
        
        res_path = os.path.join(td, "result.json")
        if os.path.exists(res_path):
            r = json.load(open(res_path))
            # FIX: The 'or 0.0' is redundant but harmless.
            # Kept it as it provides a fallback if 'val_acc' is None.
            acc = r.get("val_acc", 0.0) or 0.0
            return 1.0 - acc
        else:
            print(f"No result.json found in {td}")
            return 1.0 # Maximize loss
            
    except subprocess.TimeoutExpired:
        print(f"Timeout during KD run in {td}")
        return 1.0 # Maximize loss
    except subprocess.CalledProcessError as e:
        print(f"KD run failed in {td}: {e.stderr}")
        return 1.0 # Maximize loss
    finally:
        # FIX: Added a finally block to ensure the temporary
        # directory is always deleted, preventing file buildup.
        if os.path.exists(td):
            shutil.rmtree(td)