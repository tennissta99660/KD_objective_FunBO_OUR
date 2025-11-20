#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
funbo_core_v2.py  —  Dual-LLM FunBO (Groq, sequential, RTX-friendly)

What you get:
- Two Groq LLMs (same model, two API keys) with DIFFERENT temperatures
  * LLM-A (Exploration): high temperature, no examples (encourage novelty)
  * LLM-B (Exploitation): low temperature, few-shot guidance (encourage stability)
- Separate feedback:
  * A gets ONLY its own historical best score
  * B gets the global best score so far (EI or any AF)
- No SQLite, no multiprocessing (safe for 8GB RTX 5050)
- AUC-based scoring over BO steps + per-candidate validation accuracy stats
- Per-candidate printout: mean AUC, std, mean best val_acc, std, and EI baseline for comparison
- Robust AF compilation (works even if LLM forgets to import; we inject np/norm)
- Simple, sequential flow with tqdm progress

.env required:
  GROQ_API_KEY_A=...
  GROQ_API_KEY_B=...
  (optional) GROQ_MODEL=llama-3.3-70b-versatile
"""

import os
import re
import json
import argparse
import time
import numpy as np
from scipy.stats import norm
from dotenv import load_dotenv
from groq import Groq
from kd_objective import kd_objective
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from tqdm import tqdm
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# -----------------------------
# Tunables (safe defaults for RTX 5050 8GB + 16GB RAM)
# -----------------------------
GRID_SIZE = 96            # grid size per BO step (keep modest for speed)
DEFAULT_INNER_T = 5       # BO steps per evaluation
REPEATS_PER_AF = 3        # repeats per candidate (stability)
TEMP_A = 0.75             # exploration (LLM-A) temperature
TEMP_B = 0.35             # exploitation (LLM-B) temperature
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

load_dotenv()

# -----------------------------
# Groq setup (two separate API keys)
# -----------------------------
API_KEY_A = os.getenv("GROQ_API_KEY_A")
API_KEY_B = os.getenv("GROQ_API_KEY_B")
if not API_KEY_A or not API_KEY_B:
    raise RuntimeError("Please set GROQ_API_KEY_A and GROQ_API_KEY_B in your .env")

client_a = Groq(api_key=API_KEY_A)
client_b = Groq(api_key=API_KEY_B)
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# -----------------------------
# AF / GP Utilities
# -----------------------------
def clean_llm_code(output: str) -> str:
    cleaned = re.sub(r"```(?:python)?", "", output)
    return cleaned.replace("```", "").strip()

def split_candidates(output: str, want=3):
    """
    Split multiple functions. Accepts 'def AF' or 'def AF1/AF2...'.
    Renames to 'def AF(' for compilation.
    """
    parts = re.split(r"(?=def\s+AF\d*\s*\()", output)
    cands = [p.strip() for p in parts if p.strip()]
    fixed = [re.sub(r"def\s+AF\d*\s*\(", "def AF(", c) for c in cands]
    return fixed[:want]

def compile_af(code_str):
    """
    Compile AF code; inject numpy and norm so it works even if LLM forgot imports.
    """
    safe_globals = {"np": np, "norm": norm}
    try:
        exec(code_str, safe_globals)
    except Exception as e:
        raise RuntimeError(f"Compilation failed: {e}")
    af = safe_globals.get("AF")
    if not callable(af):
        # fallback: any callable containing 'AF' in name
        cands = [v for k, v in safe_globals.items() if callable(v) and "AF" in k]
        if cands:
            af = cands[0]
        else:
            raise RuntimeError("Compiled AF does not define AF(mu, sigma, f_best).")
    return af

def expected_improvement(mu, sigma, f_best):
    z = (f_best - mu) / (sigma + 1e-9)
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

def unnormalize(x):
    """
    Map [0,1]^4 -> KD hparams expected by student_kd.py
    Keys match your earlier script (alpha, weight_decay, temperature, learning_rate).
    """
    return {
        "alpha": float(x[0]),
        "weight_decay": 10 ** (-5 + 3.0 * float(x[1])),
        "temperature": 2.0 + 18.0 * float(x[2]),
        "learning_rate": 10 ** (-4 + 3.0 * float(x[3])),
        "batch_size": 32,
        "epochs": 3,   # keep short for speed; adjust as needed
    }

def auc_from_best_losses(best_losses):
    """
    best_losses[t] = best loss after t steps (lower is better).
    Convert to reward = 1 - loss, then take mean (AUC-like).
    """
    rewards = [1.0 - float(l) for l in best_losses]
    return float(np.mean(rewards))

def run_inner_bo_single(AF_callable, X_init, y_init, T=DEFAULT_INNER_T, desc="BO"):
    """
    One BO run: returns (auc, best_loss_series, best_val_acc)
      - auc: area under best reward curve over steps
      - best_loss_series: list of best loss after each step (len T+1 including initial)
      - best_val_acc: 1 - min_loss observed
    """
    dim = X_init.shape[1]
    X, y = X_init.copy(), y_init.copy()

    kernel = C(1.0) * RBF(length_scale=np.ones(dim))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

    best_losses = [float(np.min(y))]
    for _ in tqdm(range(T), desc=desc, leave=False):
        gp.fit(X, y)
        grid = np.random.rand(GRID_SIZE, dim)
        mu, sigma = gp.predict(grid, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        try:
            af_vals = AF_callable(mu, sigma, float(np.min(y)))
        except Exception as e:
            # If AF fails, fall back to EI for this step (keeps flow robust)
            af_vals = expected_improvement(mu, sigma, float(np.min(y)))
        af_vals = np.asarray(af_vals)
        af_vals = np.nan_to_num(af_vals, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        idx = int(np.argmax(af_vals))
        x_next = grid[idx]
        y_next = kd_objective(unnormalize(x_next), short=True)  # short=True for speed
        X = np.vstack([X, x_next])
        y = np.append(y, y_next)
        best_losses.append(float(np.min(y)))

    auc = auc_from_best_losses(best_losses)
    best_val_acc = 1.0 - float(np.min(best_losses))
    return auc, best_losses, best_val_acc

# -----------------------------
# LLM prompts (Dual)
# -----------------------------
SYSTEM_PROMPT = (
    "You are an expert in Bayesian Optimization and Machine Learning. "
    "Return only valid Python code — no markdown, comments, or text outside of functions. "
    "Every function must be a standalone definition of `def AF(mu, sigma, f_best):` "
    "and return a numpy array (same shape as mu). Ensure numeric stability."
)

PROMPT_A_TEMPLATE = """
Design {n} distinct acquisition functions for Bayesian Optimization in Knowledge Distillation.
Your role is EXPLORATION: generate novel, diverse, non-standard AFs.
Constraints:
- Each is a function: def AF(mu, sigma, f_best):
- Include any needed imports INSIDE each function (e.g., numpy as np, scipy.stats.norm).
- Be differentiable and numerically stable (avoid division by zero; clamp sigma).
- Avoid standard EI/PI/UCB formulas or trivial tweaks.
- Return only functions, separated by one blank line.

For context (do not copy):
Current best score YOU have achieved so far: {a_best:.6f}

Output exactly {n} functions.
"""

FEW_SHOTS = """
def AF_example1(mu, sigma, f_best):
    import numpy as np
    from scipy.stats import norm
    s = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / s
    return np.sqrt(s) * np.exp(-0.5*z*z) + np.maximum(0, f_best - mu)

def AF_example2(mu, sigma, f_best):
    import numpy as np
    s = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / s
    t = np.tanh(z)
    return (1 - t) * np.sqrt(s) + 0.2 * s * np.exp(-np.abs(z))
"""

PROMPT_B_TEMPLATE = """
Design {n} acquisition functions for Bayesian Optimization in Knowledge Distillation.
Your role is EXPLOITATION: remain effective and stable; light inspiration from examples.
Constraints:
- Each is a function: def AF(mu, sigma, f_best):
- Include necessary imports INSIDE each function (numpy as np, scipy.stats.norm).
- Allow mild use of EI-like components but avoid pure copies.

Reference ideas (do NOT copy verbatim):
{few_shots}

Global best score to beat so far: {global_best:.6f}

Output exactly {n} functions, separated by one blank line.
"""

def groq_generate_afs_dual(n_funcs, a_best, global_best):
    """
    Returns (cands_a, cands_b): lists of code strings (up to n_funcs each)
    """
    # LLM-A (exploration)
    prompt_a = PROMPT_A_TEMPLATE.format(n=n_funcs, a_best=a_best)
    try:
        resp_a = client_a.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_a},
            ],
            temperature=TEMP_A,   # high temp for exploration
            max_tokens=1400,
            top_p=1.0,
        )
        raw_a = resp_a.choices[0].message.content.strip()
        cands_a = split_candidates(clean_llm_code(raw_a), want=n_funcs)
        print(f"🧠 LLM-A produced {len(cands_a)} candidates.")
    except Exception as e:
        print(f"❌ LLM-A generation error: {e}")
        cands_a = []

    # LLM-B (exploitation)
    prompt_b = PROMPT_B_TEMPLATE.format(n=n_funcs, few_shots=FEW_SHOTS, global_best=global_best)
    try:
        resp_b = client_b.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_b},
            ],
            temperature=TEMP_B,   # low temp for exploitation
            max_tokens=1400,
            top_p=0.35,
        )
        raw_b = resp_b.choices[0].message.content.strip()
        cands_b = split_candidates(clean_llm_code(raw_b), want=n_funcs)
        print(f"🧩 LLM-B produced {len(cands_b)} candidates.")
    except Exception as e:
        print(f"❌ LLM-B generation error: {e}")
        cands_b = []

    return cands_a, cands_b

# -----------------------------
# Evaluation helpers
# -----------------------------
def evaluate_candidate_code(code_str, X_init, y_init, inner_T=DEFAULT_INNER_T, desc="AF"):
    """
    Compile and evaluate one AF code over REPEATS_PER_AF runs.
    Returns:
      mean_auc, std_auc, mean_best_acc, std_best_acc
    """
    try:
        AF = compile_af(code_str)
    except Exception as e:
        print(f"[COMPILE FAIL] {e}")
        return 0.0, 0.0, 0.0, 0.0

    aucs = []
    best_accs = []
    for r in range(REPEATS_PER_AF):
        # different random seed per repeat
        seed = np.random.randint(0, 2**31 - 1)
        np.random.seed(seed)
        auc, best_losses, best_acc = run_inner_bo_single(
            AF, X_init, y_init, T=inner_T, desc=f"{desc} (repeat {r+1}/{REPEATS_PER_AF})"
        )
        aucs.append(auc)
        best_accs.append(best_acc)

    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(best_accs)), float(np.std(best_accs))

def evaluate_ei_baseline(X_init, y_init, inner_T=DEFAULT_INNER_T):
    """
    Evaluate EI over REPEATS_PER_AF runs (for fair comparison).
    """
    aucs, best_accs = [], []
    for r in range(REPEATS_PER_AF):
        seed = np.random.randint(0, 2**31 - 1)
        np.random.seed(seed)
        ei_auc, _, ei_best_acc = run_inner_bo_single(
            expected_improvement, X_init, y_init, T=inner_T, desc=f"EI (repeat {r+1}/{REPEATS_PER_AF})"
        )
        aucs.append(ei_auc)
        best_accs.append(ei_best_acc)
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(best_accs)), float(np.std(best_accs))

# -----------------------------
# Core runner
# -----------------------------
class FunBOGroqDual:
    def __init__(self, out_dir="funbo_dual_runs", seed=RANDOM_SEED):
        np.random.seed(seed)
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.log = []

    @staticmethod
    def base_af_code():
        return """
def AF(mu, sigma, f_best):
    import numpy as np
    from scipy.stats import norm
    s = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / s
    return (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
"""

    def save_candidate(self, gen, tag, code):
        path = os.path.join(self.out_dir, f"gen{gen}_{tag}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"💾 Saved: {path}")

    def run(self, generations=7, candidates_per_llm=3, inner_T=DEFAULT_INNER_T):
        # Initial GP seed points (same for all evals)
        dim = 4
        n_init = 2
        X_init = np.random.rand(n_init, dim)
        y_init = np.array([kd_objective(unnormalize(x), short=True) for x in X_init])

        # Evaluate EI baseline once per generation (with repeats inside)
        best_global_score = -np.inf  # track best AUC
        best_global_code = self.base_af_code()
        best_A_hist = -np.inf         # LLM-A's historical best AUC

        print("Evaluating EI baseline...")
        ei_auc_mean, ei_auc_std, ei_acc_mean, ei_acc_std = evaluate_ei_baseline(X_init, y_init, inner_T=inner_T)
        print(f"EI | mean AUC={ei_auc_mean:.4f} ± {ei_auc_std:.4f} | mean best acc={ei_acc_mean:.4f} ± {ei_acc_std:.4f}")
        if ei_auc_mean > best_global_score:
            best_global_score = ei_auc_mean
            best_global_code = self.base_af_code()

        for g in range(1, generations + 1):
            print(f"\n🚀 Generation {g}/{generations}")

            # Feedback separation
            feedback_A = best_A_hist if best_A_hist != -np.inf else 0.0
            feedback_B = best_global_score if best_global_score != -np.inf else 0.0

            # Generate candidates
            cand_a, cand_b = groq_generate_afs_dual(
                n_funcs=candidates_per_llm,
                a_best=feedback_A,
                global_best=feedback_B
            )

            # Evaluate LLM-A candidates
            results_a = []
            for i, code in enumerate(cand_a, start=1):
                tag = f"LLM-A_cand{i}"
                print(f"\n🔎 Evaluating {tag}")
                auc_m, auc_s, acc_m, acc_s = evaluate_candidate_code(code, X_init, y_init, inner_T=inner_T, desc=tag)
                print(f"{tag} | mean AUC={auc_m:.4f} ± {auc_s:.4f} | mean best acc={acc_m:.4f} ± {acc_s:.4f} | EI best acc={ei_acc_mean:.4f}")
                self.save_candidate(g, tag, code)
                results_a.append((code, auc_m, auc_s, acc_m, acc_s))

            # Evaluate LLM-B candidates
            results_b = []
            for i, code in enumerate(cand_b, start=1):
                tag = f"LLM-B_cand{i}"
                print(f"\n🔧 Evaluating {tag}")
                auc_m, auc_s, acc_m, acc_s = evaluate_candidate_code(code, X_init, y_init, inner_T=inner_T, desc=tag)
                print(f"{tag} | mean AUC={auc_m:.4f} ± {auc_s:.4f} | mean best acc={acc_m:.4f} ± {acc_s:.4f} | EI best acc={ei_acc_mean:.4f}")
                self.save_candidate(g, tag, code)
                results_b.append((code, auc_m, auc_s, acc_m, acc_s))

            # Update bests
            if results_a:
                gen_best_a = max(results_a, key=lambda r: r[1])  # by mean AUC
                if gen_best_a[1] > best_A_hist:
                    best_A_hist = gen_best_a[1]
                if gen_best_a[1] > best_global_score:
                    best_global_score = gen_best_a[1]
                    best_global_code = gen_best_a[0]
                    print(f"🎉 New GLOBAL BEST from LLM-A: AUC={best_global_score:.4f}")

            if results_b:
                gen_best_b = max(results_b, key=lambda r: r[1])
                if gen_best_b[1] > best_global_score:
                    best_global_score = gen_best_b[1]
                    best_global_code = gen_best_b[0]
                    print(f"🏆 New GLOBAL BEST from LLM-B: AUC={best_global_score:.4f}")

            # Log summary
            self.log.append({
                "generation": g,
                "ei_mean_auc": ei_auc_mean,
                "ei_std_auc": ei_auc_std,
                "ei_mean_best_acc": ei_acc_mean,
                "ei_std_best_acc": ei_acc_std,
                "A_best_this_gen_auc": max([r[1] for r in results_a], default=None),
                "B_best_this_gen_auc": max([r[1] for r in results_b], default=None),
                "A_hist_best_auc": None if best_A_hist == -np.inf else best_A_hist,
                "global_best_auc": best_global_score
            })
            with open(os.path.join(self.out_dir, "funbo_dual_results.json"), "w") as f:
                json.dump(self.log, f, indent=2)

        print("\n✅ All generations complete.")
        print(f"Final GLOBAL BEST AUC: {best_global_score:.4f}")
        return best_global_code, best_global_score

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-LLM FunBO (Groq) — sequential & RTX-friendly")
    parser.add_argument("--generations", type=int, default=7, help="number of outer generations")
    parser.add_argument("--cands_per_llm", type=int, default=3, help="candidates generated by each LLM per generation")
    parser.add_argument("--inner_T", type=int, default=DEFAULT_INNER_T, help="BO steps per evaluation")
    args = parser.parse_args()

    funbo = FunBOGroqDual(out_dir="funbo_dual_runs")
    best_code, best_score = funbo.run(
        generations=args.generations,
        candidates_per_llm=args.cands_per_llm,
        inner_T=args.inner_T,
    )
    print("\nFinal Best AUC:", best_score)
    print("\nBest AF code:\n")
    print(best_code)
