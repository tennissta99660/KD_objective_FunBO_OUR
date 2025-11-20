#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
funbo_improvement_core.py

Three-LLM FunBO core (Groq):
- LLM-A (exploration, high temp) generates 3 unique AFs (no few-shot).
- LLM-B (exploitation, low temp) generates 3 AFs guided by light few-shot.
- LLM-C proposes smart X_init (4 points) each generation, replacing random init.

Design:
- No SQL cache, no repeats-per-AF (fast, low FLOPs).
- AUC-based scoring of BO trace + best accuracy per candidate.
- EI baseline evaluated once per generation for comparison.
- Robust to bad AFs (runtime errors → penalized).
- Prints per-candidate metrics and saves AF code files.

Requires:
- kd_objective.py and student_kd.py in your project as before.
"""

import os
import json
import re
import argparse
import time
from datetime import datetime
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
load_dotenv()

# -----------------------------
# Config
# -----------------------------
GRID_SIZE = 128          # number of points sampled per BO step
DEFAULT_INNER_T = 5      # BO steps per candidate
N_INIT_POINTS = 4        
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Groq model + API keys
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
API_KEY_A = os.getenv("GROQ_API_KEY_A")  # exploration
API_KEY_B = os.getenv("GROQ_API_KEY_B")  # exploitation
API_KEY_C = os.getenv("GROQ_API_KEY_C")  # X_init proposer

if not API_KEY_A or not API_KEY_B or not API_KEY_C:
    raise RuntimeError("Please set GROQ_API_KEY_A, GROQ_API_KEY_B, GROQ_API_KEY_C in .env")

client_a = Groq(api_key=API_KEY_A)
client_b = Groq(api_key=API_KEY_B)
client_c = Groq(api_key=API_KEY_C)

# -----------------------------
# Utility: search space mapping
# -----------------------------
def unnormalize(x):
    """
    Map [0,1]^4 -> KD hyperparameters.
    dims: [alpha, weight_decay, temperature, learning_rate]
    """
    return {
        "alpha": float(x[0]),
        "weight_decay": 10 ** (-5 + 3.0 * float(x[1])),  # 1e-5 .. 1e-2
        "temperature": 2.0 + 18.0 * float(x[2]),         # 2 .. 20
        "learning_rate": 10 ** (-4 + 3.0 * float(x[3])), # 1e-4 .. 1e-1
        "batch_size": 32,
        "epochs": 3,
    }

# -----------------------------
# AF parsing/compilation
# -----------------------------
def clean_llm_code(output: str) -> str:
    c = re.sub(r"```(python)?", "", output)
    return c.replace("```", "").strip()

def split_candidates(output: str, max_funcs=3):
    """
    Extract multiple 'def AF...' functions. Also tolerate AF1/AF2 names and rename to AF.
    """
    # Split on def AF or def AF[digits]
    parts = re.split(r"(?=def\s+AF\d*\s*\()", output.strip())
    funcs = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # keep only the function block (stop at next def or EOF)
        # heuristic: if multiple defs in one part, keep up to first double newline-def
        # but simplest: keep full chunk
        code = re.sub(r"def\s+AF\d*\s*\(", "def AF(", p)
        if "def AF(" in code:
            funcs.append(code)
        if len(funcs) >= max_funcs:
            break
    return funcs

def compile_af(code_str):
    """
    Compile AF string into callable.
    We still provide np/norm in globals in case the model forgot imports,
    but prompts enforce imports-inside-def.
    """
    safe_globals = {"np": np, "norm": norm}
    try:
        exec(code_str, safe_globals)
    except Exception as e:
        raise RuntimeError(f"Compilation failed: {e}")
    af = safe_globals.get("AF")
    if not callable(af):
        # try to find any callable with AF in name
        cands = [v for k, v in safe_globals.items() if callable(v) and "AF" in k]
        if cands:
            af = cands[0]
        else:
            raise RuntimeError("Compiled AF does not define AF(mu, sigma, f_best).")
    return af

# -----------------------------
# Scoring helpers
# -----------------------------
def expected_improvement(mu, sigma, f_best):
    z = (f_best - mu) / (sigma + 1e-9)
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

def auc_from_losses(best_losses):
    """
    Given best_losses over time (lower is better; they are 1-acc),
    convert to rewards (acc) and average = AUC-like summary.
    """
    rewards = [1.0 - float(l) for l in best_losses]
    return float(np.mean(rewards))

# -----------------------------
# Inner BO (single run; no repeats)
# -----------------------------
def run_inner_bo_single(AF_callable, X_init, y_init, T=DEFAULT_INNER_T, short=False, desc="BO"):
    """
    One BO trace with GP surrogate + AF.
    Returns: auc_score, best_acc, best_losses_series
    """
    dim = X_init.shape[1]
    X, y = X_init.copy(), y_init.copy()

    kernel = C(1.0, (1e-5, 1e5)) * RBF(length_scale=np.ones(dim), length_scale_bounds=(1e-5, 1e5))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

    best_losses = [float(np.min(y))]

    for _ in tqdm(range(T), desc=desc, leave=False):
        gp.fit(X, y)
        grid = np.random.rand(GRID_SIZE, dim)
        mu, sigma = gp.predict(grid, return_std=True)

        try:
            af_vals = AF_callable(mu, sigma, float(np.min(y)))
            af_vals = np.asarray(af_vals, dtype=float)
        except Exception as e:
            # bad AF -> penalize heavily
            return 0.0, 0.0, best_losses

        # handle nan/inf
        af_vals = np.nan_to_num(af_vals, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        idx = int(np.argmax(af_vals))
        x_next = grid[idx]

        # Evaluate KD at unnormalized params
        loss_next = kd_objective(unnormalize(x_next), short=short)
        X = np.vstack([X, x_next])
        y = np.append(y, loss_next)

        best_losses.append(float(np.min(y)))

    auc = auc_from_losses(best_losses)
    best_acc = 1.0 - float(np.min(best_losses))
    return auc, best_acc, best_losses

# -----------------------------
# LLM prompts
# -----------------------------
SYSTEM_PROMPT = (
    "You are an expert in Bayesian Optimization and Machine Learning. "
    "All outputs must be valid Python code only (no markdown, no prose). "
    "Return only function definitions named `AF` with signature `def AF(mu, sigma, f_best):`."
)

FEW_SHOT = """
def AF_example1(mu, sigma, f_best):
    import numpy as np
    z = (f_best - mu) / (np.maximum(sigma, 1e-9))
    return np.sqrt(np.maximum(sigma, 1e-9)) * np.exp(-0.5 * z**2) + np.maximum(0, f_best - mu)

def AF_example2(mu, sigma, f_best):
    import numpy as np
    from scipy.stats import norm
    s = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / s
    return (f_best - mu) * norm.cdf(z) + s * norm.pdf(z) * (1 + np.tanh(z))
"""

def groq_generate_afs_A(best_score_explorer: float):
    """
    LLM-A (exploration): high temperature, no examples. Generate 3 distinct AFs.
    Receives its own best score (not global).
    """
    prompt = f"""
Design 3 distinct and novel acquisition functions for Bayesian Optimization in Knowledge Distillation.
Constraints:
- Each must be a valid Python function exactly named `AF` with signature `def AF(mu, sigma, f_best):`
- Include necessary imports INSIDE each function (e.g., `import numpy as np`, `from scipy.stats import norm` if used).
- Return a numpy array with the same shape as `mu`.
- Ensure numerical stability (use np.maximum(sigma, 1e-9)).
- Avoid reusing standard EI/PI/UCB exactly; encourage creative nonlinear interactions.
- Do not include markdown, explanations, comments, or text outside function bodies.

Your historical best AUC score so far (exploration-only): {best_score_explorer:.6f}

Output exactly 3 function definitions, separated by one blank line.
"""
    try:
        resp = client_a.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            temperature=0.85,      # higher temp for exploration
            max_tokens=1200,
            top_p=1.0,
        )
        raw = resp.choices[0].message.content.strip()
        return split_candidates(clean_llm_code(raw), max_funcs=3)
    except Exception as e:
        print(f"❌ LLM-A error: {e}")
        return []

def groq_generate_afs_B(global_best_score: float):
    """
    LLM-B (exploitation): lower temperature, few-shot guided. Generate 3 AFs.
    Receives global best score as feedback.
    """
    prompt = f"""
Generate 3 acquisition functions for Bayesian Optimization in Knowledge Distillation.
Use the following examples for light guidance but do not copy them verbatim:
{FEW_SHOT}

Constraints:
- Each must be a valid Python function exactly named `AF` with signature `def AF(mu, sigma, f_best):`
- Include necessary imports INSIDE each function.
- Return a numpy array with the same shape as `mu`.
- Ensure numerical stability using np.maximum(sigma, 1e-9).

Global best AUC score so far: {global_best_score:.6f}

Output exactly 3 function definitions, separated by one blank line.
"""
    try:
        resp = client_b.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            temperature=0.35,      # lower temp for exploitation
            max_tokens=1200,
            top_p=0.4,
        )
        raw = resp.choices[0].message.content.strip()
        return split_candidates(clean_llm_code(raw), max_funcs=3)
    except Exception as e:
        print(f"❌ LLM-B error: {e}")
        return []

def groq_propose_x_init(best_af_code: str, prev_global_best: float, prev_x_init=None, n_points=N_INIT_POINTS):
    """
    LLM-C proposes N_INIT_POINTS normalized points in [0,1]^4 for X_init.
    Returns np.ndarray of shape (n_points, 4).
    """
    spec = {
        "dims": [
            {"name": "alpha", "range": [0.0, 1.0]},
            {"name": "weight_decay_log", "range": [0.0, 1.0]},
            {"name": "temperature_norm", "range": [0.0, 1.0]},
            {"name": "learning_rate_log", "range": [0.0, 1.0]},
        ],
        "n_points": n_points
    }

    prompt = f"""
You propose initial normalized points X_init for Bayesian Optimization in Knowledge Distillation.
- Produce exactly {n_points} points in [0,1]^4 (alpha, weight_decay_log, temperature_norm, learning_rate_log).
- Focus on diversity but bias around historically good regions if possible.
- Return ONLY a JSON array of arrays, no text, no comments.

Historical summary:
- Previous global best AUC: {prev_global_best:.6f}

Current best AF code for reference (do not modify it, only use as signal):
{best_af_code}

Previous X_init (if any) to avoid duplicates:
{json.dumps(prev_x_init) if prev_x_init is not None else "null"}
"""
    try:
        resp = client_c.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": "You return only pure JSON arrays. No prose."},
                      {"role": "user", "content": prompt}],
            temperature=0.45,
            max_tokens=600,
            top_p=0.9,
        )
        raw = resp.choices[0].message.content.strip()
        # try parse JSON; if fails, fallback random
        try:
            arr = json.loads(raw)
            arr = np.array(arr, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 4:
                raise ValueError("Bad shape for X_init JSON.")
            # clip to [0,1]
            arr = np.clip(arr, 0.0, 1.0)
            # ensure exactly n_points rows
            if arr.shape[0] < n_points:
                # pad with randoms
                extra = np.random.rand(n_points - arr.shape[0], 4)
                arr = np.vstack([arr, extra])
            elif arr.shape[0] > n_points:
                arr = arr[:n_points, :]
            return arr
        except Exception:
            print("⚠️ LLM-C returned non-JSON or invalid JSON. Falling back to random X_init.")
            return np.random.rand(n_points, 4)
    except Exception as e:
        print(f"❌ LLM-C error: {e}. Falling back to random X_init.")
        return np.random.rand(n_points, 4)

# -----------------------------
# Baseline AF (EI)
# -----------------------------
def base_ei_code():
    return """
def AF(mu, sigma, f_best):
    import numpy as np
    from scipy.stats import norm
    s = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / s
    return (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
"""

# -----------------------------
# Core runner
# -----------------------------
class FunBODualGroq:
    def __init__(self, out_dir="funbo_dual_runs", seed=RANDOM_SEED):
        np.random.seed(seed)
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.log = []

        self.best_global_score = -np.inf
        self.best_global_code = base_ei_code()

        self.best_A_score = -np.inf  # exploration best (for A's feedback)
        self.prev_x_init = None      # for LLM-C feedback

    def save_af(self, gen, tag, auc, best_acc, code):
        fname = f"gen{gen}_{tag}_AUC{auc:.6f}_bestAcc{best_acc:.6f}.py"
        path = os.path.join(self.out_dir, fname)
        with open(path, "w") as f:
            f.write(code)
        print(f"💾 Saved: {path}")

    def run(self, generations=5, inner_T=DEFAULT_INNER_T):
        # Initial EI baseline with a random X_init just for reference (not reused later)
        print("Evaluating EI baseline...")
        ei_callable = compile_af(base_ei_code())
        X0 = np.random.rand(N_INIT_POINTS, 4)
        y0 = np.array([kd_objective(unnormalize(x), short=True) for x in X0])
        ei_auc, ei_best_acc, _ = run_inner_bo_single(ei_callable, X0, y0, T=inner_T, short=True, desc="EI baseline")
        print(f"EI | mean AUC={ei_auc:.4f} | best acc={ei_best_acc:.4f}")

        # set global best to EI as starting point
        self.best_global_score = ei_auc
        self.best_global_code = base_ei_code()

        for g in range(1, generations + 1):
            print(f"\n🚀 Generation {g}/{generations}")

            # 1) LLM-C proposes smart X_init
            X_init = groq_propose_x_init(self.best_global_code, self.best_global_score, self.prev_x_init, n_points=N_INIT_POINTS)
            self.prev_x_init = X_init.tolist()
            y_init = np.array([kd_objective(unnormalize(x), short=True) for x in X_init])

            # 2) Evaluate EI on this same X_init for fair comparison
            ei_auc_g, ei_best_acc_g, _ = run_inner_bo_single(
                ei_callable, X_init, y_init, T=inner_T, short=True, desc=f"EI (gen {g})"
            )
            print(f"EI (gen {g}) | AUC={ei_auc_g:.4f} | best acc={ei_best_acc_g:.4f}")

            # 3) Generate candidates from both LLMs
            cand_codes_A = groq_generate_afs_A(best_score_explorer=max(0.0, self.best_A_score))
            cand_codes_B = groq_generate_afs_B(global_best_score=max(0.0, self.best_global_score))
            print(f"🧠 LLM-A produced {len(cand_codes_A)} candidates.")
            print(f"🧩 LLM-B produced {len(cand_codes_B)} candidates.")

            results_this_gen = []

            # 4) Evaluate A-candidates
            for i, code in enumerate(cand_codes_A, start=1):
                tag = f"LLM-A_cand{i}"
                try:
                    af = compile_af(code)
                    auc, best_acc, _ = run_inner_bo_single(af, X_init, y_init, T=inner_T, short=True, desc=tag)
                    print(f"{tag} | AUC={auc:.4f} | best_acc={best_acc:.4f} | EI_best={ei_best_acc_g:.4f}")
                    self.save_af(g, tag, auc, best_acc, code)
                    results_this_gen.append((tag, code, auc, best_acc, "A"))
                    # update exploration-best for feedback to A
                    if auc > self.best_A_score:
                        self.best_A_score = auc
                except Exception as e:
                    print(f"❌ {tag} failed: {e}")

            # 5) Evaluate B-candidates
            for i, code in enumerate(cand_codes_B, start=1):
                tag = f"LLM-B_cand{i}"
                try:
                    af = compile_af(code)
                    auc, best_acc, _ = run_inner_bo_single(af, X_init, y_init, T=inner_T, short=True, desc=tag)
                    print(f"{tag} | AUC={auc:.4f} | best_acc={best_acc:.4f} | EI_best={ei_best_acc_g:.4f}")
                    self.save_af(g, tag, auc, best_acc, code)
                    results_this_gen.append((tag, code, auc, best_acc, "B"))
                except Exception as e:
                    print(f"❌ {tag} failed: {e}")

            # 6) Pick the generation winner and update global best
            if results_this_gen:
                gen_winner = max(results_this_gen, key=lambda t: t[2])  # by AUC
                if gen_winner[2] > self.best_global_score:
                    print(f"🎉 New global best from {gen_winner[0]}: AUC={gen_winner[2]:.6f}")
                    self.best_global_score = gen_winner[2]
                    self.best_global_code = gen_winner[1]

            # 7) Log summary
            self.log.append({
                "generation": g,
                "time": datetime.utcnow().isoformat(),
                "EI_AUC": float(ei_auc_g),
                "EI_best_acc": float(ei_best_acc_g),
                "A_best_feedback": float(max(0.0, self.best_A_score)),
                "global_best_AUC": float(self.best_global_score),
            })
            with open(os.path.join(self.out_dir, "funbo_dual_log.json"), "w") as f:
                json.dump(self.log, f, indent=2)

        print("\n✅ All generations complete.")
        print(f"Final global best AUC: {self.best_global_score:.6f}")
        return self.best_global_code, self.best_global_score

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--inner_T", type=int, default=DEFAULT_INNER_T)
    parser.add_argument("--out_dir", type=str, default="funbo_dual_runs")
    args = parser.parse_args()

    funbo = FunBODualGroq(out_dir=args.out_dir)
    best_code, best_score = funbo.run(generations=args.generations, inner_T=args.inner_T)
    print("\nFinal Best Score (AUC):", best_score)
    print("\nBest AF code:\n")
    print(best_code)
