#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
funbo_core_nofeedback.py

- No feedback to LLMs (no scores, no baselines, no prior code)
- Classic FunBO-style loop:
  * Each generation: get fresh x_init via LLM-C
  * Evaluate EI baseline on that x_init (for comparison only)
  * LLM-A (explore, high temp) generates AF candidates
  * LLM-B (exploit, low temp) generates AF candidates
  * Evaluate all AFs with GP-based inner BO (like FunBO)
  * Pick global best AF by best validation accuracy
- Never fallback to EI; EI is only a baseline comparator.
- If LLM output is unparsable, that candidate is skipped.
"""

import os
import re
import json
import argparse
import warnings
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from scipy.stats import norm
from tqdm import tqdm
from kd_objective import kd_objective
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

warnings.filterwarnings("ignore")
load_dotenv()

# ==========
# Groq setup
# ==========
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
API_KEY_A = os.getenv("GROQ_API_KEY_A")  # exploration
API_KEY_B = os.getenv("GROQ_API_KEY_B")  # exploitation
API_KEY_C = os.getenv("GROQ_API_KEY_C")  # x_init generator

if not (API_KEY_A and API_KEY_B and API_KEY_C):
    raise RuntimeError("Please set GROQ_API_KEY_A, GROQ_API_KEY_B and GROQ_API_KEY_C in your .env")

client_a = Groq(api_key=API_KEY_A)
client_b = Groq(api_key=API_KEY_B)
client_c = Groq(api_key=API_KEY_C)

# =================
# Search parameters
# =================
GRID_SIZE = 96          # grid size per BO step (kept small for speed; raise for quality)
INNER_T = 5             # BO steps per evaluation of an AF
DIM = 4                 # alpha, beta, temperature, learning_rate
RNG = np.random.default_rng(42)

# ==========
# Utilities
# ==========
def clean_code(text: str) -> str:
    txt = re.sub(r"```(?:python)?", "", text)
    return txt.replace("```", "").strip()

def split_af_functions(output: str, max_funcs=3):
    # Split at function boundaries like def AF(...) or def AF1(...)
    parts = re.split(r"(?=def\s+AF\d*\s*\()", output)
    funcs = [p.strip() for p in parts if p.strip()]
    # Standardize name to def AF(...
    fixed = [re.sub(r"def\s+AF\d*\s*\(", "def AF(", f) for f in funcs]
    return fixed[:max_funcs]

def compile_af(code_str):
    safe_globals = {"np": np, "norm": norm}
    try:
        exec(code_str, safe_globals)
    except Exception as e:
        raise RuntimeError(f"Compilation failed: {e}")
    af = safe_globals.get("AF")
    if not callable(af):
        # try any callable with AF in name
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
    # x in [0,1]^4  -> KD hparams
    return {
        "alpha": float(x[0]),
        "beta": float(x[1]) * 0.1,
        "temperature": 2.0 + 18.0 * float(x[2]),
        "learning_rate": 10 ** (-4 + 3.0 * float(x[3])),
        "batch_size": 32,
        "epochs": 3,
    }

# ===========================
# LLM prompts (no feedback!)
# ===========================
SYSTEM_AF = (
    "You are an expert in Bayesian Optimization for hyperparameter tuning of Knowledge Distillation. "
    "Return ONLY valid Python function definitions. No markdown, no comments, no prose."
)

PROMPT_AF_OBJECTIVE = """
Design {K} distinct acquisition functions for Bayesian Optimization.

Hard requirements for EACH function:
- It MUST define: def AF(mu, sigma, f_best):
- It MUST import its own dependencies INSIDE the function (e.g., numpy as np, scipy.stats.norm if needed)
- It MUST return a numpy array of acquisition values with the same shape as mu
- It MUST be numerically stable when sigma -> 0 (e.g., add 1e-9)
- It MUST be differentiable (use smooth nonlinearities)
- It MUST NOT be a trivial Expected Improvement / Probability of Improvement / UCB clone
- It MUST avoid extreme values that create overflows (e.g., exp(large)), and avoid runtime errors

Do NOT include any explanation or text — ONLY the function definitions, separated by one blank line.
"""

SYSTEM_XINIT = (
    "You are a hyperparameter seed generator for a 4D KD search space."
)

PROMPT_XINIT = """
Generate exactly {N} initial points x in [0,1]^4 for Bayesian Optimization.
Space ordering: [alpha, beta_scale (0..0.1 scaled by value), temperature_scale (0..1 for mapping to 2..20), learning_rate_scale (0..1 for mapping 1e-4..1e-1)].

Constraints:
- Ensure diversity: include at least one near the lower edge, one near the upper edge, and some mid-space points.
- Avoid clustering: points should be well-spread across the hypercube.
- Output strictly JSON with the schema:
{{
  "points": [[a1,b1,t1,l1], [a2,b2,t2,l2], ...]
}}
No extra text. No markdown. No code block. Just JSON.
"""

def llm_generate_x_init(n_points: int):
    msg = PROMPT_XINIT.format(N=n_points)
    resp = client_c.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": SYSTEM_XINIT},
                  {"role": "user", "content": msg}],
        temperature=0.65, top_p=0.9, max_tokens=600,
    )
    content = resp.choices[0].message.content.strip()
    # Parse JSON
    try:
        data = json.loads(content)
        pts = np.array(data["points"], dtype=float)
        if pts.shape[1] != 4:
            raise ValueError("points must be 2D with 4 columns")
        # Clamp to [0,1]
        pts = np.clip(pts, 0.0, 1.0)
        return pts
    except Exception:
        warnings.warn("LLM-C returned unparsable x_init; falling back to random.")
        return RNG.random((n_points, DIM))

def llm_generate_afs_explore(K: int = 3):
    # Exploration: higher diversity, higher randomness
    user = PROMPT_AF_OBJECTIVE.format(K=K)
    resp = client_a.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": SYSTEM_AF},
                  {"role": "user", "content": user}],
        temperature=0.9, top_p=0.95, max_tokens=1200,
    )
    raw = resp.choices[0].message.content.strip()
    return split_af_functions(clean_code(raw), max_funcs=K)

def llm_generate_afs_exploit(K: int = 3):
    # Exploitation: more conservative, crisper sampling
    user = PROMPT_AF_OBJECTIVE.format(K=K)
    resp = client_b.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": SYSTEM_AF},
                  {"role": "user", "content": user}],
        temperature=0.35, top_p=0.3, max_tokens=1200,
    )
    raw = resp.choices[0].message.content.strip()
    return split_af_functions(clean_code(raw), max_funcs=K)

# ==========================
# Inner BO (FunBO-like GP)
# ==========================
def run_inner_bo(AF_callable, X_init, y_init, T=INNER_T, short=True, desc="BO"):
    """
    GP-based inner loop (classic FunBO style).
    Returns best validation accuracy achieved during the run.
    """
    dim = X_init.shape[1]
    X, y = X_init.copy(), y_init.copy()
    kernel = C(1.0) * RBF(length_scale=np.ones(dim))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

    for _ in tqdm(range(T), desc=desc, leave=False):
        gp.fit(X, y)
        grid = RNG.random((GRID_SIZE, dim))
        mu, sigma = gp.predict(grid, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        f_best = float(np.min(y))

        try:
            af_vals = AF_callable(mu, sigma, f_best)
        except Exception as e:
            # Hard skip this AF by returning current best accuracy (no fallback to EI)
            raise RuntimeError(f"AF callable error: {e}")

        af_vals = np.asarray(af_vals, dtype=float)
        if not np.all(np.isfinite(af_vals)) or af_vals.shape[0] != GRID_SIZE:
            raise RuntimeError("AF produced invalid values or wrong shape.")

        x_next = grid[int(np.argmax(af_vals))]
        y_next = kd_objective(unnormalize(x_next), short=short)  # returns 1 - acc
        X = np.vstack([X, x_next])
        y = np.append(y, y_next)

    best_acc = 1.0 - float(np.min(y))
    return best_acc

def evaluate_af_code(code_str, X_init, y_init, inner_T=INNER_T, label="AF"):
    """
    Compile and evaluate AF code string. Returns best_acc.
    If compilation/runtime fails, raises (caller will skip).
    """
    AF = compile_af(code_str)
    best_acc = run_inner_bo(AF, X_init, y_init, T=inner_T, short=True, desc=label)
    return best_acc

# =========
# Baseline
# =========
def evaluate_ei_baseline(X_init, y_init, inner_T=INNER_T):
    AF = expected_improvement
    return run_inner_bo(AF, X_init, y_init, T=inner_T, short=True, desc="EI")

# =====
# Main
# =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--n_init", type=int, default=4)
    parser.add_argument("--cands_per_llm", type=int, default=3)
    parser.add_argument("--inner_T", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="funbo_dual_runs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    global_best_code = None
    global_best_acc = -1.0

    # A one-time EI reference (optional, using random init) — not shown to LLMs
    X0 = RNG.random((args.n_init, DIM))
    y0 = np.array([kd_objective(unnormalize(x), short=True) for x in X0])
    ei_ref = evaluate_ei_baseline(X0, y0, inner_T=args.inner_T)
    print(f"EI reference (random init) best acc: {ei_ref:.4f}")

    for g in range(1, args.generations + 1):
        print(f"\n🚀 Generation {g}/{args.generations}")

        # Fresh x_init from LLM-C (no feedback)
        print(f"x_init Generation {g}/{args.generations}")
        X_init = llm_generate_x_init(args.n_init)
        y_init = np.array([kd_objective(unnormalize(x), short=True) for x in X_init])

        # EI baseline on THIS x_init
        ei_best = evaluate_ei_baseline(X_init, y_init, inner_T=args.inner_T)
        print(f"EI (gen {g}) | best acc={ei_best:.4f}")

        # Generate AF candidates (no feedback)
        try:
            cands_a = llm_generate_afs_explore(args.cands_per_llm)
            print(f"🧠 LLM-A produced {len(cands_a)} candidates.")
        except Exception as e:
            print(f"LLM-A generation error: {e}")
            cands_a = []

        try:
            cands_b = llm_generate_afs_exploit(args.cands_per_llm)
            print(f"🧩 LLM-B produced {len(cands_b)} candidates.")
        except Exception as e:
            print(f"LLM-B generation error: {e}")
            cands_b = []

        # Evaluate all candidates
        gen_best_code = None
        gen_best_acc = -1.0

        def eval_and_log(code_str, tag):
            nonlocal gen_best_acc, gen_best_code
            try:
                acc = evaluate_af_code(code_str, X_init, y_init, inner_T=args.inner_T, label=tag)
                print(f"{tag} | best_acc={acc:.4f} | EI_best={ei_best:.4f}")
                # Save candidate for inspection
                fname = os.path.join(args.out_dir, f"gen{g}_{tag}_bestAcc{acc:.6f}.py")
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(code_str)
                print(f"💾 Saved: {fname}")
                if acc > gen_best_acc:
                    gen_best_acc = acc
                    gen_best_code = code_str
            except Exception as e:
                print(f"{tag} failed: {e}")

        for idx, code in enumerate(cands_a, start=1):
            eval_and_log(code, f"LLM-A_cand{idx}")
        for idx, code in enumerate(cands_b, start=1):
            eval_and_log(code, f"LLM-B_cand{idx}")

        # Update global best (FunBO-style: keep best code across generations)
        if gen_best_code is not None and gen_best_acc > global_best_acc:
            global_best_acc = gen_best_acc
            global_best_code = gen_best_code
            with open(os.path.join(args.out_dir, f"gen{g}_GLOBAL_BEST_{global_best_acc:.6f}.py"), "w", encoding="utf-8") as f:
                f.write(global_best_code)
            print(f"🏆 New global best acc: {global_best_acc:.4f}")

        # Log compact JSON line per generation
        gen_summary = {
            "generation": g,
            "A_best_acc": float(max([0.0] + [0.0])),  # kept simple; detailed per-candidate already printed
            "B_best_acc": None,                       # (you can extend to track per-LLM best if needed)
            "EI_best_acc": float(ei_best),
            "global_best": float(global_best_acc if global_best_acc >= 0 else 0.0)
        }
        print(json.dumps(gen_summary))

    print("\nFinal Global Best Accuracy:", f"{global_best_acc:.4f}" if global_best_acc >= 0 else "N/A")
    if global_best_code:
        print("Final Best AF code saved in:", args.out_dir)

if __name__ == "__main__":
    main()
