#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
funbo_core.py

FunBO loop (GP-based inner BO) using kd_objective. This version:
- Uses LLM-C to propose x_init each generation
- Uses two LLMs (A: explore, B: exploit) to propose AFs
- Evaluates EI baseline (for comparison) but never falls back to it
- Uses kd_objective (subprocess) which in turn runs student_kd.py
- Assumes kd_objective.py and student_kd.py are in the same folder
- Keeps outputs in out_dir
"""

import os, json, re, traceback, argparse, time, warnings
import numpy as np
from scipy.stats import norm
from dotenv import load_dotenv
from groq import Groq
from kd_objective import kd_objective
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from tqdm import tqdm

warnings.filterwarnings("ignore")
load_dotenv()

# -----------------------
# Config / Clients
# -----------------------
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
API_KEY_A = os.getenv("GROQ_API_KEY_A")  # exploration (higher temp)
API_KEY_B = os.getenv("GROQ_API_KEY_B")  # exploitation (lower temp)
API_KEY_C = os.getenv("GROQ_API_KEY_C")  # x_init generator
if not (API_KEY_A and API_KEY_B and API_KEY_C):
    raise RuntimeError("Please set GROQ_API_KEY_A, GROQ_API_KEY_B and GROQ_API_KEY_C in your .env")

client_a = Groq(api_key=API_KEY_A)
client_b = Groq(api_key=API_KEY_B)
client_c = Groq(api_key=API_KEY_C)

OUT_DIR = "funbo_runs"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Hyperparams
# -----------------------
DIM = 4
GRID_SIZE = 96
INNER_T = 5
RNG = np.random.default_rng(42)

# -----------------------
# Utils
# -----------------------
def clean_code(text: str) -> str:
    txt = re.sub(r"```(?:python)?", "", text)
    return txt.replace("```", "").strip()

def split_af_functions(output: str, max_funcs=3):
    parts = re.split(r"(?=def\s+AF\d*\s*\()", output)
    funcs = [p.strip() for p in parts if p.strip()]
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
    return {
        "alpha": float(x[0]),
        "beta": float(x[1]) * 0.1,
        "temperature": 2.0 + 18.0 * float(x[2]),
        "learning_rate": 10 ** (-4 + 3.0 * float(x[3])),
        "batch_size": 32,
        "epochs": 3,
    }

# -----------------------
# LLM prompts (no feedback)
# -----------------------
SYSTEM_XINIT = "You are a hyperparameter seed generator for a 4D KD search space. Return json only."
PROMPT_XINIT = """
Generate exactly {N} diverse points in [0,1]^4 as JSON:
{{ "points": [[a,b,c,d], ...] }}
No extra text.
"""

SYSTEM_AF = "You are an expert in Bayesian Optimization. Return ONLY valid Python functions defining def AF(mu, sigma, f_best):"

PROMPT_AF = """
Design {K} distinct acquisition functions for Bayesian Optimization.
Requirements (each):
- def AF(mu, sigma, f_best):
- include imports inside the function
- return a numpy array of same shape as mu
- be numerically stable when sigma->0 (add 1e-9)
- avoid trivial EI/PI/UCB clones
Return ONLY the {K} functions separated by a blank line.
"""

# -----------------------
# LLM wrappers
# -----------------------
def llm_generate_x_init(n_points=4, temp=0.65):
    msg = PROMPT_XINIT.format(N=n_points)
    resp = client_c.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": SYSTEM_XINIT}, {"role": "user", "content": msg}],
        temperature=temp, top_p=0.9, max_tokens=400
    )
    txt = resp.choices[0].message.content.strip()
    try:
        data = json.loads(txt)
        pts = np.array(data["points"], dtype=float)
        pts = np.clip(pts, 0.0, 1.0)
        if pts.shape[1] != DIM:
            raise ValueError("bad shape")
        return pts
    except Exception:
        # fallback to random but keep warning
        print("Warning: x_init parse failed; falling back to RNG.")
        return RNG.random((n_points, DIM))

def llm_generate_afs(client, K=3, temp=0.8):
    msg = PROMPT_AF.format(K=K)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": SYSTEM_AF}, {"role": "user", "content": msg}],
        temperature=temp, top_p=0.95, max_tokens=1200
    )
    raw = resp.choices[0].message.content.strip()
    return split_af_functions(clean_code(raw), max_funcs=K)

# -----------------------
# Inner BO (GP)
# -----------------------
def run_inner_bo(AF_callable, X_init, y_init, T=INNER_T, short=True, desc="BO"):
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
            raise RuntimeError(f"AF callable error: {e}")

        af_vals = np.asarray(af_vals, dtype=float)
        if af_vals.shape[0] != GRID_SIZE:
            raise RuntimeError("AF returned wrong shape")
        if not np.all(np.isfinite(af_vals)):
            raise RuntimeError("AF returned non-finite values")

        x_next = grid[int(np.argmax(af_vals))]
        y_next = kd_objective(unnormalize(x_next), short=short)
        X = np.vstack([X, x_next])
        y = np.append(y, y_next)

    best_acc = 1.0 - float(np.min(y))
    return best_acc

def evaluate_af_code(code_str, X_init, y_init, inner_T=INNER_T, label="AF"):
    AF = compile_af(code_str)
    best_acc = run_inner_bo(AF, X_init, y_init, T=inner_T, short=True, desc=label)
    return best_acc

def evaluate_ei_baseline(X_init, y_init, inner_T=INNER_T):
    return run_inner_bo(expected_improvement, X_init, y_init, T=inner_T, short=True, desc="EI")

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--n_init", type=int, default=4)
    parser.add_argument("--cands_per_llm", type=int, default=3)
    parser.add_argument("--inner_T", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    global_best_acc = -1.0
    global_best_code = None

    # One-time EI reference using random init (not shown to LLMs)
    X0 = RNG.random((args.n_init, DIM))
    y0 = np.array([kd_objective(unnormalize(x), short=True) for x in X0])
    ei_ref = evaluate_ei_baseline(X0, y0, inner_T=args.inner_T)
    print(f"EI reference (random init) best acc: {ei_ref:.4f}")

    for g in range(1, args.generations + 1):
        print(f"\n🚀 Generation {g}/{args.generations}")

        # 1) x_init from LLM-C
        print("x_init generation")
        X_init = llm_generate_x_init(args.n_init, temp=0.65)
        y_init = np.array([kd_objective(unnormalize(x), short=True) for x in X_init])

        # 2) EI baseline on this x_init
        ei_best = evaluate_ei_baseline(X_init, y_init, inner_T=args.inner_T)
        print(f"EI (gen {g}) | best acc={ei_best:.4f}")

        # 3) Generate AF candidates from LLM-A (explore) and LLM-B (exploit)
        try:
            cands_a = llm_generate_afs(client_a, K=args.cands_per_llm, temp=0.9)
            print(f"LLM-A produced {len(cands_a)} candidates")
        except Exception as e:
            print(f"LLM-A generation error: {e}")
            cands_a = []

        try:
            cands_b = llm_generate_afs(client_b, K=args.cands_per_llm, temp=0.35)
            print(f"LLM-B produced {len(cands_b)} candidates")
        except Exception as e:
            print(f"LLM-B generation error: {e}")
            cands_b = []

        # 4) Evaluate candidates
        gen_best_acc = -1.0
        gen_best_code = None

        def eval_and_log(code_str, tag):
            nonlocal gen_best_acc, gen_best_code
            try:
                acc = evaluate_af_code(code_str, X_init, y_init, inner_T=args.inner_T, label=tag)
                print(f"{tag} | best_acc={acc:.4f} | EI_best={ei_best:.4f}")
                fname = os.path.join(args.out_dir, f"gen{g}_{tag}_bestAcc{acc:.6f}.py")
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(code_str)
                if acc > gen_best_acc:
                    gen_best_acc = acc
                    gen_best_code = code_str
            except Exception as e:
                print(f"{tag} failed: {e}")

        for i, code in enumerate(cands_a, start=1):
            eval_and_log(code, f"LLM-A_cand{i}")
        for i, code in enumerate(cands_b, start=1):
            eval_and_log(code, f"LLM-B_cand{i}")

        # Update global best
        if gen_best_code is not None and gen_best_acc > global_best_acc:
            global_best_acc = gen_best_acc
            global_best_code = gen_best_code
            with open(os.path.join(args.out_dir, f"gen{g}_GLOBAL_BEST_{global_best_acc:.6f}.py"), "w", encoding="utf-8") as f:
                f.write(global_best_code)
            print(f"🏆 New global best acc: {global_best_acc:.4f}")

        # Log summary
        summary = {
            "generation": g,
            "EI_best_acc": float(ei_best),
            "gen_best_acc": float(gen_best_acc if gen_best_acc > -1 else 0.0),
            "global_best_acc": float(global_best_acc if global_best_acc > -1 else 0.0)
        }
        with open(os.path.join(args.out_dir, "run_summary.json"), "a") as f:
            f.write(json.dumps(summary) + "\n")
        print(json.dumps(summary))

    print("\nFinal Global Best Accuracy:", f"{global_best_acc:.4f}" if global_best_acc >= 0 else "N/A")
    if global_best_code:
        print("Final Best AF code saved in:", args.out_dir)

if __name__ == "__main__":
    main()
