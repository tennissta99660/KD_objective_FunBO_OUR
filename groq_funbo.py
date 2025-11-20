#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FunBO Core (Groq Dual-LLM Coevolution Version)

- LLM-A (Exploration): Generates creative, structurally novel AFs.
  → Receives feedback from *its own best score*.

- LLM-B (Exploitation): Generates guided, improved AFs.
  → Receives feedback from the *global best score (A or B)*.

Both run per generation, producing candidate AFs that compete via inner Bayesian optimization.
"""

import os, json, re, traceback, argparse, time
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

# Suppress only this specific warning from scikit-learn
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ============================================================
# Setup
# ============================================================

load_dotenv()

API_KEY_A = os.getenv("GROQ_API_KEY_A")
API_KEY_B = os.getenv("GROQ_API_KEY_B")
if not API_KEY_A or not API_KEY_B:
    raise RuntimeError("Please set GROQ_API_KEY_A and GROQ_API_KEY_B in your .env file.")

client_a = Groq(api_key=API_KEY_A)
client_b = Groq(api_key=API_KEY_B)
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ============================================================
# Utility
# ============================================================

def clean_llm_code(output: str) -> str:
    cleaned = re.sub(r"```(python)?", "", output)
    cleaned = cleaned.replace("```", "").strip()
    return cleaned

def split_candidates(output: str):
    """Split multiple AF definitions, rename AF1/2/3 to AF."""
    parts = re.split(r"(?=def\s+AF\d*\s*\()", output)
    candidates = [p.strip() for p in parts if p.strip()]
    fixed = [re.sub(r"def\s+AF\d*\s*\(", "def AF(", c) for c in candidates]
    return fixed[:3]

def compile_af(code_str):
    safe_globals = {}
    try:
        exec(code_str, safe_globals)
    except Exception as e:
        raise RuntimeError(f"Compilation failed: {e}")
    af = safe_globals.get("AF")
    if not callable(af):
        funcs = [v for k, v in safe_globals.items() if callable(v) and "AF" in k]
        if funcs:
            af = funcs[0]
        else:
            raise RuntimeError("Compiled AF does not define AF(mu, sigma, f_best).")
    return af

def expected_improvement(mu, sigma, f_best):
    z = (f_best - mu) / (sigma + 1e-9)
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

def unnormalize(x):
    # FIX: Replaced unused 'beta' (x[1]) with 'weight_decay' on a log scale.
    # The 'student_kd.py' script uses 'weight_decay', not 'beta'.
    # This ensures all 4 optimized dimensions are active.
    return {
        "alpha": float(x[0]), # Range [0.0, 1.0]
        "weight_decay": 10 ** (-5 + 3.0 * float(x[1])), # Range [1e-5, 1e-2]
        "temperature": 2.0 + 18.0 * float(x[2]), # Range [2.0, 20.0]
        "learning_rate": 10 ** (-4 + 3.0 * float(x[3])), # Range [1e-4, 1e-1]
        "batch_size": 32, # Fixed
        "epochs": 3, # Fixed
    }

# ============================================================
# Dual-Llama Generation
# ============================================================

FEW_SHOT_EXAMPLES = """
def AF_example1(mu, sigma, f_best):
    import numpy as np
    from scipy.stats import norm
    z = (f_best - mu) / (sigma + 1e-9)
    return sigma * np.exp(-0.5 * z**2) + np.maximum(0, f_best - mu)

def AF_example2(mu, sigma, f_best):
    import numpy as np
    z = np.tanh((f_best - mu) / (sigma + 1e-9))
    return (1 - z) * np.sqrt(sigma) + 0.3 * sigma * np.exp(-np.abs(z))
"""

SYSTEM_PROMPT = (
    "You are an expert in Bayesian Optimization and Machine Learning. "
    "All outputs must be valid Python — no markdown, comments, or text. "
    "Return only function definitions."
)

def groq_generate_af_batch(best_code: str, feedback_a: str, feedback_b: str):
    """Generate candidates from both LLMs with independent feedback."""

    # LLM-A (Exploration)
    # FIX: Removed extraneous quotes ("") inside the f-string
    prompt_a = f"""
You are a top researcher in Bayesian Optimization and Knowledge Distillation.
Generate 3 distinct acquisition functions for Bayesian Optimization.
Write valid, standalone Python functions that define `def AF(mu, sigma, f_best):`.
Requirements:
- Define def AF(mu, sigma, f_best):.
- Include imports inside the function.
- Return a numpy array with the same shape as `mu`.
- Be numerically stable when sigma -> 0 (avoid division by zero).
- Be differentiable w.r.t. mu and sigma (use smooth functions).
- Avoid standard EI, PI, UCB formulations or trivially copying them.
- Use nonlinear interactions (multiplicative mu*sigma, logs, tanh, sigmoid, ratios, etc.).
Do not include explanations, markdown, or extraneous text — only Python code.

Baseline (for reference):
{best_code}

Feedback from your previous generation (LLM-A only):
{feedback_a}

Output exactly 3 valid Python functions separated by one blank line.
"""

    # LLM-B (Exploitation)
    prompt_b = f"""
Generate 3 acquisition functions for Bayesian Optimization.
Each must:
- Define def AF(mu, sigma, f_best):
- Include imports inside each function
- Be stable, differentiable, and efficient
- Balance exploration and exploitation
- Take light inspiration from the following examples (do not copy):
{FEW_SHOT_EXAMPLES}

Baseline (for reference):
{best_code}

Feedback (best global score so far):
{feedback_b}

Output exactly 3 valid Python functions separated by one blank line.
"""

    # Query LLM-A
    try:
        response_a = client_a.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_a},
            ],
            temperature=0.85,
            max_tokens=1000,
            top_p=1.0,
        )
        raw_a = response_a.choices[0].message.content.strip()
        candidates_a = split_candidates(clean_llm_code(raw_a))
        print(f"🧠 LLM-A generated {len(candidates_a)} candidates.")
    except Exception as e:
        print(f"❌ LLM-A error: {e}")
        candidates_a = []

    # Query LLM-B
    try:
        response_b = client_b.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_b},
            ],
            temperature=0.45, # Lowered temp for exploitation
            max_tokens=1000,
            top_p=0.3, # Lowered top_p for exploitation
        )
        raw_b = response_b.choices[0].message.content.strip()
        candidates_b = split_candidates(clean_llm_code(raw_b))
        print(f"🧩 LLM-B generated {len(candidates_b)} candidates.")
    except Exception as e:
        print(f"❌ LLM-B error: {e}")
        candidates_b = []

    return candidates_a, candidates_b


# ============================================================
# Bayesian Optimization
# ============================================================

def run_inner_bo(AF_callable, X_init, y_init, T=5, short=True, desc="BO"):
    dim = X_init.shape[1]
    X, y = X_init.copy(), y_init.copy()
    
    # FIX: Set bounds on kernel parameters to help avoid ConvergenceWarnings
    kernel = C(1.0, (1e-5, 1e5)) * RBF(length_scale=np.ones(dim), length_scale_bounds=(1e-5, 1e5))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

    for _ in tqdm(range(T), desc=desc, leave=False):
        gp.fit(X, y)
        grid = np.random.rand(96, dim)
        mu, sigma = gp.predict(grid, return_std=True)
        af_vals = AF_callable(mu, sigma, np.min(y))
        x_next = grid[int(np.argmax(af_vals))]
        y_next = kd_objective(unnormalize(x_next), short=short)
        X = np.vstack([X, x_next])
        y = np.append(y, y_next)
    # Return best *accuracy* (1.0 - min_loss)
    return 1.0 - float(np.min(y))

# ============================================================
# Core FunBO Class
# ============================================================

class FunBOGroq:
    def __init__(self, out_dir="funbo_groq_generations", seed=42):
        np.random.seed(seed)
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.results_log = []

    def base_af(self):
        return """
def AF(mu, sigma, f_best):
    import numpy as np
    from scipy.stats import norm
    z = (f_best - mu) / (sigma + 1e-9)
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
"""

    def save_af_file(self, gen, tag, score, code):
        fname = f"gen{gen}_{tag}_score{score:.4f}.py"
        path = os.path.join(self.out_dir, fname)
        with open(path, "w") as f:
            f.write(code)
        print(f"💾 Saved: {path}")

    def run(self, generations=5, inner_T=3):
        # ==================================================
        # UPDATE: Evaluate EI to set a real baseline score
        # ==================================================
        base_code = self.base_af()
        ei_callable = compile_af(base_code)

        dim = 4
        # FIX: Use 4 initial points for a more stable GPR
        n_init = 4
        X_init = np.random.rand(n_init, dim)
        y_init = np.array([kd_objective(unnormalize(x), short=True) for x in X_init])

        print("Evaluating baseline Expected Improvement (EI)...")
        ei_score = run_inner_bo(ei_callable, X_init, y_init, T=inner_T, short=True, desc="Baseline EI")
        print(f"Baseline EI Score: {ei_score:.4f}")

        # Historical bests, persistent across generations
        # FIX: The global best now starts at the EI score
        best_global_code, best_global_score = base_code, ei_score
        best_a_score = 0.0 # LLM-A's own historical best
        # ==================================================

        for g in range(1, generations + 1):
            print(f"\n🚀 Generation {g}/{generations}")

            # Feedback: LLM-A gets its own best, LLM-B gets the global best
            candidates_a, candidates_b = groq_generate_af_batch(
                best_global_code,
                feedback_a=f"Your previous best score: {best_a_score:.4f}",
                feedback_b=f"Global best so far: {best_global_score:.4f}"
            )

            results_a, results_b = [], []

            # Evaluate A’s candidates
            for i, code in enumerate(candidates_a, start=1):
                try:
                    AF = compile_af(code)
                    score_llm = run_inner_bo(AF, X_init, y_init, T=inner_T, short=True, desc=f"LLM-A cand {i}")
                    results_a.append((code, score_llm))
                    self.save_af_file(g, f"A_cand{i}", score_llm, code)
                except Exception as e:
                    print(f"❌ LLM-A Candidate {i} failed: {e}")

            # Evaluate B’s candidates
            for i, code in enumerate(candidates_b, start=1):
                try:
                    AF = compile_af(code)
                    score_llm = run_inner_bo(AF, X_init, y_init, T=inner_T, short=True, desc=f"LLM-B cand {i}")
                    results_b.append((code, score_llm))
                    self.save_af_file(g, f"B_cand{i}", score_llm, code)
                except Exception as e:
                    print(f"❌ LLM-B Candidate {i} failed: {e}")

            # Find bests *for this generation*
            gen_best_a_code, gen_best_a_score = (None, -1.0)
            gen_best_b_code, gen_best_b_score = (None, -1.0)

            if results_a:
                gen_best_a_code, gen_best_a_score = max(results_a, key=lambda x: x[1])
            if results_b:
                gen_best_b_code, gen_best_b_score = max(results_b, key=lambda x: x[1])

            # Update LLM-A's historical best (for its own feedback loop)
            if gen_best_a_score > best_a_score:
                best_a_score = gen_best_a_score

            # Update Global Best (must be better than *historical* global best)
            if gen_best_a_score > best_global_score:
                print(f"🎉 New global best from LLM-A: {gen_best_a_score:.4f}")
                best_global_score = gen_best_a_score
                best_global_code = gen_best_a_code
            
            if gen_best_b_score > best_global_score:
                print(f"🎉 New global best from LLM-B: {gen_best_b_score:.4f}")
                best_global_score = gen_best_b_score
                best_global_code = gen_best_b_code

            print(f"🏆 Generation {g} Summary:")
            # FIX: Added EI score to the summary for reference
            print(f"    Baseline EI Score: {ei_score:.4f} (fixed)")
            print(f"    LLM-A best (this gen): {gen_best_a_score:.4f} (Historical: {best_a_score:.4f})")
            print(f"    LLM-B best (this gen): {gen_best_b_score:.4f}")
            print(f"    Global best (so far): {best_global_score:.4f}")

            # FIX: Updated log to be more descriptive
            self.results_log.append({
                "generation": g,
                "baseline_ei": ei_score,
                "A_best_this_gen": gen_best_a_score,
                "B_best_this_gen": gen_best_b_score,
                "A_historical_best": best_a_score,
                "global_best": best_global_score
            })

            with open(os.path.join(self.out_dir, "funbo_dual_results.json"), "w") as f:
                json.dump(self.results_log, f, indent=2)

        print("\n✅ All generations complete. Final global best AF score:", best_global_score)
        return best_global_code, best_global_score


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--inner_T", type=int, default=3)
    args = parser.parse_args()

    funbo = FunBOGroq(out_dir="funbo_dual_groq_generations")
    best_code, best_score = funbo.run(generations=args.generations, inner_T=args.inner_T)
    print("\nFinal Best Score:", best_score)