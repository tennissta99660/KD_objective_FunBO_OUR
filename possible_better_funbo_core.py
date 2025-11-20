import os, json, re, traceback, argparse, time
import numpy as np
from scipy.stats import norm
from openai import OpenAI
from dotenv import load_dotenv
from kd_objective import kd_objective
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from tqdm import tqdm

load_dotenv()

# ==============================
# Setup
# ==============================
API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY2 = os.getenv("OPENAI_API_KEY2")
if not API_KEY:
    raise RuntimeError("Please set your OPENAI_API_KEY in .env")

## MERGED: Using a strong model is key for this task
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")


def get_openai_client(use_backup=False):
    """Returns primary or backup OpenAI client."""
    key = API_KEY2 if use_backup else API_KEY
    return OpenAI(api_key=key)

client = get_openai_client()

# ==============================
# Utility helpers
# ==============================

def clean_llm_code(output: str) -> str:
    cleaned = re.sub(r"```(?:python)?", "", output)
    cleaned = cleaned.replace("```", "").strip()
    return cleaned


def split_candidates(output: str, num_candidates=3):
    """
    Split multiple AF function definitions and standardize names.
    Ensures we extract 'AF1', 'AF2', etc. and rename them to 'AF'.
    """
    parts = re.split(r"(?=def\s+AF\d*\s*\()", output)
    candidates = [p.strip() for p in parts if p.strip()]

    fixed = []
    for code in candidates:
        code_fixed = re.sub(r"def\s+AF\d*\s*\(", "def AF(", code)
        fixed.append(code_fixed)
    
    # Pad if LLM returned fewer than requested
    if fixed and len(fixed) < num_candidates:
         while len(fixed) < num_candidates:
             fixed.append(fixed[-1]) # Duplicate last one

    return fixed[:num_candidates]


def compile_af(code_str):
    """Compiles the AF string into a callable function."""
    safe_globals = {"np": np, "norm": norm}
    try:
        exec(code_str, safe_globals)
    except Exception as e:
        raise RuntimeError(f"Compilation failed: {e}")

    af = safe_globals.get("AF")
    if not callable(af):
        af_candidates = [v for k, v in safe_globals.items() if callable(v) and "AF" in k]
        if af_candidates:
            af = af_candidates[0]
        else:
            raise RuntimeError("Compiled AF does not define AF(mu, sigma, f_best).")
    return af


def expected_improvement(mu, sigma, f_best):
    """Standard Expected Improvement."""
    import numpy as np
    from scipy.stats import norm
    z = (f_best - mu) / (sigma + 1e-9)
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)


def smoothed_kd_objective(params, repeats=3, short=True):
    """
    ## MERGED (from Code 3):
    Runs the objective multiple times to get a "smoothed" (less noisy) value.
    """
    vals = [kd_objective(params, short=short) for _ in range(repeats)]
    return float(np.mean(vals))


def unnormalize(x):
    """Unnormalizes [0, 1] vector to KD hyperparameter space."""
    return {
        "alpha": float(x[0]),
        "beta": float(x[1]) * 0.1,
        "temperature": 2.0 + 18.0 * float(x[2]),
        "learning_rate": 10 ** (-4 + 3.0 * float(x[3])),
        "batch_size": 32,
        "epochs": 3,
    }


def similarity_score(code1: str, code2: str) -> float:
    """
    ## MERGED (from Code 3):
    Simple Jaccard similarity to filter out non-novel AFs.
    """
    s1 = re.sub(r"\s+", " ", code1).strip()
    s2 = re.sub(r"\s+", " ", code2).strip()
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / (len(set1 | set2) + 1e-9)


# ==============================
# OpenAI generation (Merged)
# ==============================

def openai_generate_af_batch(best_code: str, feedback_json: str, candidates: int = 3):
    """
    ## MERGED (from Code 1 & 3):
    Uses the multi-turn chat structure (Code 1) with structured 
    JSON feedback (Code 3).
    """
    system_prompt = (
        "You are a top researcher in Bayesian Optimization and Knowledge Distillation. "
        "Write valid, standalone Python functions that define `def AF(mu, sigma, f_best):`. "
        "Requirements:\n"
        "- Return a numpy array with the same shape as `mu`.\n"
        "- Be numerically stable when sigma -> 0 (avoid division by zero).\n"
        "- Be differentiable w.r.t. mu and sigma (use smooth functions).\n"
        "- Avoid standard EI, PI, UCB formulations or trivially copying them.\n"
        "- Use nonlinear interactions (multiplicative mu*sigma, logs, tanh, sigmoid, ratios, etc.).\n"
        "Do not include explanations, markdown, or extraneous text — only Python code."
    )
    
    few_shot_examples = """
# Example AFs for reference (for structure only, do NOT copy exactly):
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

    user_prompt = f"""
Design {candidates} distinct acquisition functions for Bayesian Optimization.

Instructions:
- Each AF must define `def AF(mu, sigma, f_best):`
- AF1 should be a small, creative mutation of the "current_best_af".
- AF2 and AF{candidates} should be novel, structurally different ideas.
- Include imports (numpy as np, scipy.stats as norm) inside each function.

Current best AF (for AF1 mutation):
{best_code}

Structured feedback from last generation (JSON):
{feedback_json}

Reference examples (DO NOT COPY):
{few_shot_examples}

Output exactly {candidates} Python functions, separated by one blank line. No other text.
"""

    def try_request(client_obj):
        return client_obj.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "I need 3 new acquisition functions."},
                {"role": "assistant", "content": "OK. I will provide 3 Python functions as requested, formatted as code only."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.8,
            max_tokens=1000,
            top_p=1,
        )

    try:
        response = try_request(client)
    except Exception as e1:
        print("⚠️ Primary API key failed, switching to backup...")
        if not API_KEY2:
            raise RuntimeError("Backup OPENAI_API_KEY2 not found in .env file.")
        try:
            response = try_request(get_openai_client(use_backup=True))
        except Exception as e2:
            traceback.print_exc()
            raise RuntimeError(f"Both API keys failed.\nPrimary error: {e1}\nBackup error: {e2}")

    raw_output = response.choices[0].message.content.strip()
    cleaned = clean_llm_code(raw_output)
    return split_candidates(cleaned, num_candidates=candidates)


# ==============================
# Inner BO (Merged)
# ==============================

def run_inner_bo(AF_callable, X_init, y_init, T=10, short=True, desc="BO", repeats_per_eval=3, normalize_af=True):
    """
    ## MERGED (from Code 1 & 3):
    - Logs sigmas for diagnostic feedback (Code 1).
    - Uses larger grid search (Code 1).
    - Uses smoothed objective (Code 3).
    - Normalizes AF output (Code 3).
    """
    dim = X_init.shape[1]
    X, y = X_init.copy(), y_init.copy()
    kernel = C(1.0) * RBF(length_scale=np.ones(dim))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
    
    sigmas_at_chosen_X = []

    for _ in tqdm(range(T), desc=desc, leave=False):
        try:
            gp.fit(X, y)
        except ValueError as e:
            print(f"GP fit failed: {e}. Adding jitter.")
            X = X + np.random.randn(*X.shape) * 1e-5
            continue
            
        ## MERGED (from Code 1): Larger grid for better optimization
        grid = np.random.rand(2000, dim)
        mu, sigma = gp.predict(grid, return_std=True)
        sigma = np.maximum(sigma, 1e-9)

        try:
            af_vals = AF_callable(mu, sigma, np.min(y))
        except Exception as e:
            print(f"⚠️ AF callable error: {e}. Falling back to EI.")
            af_vals = expected_improvement(mu, sigma, np.min(y))

        af_vals = np.array(af_vals, dtype=float)

        if normalize_af:
            ## MERGED (from Code 3): Normalize to [0, 1]
            mn, mx = np.nanmin(af_vals), np.nanmax(af_vals)
            if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) == 0:
                af_vals = expected_improvement(mu, sigma, np.min(y)) # Fallback
            else:
                af_vals = (af_vals - mn) / (mx - mn + 1e-9)

        if not isinstance(af_vals, np.ndarray) or not np.all(np.isfinite(af_vals)):
             print(f"⚠️ AF produced invalid values. Falling back to EI.")
             af_vals = expected_improvement(mu, sigma, np.min(y))

        idx_next = int(np.argmax(af_vals))
        x_next = grid[idx_next]
        sigmas_at_chosen_X.append(sigma[idx_next]) # Log sigma
        
        params = unnormalize(x_next)
        ## MERGED (from Code 3): Use smoothed objective
        y_next = smoothed_kd_objective(params, repeats=repeats_per_eval, short=short)
        
        X = np.vstack([X, x_next])
        y = np.append(y, y_next)

    final_score = 1.0 - float(np.min(y))
    return final_score, np.array(sigmas_at_chosen_X)


# ==============================
# Core FunBO Class (Merged)
# ==============================
class FunBO_Merged:
    def __init__(self, out_dir="funbo_merged_generations", seed=42):
        np.random.seed(seed)
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.results_log = []
        self.dim = 4 # Problem dimensionality

    def base_af(self):
        """Returns standard EI as the baseline."""
        return """
def AF(mu, sigma, f_best):
    import numpy as np
    from scipy.stats import norm
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return ei
"""

    def save_af_file(self, gen, tag, score, code):
        fname = f"gen{gen}_{tag}_score{score:.4f}.py"
        path = os.path.join(self.out_dir, fname)
        with open(path, "w") as f:
            f.write(code)
        print(f"💾 Saved: {path}")

    def _generate_diagnostic_feedback(self, llm_score, ei_score, sigmas):
        """
        ## MERGED (from Code 1):
        Creates diagnostic feedback based on sigma analysis.
        """
        avg_sigma = np.mean(sigmas) if len(sigmas) > 0 else 0.0
        
        if llm_score > ei_score:
            analysis = f"SUCCESS: Outperformed EI (Score: {llm_score:.4f} vs {ei_score:.4f})."
            diagnosis = f"Achieved a good balance (avg_sigma: {avg_sigma:.3f}). Try to mutate this for better efficiency or explore a new high-performing structure."
        else:
            analysis = f"FAILED: Underperformed EI (Score: {llm_score:.4f} vs {ei_score:.4f})."
            if avg_sigma < 0.1: # Threshold is arbitrary, tune it
                diagnosis = f"Likely **too exploitative** (avg_sigma: {avg_sigma:.3f}). It got stuck in a local minimum. Next functions MUST weight `sigma` higher."
            else:
                diagnosis = f"Likely **too exploratory** (avg_sigma: {avg_sigma:.3f}). It ignored good `mu` values. Next functions MUST better balance `sigma` with `(f_best - mu)`."
        return analysis, diagnosis

    def _evaluate_candidate(self, AF_callable, n_evals, inner_T, repeats_per_eval, desc_prefix):
        """
        ## MERGED (from Code 1):
        Runs a full ensemble evaluation for a single AF.
        """
        scores = []
        all_sigmas = []
        
        for i in range(n_evals):
            # Generate NEW initial data for each eval run
            X_init = np.random.rand(1, self.dim)
            y_init = np.array([smoothed_kd_objective(unnormalize(x), repeats=repeats_per_eval, short=True) for x in X_init])
            
            desc = f"{desc_prefix} (Run {i+1}/{n_evals})"
            score, sigmas = run_inner_bo(AF_callable, X_init, y_init, T=inner_T, 
                                         repeats_per_eval=repeats_per_eval, desc=desc, 
                                         normalize_af=True) # Normalize LLM AFs
            scores.append(score)
            all_sigmas.extend(sigmas)
            
        return np.mean(scores), np.array(all_sigmas)
    
    def _evaluate_hybrid(self, AF_callable, n_evals, inner_T, repeats_per_eval, desc_prefix):
        """
        ## MERGED (from Code 3):
        Runs ensemble eval for the HYBRID AF (LLM + EI).
        """
        
        def AF_hybrid_wrapper(mu, sigma, f_best):
            """This blends the LLM AF and EI."""
            try:
                vals = AF_callable(mu, sigma, f_best)
                vals = np.array(vals, dtype=float)
                # normalize LLM part
                mn, mx = np.nanmin(vals), np.nanmax(vals)
                if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) == 0:
                     vals_n = 0.5 # Neutral weight if invalid
                else:
                    vals_n = (vals - mn) / (mx - mn + 1e-9)
            except Exception:
                vals_n = 0.5 # Neutral weight if LLM AF fails
                
            ei = expected_improvement(mu, sigma, f_best)
            ei_n = (ei - np.min(ei)) / (np.max(ei) - np.min(ei) + 1e-9)
            
            # 70% LLM idea, 30% EI stability
            return 0.7 * vals_n + 0.3 * ei_n

        scores = []
        for i in range(n_evals):
            X_init = np.random.rand(1, self.dim)
            y_init = np.array([smoothed_kd_objective(unnormalize(x), repeats=repeats_per_eval, short=True) for x in X_init])
            
            desc = f"{desc_prefix} (Run {i+1}/{n_evals})"
            # Note: normalize_af=False because the wrapper handles it.
            score, _ = run_inner_bo(AF_hybrid_wrapper, X_init, y_init, T=inner_T, 
                                    repeats_per_eval=repeats_per_eval, desc=desc, 
                                    normalize_af=False) 
            scores.append(score)
            
        return np.mean(scores)


    def run(self, generations=5, candidates=3, inner_T=10, n_evals=5, repeats_per_eval=3, similarity_threshold=0.7):
        
        base_code = self.base_af()
        best_code, best_hybrid_score = base_code, 0.0
        
        feedback_obj = {
            "generation": 0,
            "previous_best_score": 0.0,
            "ei_baseline_score": 0.0,
            "analysis": "Starting first generation.",
            "diagnosis": "No failures yet. Try to outperform standard Expected Improvement.",
            "failure_modes": []
        }

        # First, get a baseline score for EI
        print("Evaluating baseline Expected Improvement (EI)...")
        ei_callable = compile_af(base_code)
        avg_ei_score, _ = self._evaluate_candidate(ei_callable, n_evals, inner_T, repeats_per_eval, "Baseline EI")
        print(f"Baseline EI Avg Score: {avg_ei_score:.4f}")
        
        best_hybrid_score = avg_ei_score
        feedback_obj["ei_baseline_score"] = avg_ei_score

        for g in range(1, generations + 1):
            print(f"\n🚀 Generation {g}/{generations}")

            feedback_obj["generation"] = g
            feedback_json = json.dumps(feedback_obj, indent=2)

            try:
                candidate_codes = openai_generate_af_batch(best_code, feedback_json, candidates=candidates)
            except Exception as e:
                print(f"❌ LLM generation failed: {e}. Retrying with base code.")
                time.sleep(5)
                candidate_codes = openai_generate_af_batch(self.base_af(), feedback_json, candidates=candidates)

            # Filter out candidates that are too similar
            filtered_codes = []
            for code in candidate_codes:
                sim = similarity_score(code, best_code)
                if sim < similarity_threshold or best_code == base_code: # Always allow first gen
                    filtered_codes.append(code)
                else:
                    print(f"— Rejected candidate due to high similarity (sim={sim:.2f})")
            
            if not filtered_codes:
                print("⚠️ All candidates too similar; re-using originals.")
                filtered_codes = candidate_codes

            results = []
            feedback_obj["failure_modes"] = [] # Reset for this gen

            for i, code in enumerate(filtered_codes, start=1):
                try:
                    AF_llm = compile_af(code)
                    
                    # Evaluate LLM AF standalone
                    score_llm, llm_sigmas = self._evaluate_candidate(AF_llm, n_evals, inner_T, repeats_per_eval, f"LLM C{i}")
                    
                    # Evaluate Hybrid AF (LLM + EI)
                    score_hybrid = self._evaluate_hybrid(AF_llm, n_evals, inner_T, repeats_per_eval, f"Hybrid C{i}")

                    print(f"  Candidate {i}: LLM Score={score_llm:.4f}, Hybrid Score={score_hybrid:.4f}")
                    results.append((code, score_llm, score_hybrid, llm_sigmas))

                except Exception as e:
                    print(f"❌ Candidate {i} failed evaluation: {e}")
                    traceback.print_exc()
                    feedback_obj["failure_modes"].append(f"Candidate {i} failed: {e}")

            if not results:
                print("⚠️ No successful candidates this generation.")
                feedback_obj["analysis"] = "All candidates failed to compile or run."
                feedback_obj["diagnosis"] = "Check LLM output or compilation logic."
                continue

            # Sort by HYBRID score first (Code 3), then standalone LLM score (Code 1)
            results.sort(key=lambda x: (x[2], x[1]), reverse=True)
            
            gen_best_code, gen_best_llm_score, gen_best_hybrid, gen_best_sigmas = results[0]

            if gen_best_hybrid > best_hybrid_score:
                print(f"📈 New Best Found! Hybrid Score: {gen_best_hybrid:.4f} (EI baseline: {avg_ei_score:.4f})")
                best_code = gen_best_code
                best_hybrid_score = gen_best_hybrid
                self.save_af_file(g, "bestAF_hybrid", gen_best_hybrid, best_code)
                
                # Update feedback for next gen
                analysis, diagnosis = self._generate_diagnostic_feedback(gen_best_llm_score, avg_ei_score, gen_best_sigmas)
                feedback_obj["analysis"] = analysis
                feedback_obj["diagnosis"] = diagnosis
                feedback_obj["previous_best_score"] = float(best_hybrid_score)
            else:
                 print(f"📉 No improvement. Best score remains {best_hybrid_score:.4f}.")
                 feedback_obj["analysis"] = "No improvement over previous best."
                 # Give feedback on the best of *this* gen, even if it wasn't an overall best
                 analysis, diagnosis = self._generate_diagnostic_feedback(gen_best_llm_score, avg_ei_score, gen_best_sigmas)
                 feedback_obj["diagnosis"] = f"(No new best) {diagnosis}"


            # Log results for this generation
            self.results_log.append({
                "generation": g,
                "best_hybrid_score": float(best_hybrid_score),
                "ei_score": float(avg_ei_score),
                "scores": [{"llm": float(r[1]), "hybrid": float(r[2])} for r in results],
            })
            with open(os.path.join(self.out_dir, "funbo_merged_results.json"), "w") as f:
                json.dump(self.results_log, f, indent=2)

        print("\n✅ All generations complete. Final best hybrid score:", best_hybrid_score)
        return best_code, best_hybrid_score

# ==============================
# Entry Point
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merged FunBO - finds AFs for KD")
    parser.add_argument("--generations", type=int, default=5, help="Number of FunBO generations")
    parser.add_argument("--candidates", type=int, default=3, help="Number of AFs to generate per generation")
    parser.add_argument("--inner_T", type=int, default=10, help="Inner BO loop iterations per evaluation")
    parser.add_argument("--n_evals", type=int, default=5, help="Number of ensemble runs for robust scoring")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats for smoothed_kd_objective")
    parser.add_argument("--sim_thresh", type=float, default=0.7, help="Similarity threshold to filter new AFs")
    args = parser.parse_args()

    funbo = FunBO_Merged(out_dir="funbo_merged_generations")
    best_code, best_score = funbo.run(
        generations=args.generations, 
        candidates=args.candidates, 
        inner_T=args.inner_T, 
        n_evals=args.n_evals,
        repeats_per_eval=args.repeats,
        similarity_threshold=args.sim_thresh
    )
    
    print("\nFinal Best Score (Hybrid):", best_score)
    print("\nFinal Best Code:")
    print(best_code)