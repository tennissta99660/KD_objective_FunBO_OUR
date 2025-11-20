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

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def get_openai_client(use_backup=False):
    """Returns primary or backup OpenAI client."""
    key = API_KEY2 if use_backup else API_KEY
    return OpenAI(api_key=key)

client = get_openai_client()

# ==============================
# LLM Generation
# ==============================
def openai_generate_af_batch(best_code: str, feedback: str = ""):
    """
    Generate 3 acquisition functions using OpenAI API:
    - AF1: Mutated version of best AF so far.
    - AF2, AF3: Unique and creative new AFs.
    """
    system_prompt = (
        "You are an expert in Bayesian Optimization and Knowledge Distillation. "
        "Write valid Python defining def AF(mu, sigma, f_best): "
        "The function must balance exploration and exploitation for Knowledge Distillation hyperparameter tuning. "
        "It must be efficient, differentiable, and structurally distinct from standard formulas (EI, PI, UCB). "
        "Do not explain, just return code."
    )

    few_shot_examples = """
# Example AFs for reference (for structure only, do NOT copy):
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
Design 3 distinct acquisition functions for Bayesian Optimization in Knowledge Distillation.

Each must be:
- Valid Python defining `def AF(mu, sigma, f_best):`
- Use numpy and scipy.stats if needed
- Return a numpy array of acquisition values
- Avoid markdown, explanations, or triple backticks
- Be structurally distinct from each other
- NOT a standard formula (EI, PI, UCB)
- Mutate the provided AF slightly for AF1 (modify nonlinearities or weights)
- Invent completely new ones for AF2 and AF3

Current best AF to mutate (for AF1):
{best_code}

Feedback from previous results:
{feedback}

Now output exactly 3 Python functions (AF1, AF2, AF3), separated by one blank line.
{few_shot_examples}
"""

    def try_request(client_obj):
        return client_obj.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.77,
            max_completion_tokens=800,
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
    return split_candidates(clean_llm_code(raw_output))

# ==============================
# Utility Functions
# ==============================
def clean_llm_code(output: str) -> str:
    cleaned = re.sub(r"```(python)?", "", output)
    cleaned = cleaned.replace("```", "").strip()
    return cleaned

def split_candidates(output: str):
    """
    Split multiple AF function definitions and standardize names.
    Ensures we extract 'AF1', 'AF2', etc. and rename them to 'AF'.
    """
    # Split by any function definition like def AF or AF1, AF2...
    parts = re.split(r"(?=def\s+AF\d*\s*\()", output)
    candidates = [p.strip() for p in parts if p.strip()]

    # Rename AF1/AF2/etc to AF so they can compile correctly
    fixed = []
    for code in candidates:
        code_fixed = re.sub(r"def\s+AF\d*\s*\(", "def AF(", code)
        fixed.append(code_fixed)
    return fixed[:3]  # max 3 candidates


def compile_af(code_str):
    safe_globals = {}
    try:
        exec(code_str, safe_globals)
    except Exception as e:
        raise RuntimeError(f"Compilation failed: {e}")

    af = safe_globals.get("AF")
    if not callable(af):
        # Try to find any function with AF in its name
        af_candidates = [v for k, v in safe_globals.items() if callable(v) and "AF" in k]
        if af_candidates:
            af = af_candidates[0]
        else:
            raise RuntimeError("Compiled AF does not define AF(mu, sigma, f_best).")

    return af


def expected_improvement(mu, sigma, f_best):
    z = (f_best - mu) / (sigma + 1e-9)
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

def run_inner_bo(AF_callable, X_init, y_init, T=5, short=True, desc="BO"):
    dim = X_init.shape[1]
    X, y = X_init.copy(), y_init.copy()
    kernel = C(1.0) * RBF(length_scale=np.ones(dim))
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
    return 1.0 - float(np.min(y))

def unnormalize(x):
    return {
        "alpha": float(x[0]),
        "beta": float(x[1]) * 0.1,
        "temperature": 2.0 + 18.0 * float(x[2]),
        "learning_rate": 10 ** (-4 + 3.0 * float(x[3])),
        "batch_size": 32,
        "epochs": 3,
    }

# ==============================
# Core FunBO Class
# ==============================
class FunBOOpenAI:
    def __init__(self, out_dir="funbo_generations", seed=42):
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

    def run(self, generations=5, candidates=3, inner_T=3):
        base_code = self.base_af()
        best_code, best_score = base_code, 0.0
        feedback = "No prior results available."

        dim = 4
        X_init = np.random.rand(1, dim)
        y_init = np.array([kd_objective(unnormalize(x), short=True) for x in X_init])

        for g in range(1, generations + 1):
            print(f"\n🚀 Generation {g}/{generations}")
            candidate_codes = openai_generate_af_batch(base_code, feedback)
            results = []

            for i, code in enumerate(candidate_codes, start=1):
                try:
                    AF = compile_af(code)
                    score_llm = run_inner_bo(AF, X_init, y_init, T=inner_T, short=True, desc=f"LLM cand {i}")
                    score_ei = run_inner_bo(expected_improvement, X_init, y_init, T=inner_T, short=True, desc=f"EI vs cand {i}")

                    self.save_af_file(g, f"candidate{i}_llm", score_llm, code)
                    print(f"   Candidate {i}: LLM Score={score_llm:.4f}, EI Score={score_ei:.4f}")
                    results.append((code, score_llm, score_ei))
                except Exception as e:
                    print(f"❌ Candidate {i} failed: {e}")

            if not results:
                print("⚠️ No successful candidates, stopping early.")
                break

            results.sort(key=lambda x: x[1], reverse=True)
            best_code, best_score, best_ei = results[0]
            self.save_af_file(g, "bestAF", best_score, best_code)

            feedback = f"""
Previous best AF scored {best_score:.4f}.
EI baseline scored {best_ei:.4f}.
Next, improve exploration–exploitation balance to outperform EI while remaining stable and diverse.
"""

            self.results_log.append({
                "generation": g,
                "best_score": best_score,
                "ei_score": best_ei,
                "scores": [{"llm": float(r[1]), "ei": float(r[2])} for r in results],
            })
            with open(os.path.join(self.out_dir, "funbo_vs_ei_results.json"), "w") as f:
                json.dump(self.results_log, f, indent=2)

            base_code = best_code

        print("\n✅ All generations complete. Final best AF score:", best_score)
        return best_code, best_score

# ==============================
# Entry Point
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--inner_T", type=int, default=3)
    args = parser.parse_args()

    funbo = FunBOOpenAI(out_dir="funbo_generations")
    best_code, best_score = funbo.run(generations=args.generations, candidates=args.candidates, inner_T=args.inner_T)
    print("\nFinal Best Score:", best_score)
