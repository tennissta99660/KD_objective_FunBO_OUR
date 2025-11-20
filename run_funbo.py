# run_funbo.py
import json
from funbo_core import FunBOOpenAI
from bo_baseline import run_bo_baseline, expected_improvement

def main():
    # parameters (small for trial)
    generations = 2
    candidates = 3
    inner_T = 3

    print("=== Running FunBO (Groq) ===")
    fb = FunBOOpenAI()
    best_code, best_score = fb.run(generations=generations, candidates=candidates, inner_T=inner_T)
    with open("funbo_best_af.py", "w") as f:
        f.write(best_code)
    print("Saved best AF to funbo_best_af.py with score:", best_score)

    print("\n=== Running BO baseline comparisons (EI/PI/UCB) ===")
    ei_score = run_bo_baseline(expected_improvement, T=inner_T, n_init=3, short=True)
    
    results = {
        "funbo_best_score": best_score,
        "ei_score": ei_score
    }
    print("Results:", results)
    with open("funbo_vs_baseline_results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
