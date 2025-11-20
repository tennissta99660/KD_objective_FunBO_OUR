#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script uses two Groq Llama-3.3-70B-Versatile LLMs
to generate candidate acquisition functions (AFs) for Bayesian Optimization.

- LLM-A (Exploration): no examples, fully creative.
- LLM-B (Guided): few-shot examples for structured guidance.
"""

import os
import re
import json
import traceback
from dotenv import load_dotenv
from groq import Groq  # pip install groq

# ============================================================
#  SETUP
# ============================================================

load_dotenv()

API_KEY_A = os.getenv("GROQ_API_KEY_A")
API_KEY_B = os.getenv("GROQ_API_KEY_B")

if not API_KEY_A or not API_KEY_B:
    raise RuntimeError("Please set both GROQ_API_KEY_A and GROQ_API_KEY_B in your .env file.")

MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Clients
client_a = Groq(api_key=API_KEY_A)
client_b = Groq(api_key=API_KEY_B)

# ============================================================
#  BASE AF (Expected Improvement)
# ============================================================

BASE_CODE = """
def AF(mu, sigma, f_best):
    import numpy as np
    from scipy.stats import norm
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return ei
"""

# ============================================================
#  FEW-SHOT EXAMPLES (for LLM-B)
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

def AF_example3(mu, sigma, f_best):
    import numpy as np
    from scipy.stats import norm
    z = (mu - f_best) / (sigma + 1e-9)
    return np.log(1 + sigma) * norm.cdf(z) + 0.2 * sigma * np.sin(z)
"""

# ============================================================
#  HELPERS
# ============================================================

def clean_llm_code(output: str) -> str:
    cleaned = re.sub(r"```(python)?", "", output)
    cleaned = cleaned.replace("```", "").strip()
    return cleaned


def split_candidates(output: str):
    """Split multiple functions separated by blank lines."""
    functions = re.split(r"\n\s*\n", output.strip())
    valid_funcs = [f for f in functions if "def AF" in f]
    return valid_funcs


# ============================================================
#  LLM-A (EXPLORATION)
# ============================================================

def groq_generate_afs_exploration(best_code=BASE_CODE, feedback=""):
    """
    LLM-A: Fully creative generator with no few-shot examples.
    """
    SYSTEM_PROMPT = (
        "You are an expert in Bayesian Optimization and Meta-Learning. "
        "Your task is to design unique, high-performing acquisition functions (AFs). "
        "Each function must be strictly valid Python code — no markdown or comments. "
        "Each must define `def AF(mu, sigma, f_best):` returning a differentiable numpy array."
    )

    user_prompt = f"""
Design 3 unique and novel acquisition functions for Bayesian Optimization.

Each function must:
- Be valid Python code: `def AF(mu, sigma, f_best):`
- Contain **all imports inside the function body only**
- Use only numpy and scipy if needed
- Be numerically stable (handle sigma → 0)
- Be differentiable and efficient
- NOT copy, repeat, or define multiple functions
- NO comments, markdown, explanations, or extra text

Encourage creativity — invent new mathematical forms using mu, sigma, f_best.

Context (for reference only):
{feedback}

Baseline (for context, not for reuse):
{best_code}

Output exactly 3 functions, each separated by one blank line.
"""

    response = client_a.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.75,
        max_tokens=1000,
    )

    raw_output = response.choices[0].message.content.strip()
    return split_candidates(clean_llm_code(raw_output))


# ============================================================
#  LLM-B (GUIDED)
# ============================================================

def groq_generate_afs_guided(best_code=BASE_CODE, feedback=""):
    """
    LLM-B: Generates structured but creative AFs using few-shot examples.
    """
    SYSTEM_PROMPT = (
        "You are an expert in Bayesian Optimization and Knowledge Distillation. "
        "You design efficient acquisition functions balancing exploration and exploitation. "
        "All outputs must be strictly valid Python functions with name `AF(mu, sigma, f_best)`."
    )

    user_prompt =  f"""
Design 3 acquisition functions for Bayesian Optimization of Knowledge Distillation models.

Each must:
- Be valid Python code: `def AF(mu, sigma, f_best):`
- Have **all imports inside the function body only**
- Be numerically stable, differentiable, and efficient
- Use innovative combinations of mu, sigma, f_best
- NOT repeat or restate previous code — exactly one function per candidate
- No comments, no markdown, no explanations

You may take mild inspiration from these examples but must produce distinct functions:
{FEW_SHOT_EXAMPLES}

Context:
{feedback}

Baseline (for context):
{best_code}

Output exactly 3 functions, each separated by a blank line.
"""


    response = client_b.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.45,
        max_tokens=1000,
    )

    raw_output = response.choices[0].message.content.strip()
    return split_candidates(clean_llm_code(raw_output))


# ============================================================
#  PREVIEW BOTH LLMs
# ============================================================

def preview_dual_llms(out_dir="funbo_llm_dual"):
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n🚀 Generating candidates using Groq ({MODEL_NAME})\n")

    # === LLM-A ===
    print("🔹 LLM-A (Exploration) — No examples")
    try:
        codes_a = groq_generate_afs_exploration()
        for i, code in enumerate(codes_a, start=1):
            print(f"\n🧠 LLM-A Candidate {i}:\n{'-'*40}\n{code}\n")
            fname = os.path.join(out_dir, f"LLM_A_candidate_{i}.py")
            with open(fname, "w", encoding="utf-8") as f:
                f.write(f"# LLM-A: Exploration\n# Model: {MODEL_NAME}\n\n{code}")
    except Exception as e:
        print(f"❌ LLM-A failed: {e}")

    # === LLM-B ===
    print("\n🔹 LLM-B (Guided) — With few-shot examples")
    try:
        codes_b = groq_generate_afs_guided()
        for i, code in enumerate(codes_b, start=1):
            print(f"\n🧠 LLM-B Candidate {i}:\n{'-'*40}\n{code}\n")
            fname = os.path.join(out_dir, f"LLM_B_candidate_{i}.py")
            with open(fname, "w", encoding="utf-8") as f:
                f.write(f"# LLM-B: Guided\n# Model: {MODEL_NAME}\n\n{code}")
    except Exception as e:
        print(f"❌ LLM-B failed: {e}")

    print("\n✅ All candidates generated and saved successfully.")


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    preview_dual_llms()
