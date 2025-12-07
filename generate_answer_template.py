#!/usr/bin/env python3
"""
Generate a placeholder answer file that matches the expected auto-grader format.

Replace the placeholder logic inside `build_answers()` with your own agent loop
before submitting so the ``output`` fields contain your real predictions.

Reads the input questions from cse_476_final_project_test_data.json and writes
an answers JSON file where each entry contains a string under the "output" key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import os
import re
import requests
import time


# ---- Provided API settings ----
# I keep these configurable via env vars so the TA can run it easily.
API_KEY = os.getenv("OPENAI_API_KEY", "cse476")
API_BASE = os.getenv("API_BASE", "http://10.4.58.53:41701/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "bens_model")

INPUT_PATH = Path("cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("cse_476_final_project_answers.json")

# ---- Simple cache (in-memory + optional disk) ----
CACHE_PATH = Path(".answer_cache.json")
CACHE: Dict[str, str] = {}

def load_cache() -> None:
    """Load cached answers from disk (if present)."""
    global CACHE
    if CACHE_PATH.exists():
        try:
            CACHE = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            if not isinstance(CACHE, dict):
                CACHE = {}
        except Exception:
            CACHE = {}

def save_cache() -> None:
    """Persist cache to disk so I can resume long runs without losing progress."""
    try:
        CACHE_PATH.write_text(json.dumps(CACHE, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def is_mcq(q: str) -> bool:
    q = q.strip()

    # A./B./C./D. format
    # This checks for classical multiple-choice patterns where each option begins with “A.” “B.” etc.
    dot = all(opt in q for opt in ["A.", "B.", "C.", "D."])

    # Options: (A) (B) format
    # This detects datasets with “Options:” followed by choices labeled with parentheses.
    paren = ("Options" in q) and ("(A)" in q or "(B)" in q or "(C)" in q)

    # Return True if the question uses *any* recognizable MCQ pattern
    return dot or paren


def classify_question(q: str) -> str:
    """
    Better routing across dataset types.
    Returns: mcq | yesno | context_qa | math | open

    This function detects the *type* of question so the system prompt
    can ensure the model outputs answers in the correct format.
    """
    t = q.strip()
    low = t.lower()

    # Yes/No plausibility / entailment-type prompts
    # These patterns often belong to datasets where the answer is strictly Yes or No.
    if low.startswith("is the following sentence plausible"):
        return "yesno"
    if low.startswith("is ") and "facts:" in low:
        return "yesno"
    if low.startswith("does ") and "facts:" in low:
        return "yesno"

    # Context-bounded QA (answer usually stated in the context)
    # Many reasoning datasets embed the answer in a 'Context:' field.
    if "context:" in low or "facts:" in low:
        return "context_qa"

    # Multiple choice (includes “Find a movie similar to … Options: …” etc.)
    # Uses the helper is_mcq() to capture several MCQ formats.
    if is_mcq(t):
        return "mcq"

    # Math-ish word problems (multilingual-friendly signals)
    # These keywords catch numerical/computation-heavy tasks across languages.
    math_signals = [
        "$", "%", "percent", "commission", "calculate", "how many", "total",
        "weigh", "pieces", "times as much", "fewer than", "every week", "average",
        "¿cuánt", "ovejas", "dólar", "多少钱", "多少", "总共", "每块"
    ]
    # If any math keyword appears, route to numeric-answer mode
    if any(sig in low for sig in math_signals):
        return "math"

    # Default fallback: open-ended short-text questions
    return "open"


def build_system_prompt(qtype: str) -> str:
    """
    Prompts must enforce: output is ONLY the final answer string.

    This function defines strict output formats for each question type
    to match autograder requirements.
    """
    if qtype == "mcq":
        # Ensures no explanation leaks—only the final MCQ letter is allowed.
        return "Output ONLY one letter: A, B, C, or D. No other text."

    if qtype == "yesno":
        # Prevents models from outputting long answers—keeps output binary.
        return "Output ONLY 'Yes' or 'No'. No other text."

    if qtype == "context_qa":
        # Forces grounding strictly in provided context, not world knowledge.
        return (
            "Use ONLY the provided Context/Facts in the input. "
            "If the answer is not explicitly stated, output 'N/A'. "
            "Output ONLY the final answer (no explanation)."
        )

    if qtype == "math":
        # Ensures numeric correctness by forbidding reasoning steps.
        return "Output ONLY the final numeric answer (no steps, no explanation)."

    # Default case: short free-text answers permitted, but no reasoning.
    return "Output ONLY the final answer as plain text (one short line). No reasoning."


def call_llm(prompt: str, multiple_choice: bool) -> str:
    """
    Calls ONLY the course-provided OpenAI-style API endpoint.

    Rules I follow here:
    - I do NOT call any other LLM provider.
    - I do only 1 call per question (efficient: << 20 calls/question).
    - I return a short final answer string only.
    """
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    # I change the system prompt depending on whether it looks like MCQ.
    system = (
        "Choose the best option. Reply with ONLY the letter (A, B, C, or D)."
        if multiple_choice
        else "Reply ONLY with the final answer (short). No explanation or reasoning."
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 16 if multiple_choice else 128,

    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
    except Exception:
        # If something breaks, I don't crash the whole run.
        # I use a safe fallback so the answers JSON still validates.
        return "A" if multiple_choice else "N/A"

    # Basic cleanup to avoid weird quoting or long junk.
    text = text.strip().strip('"').strip("'")
    if len(text) > 5000:
        text = text[:5000]

    # If MCQ, extract a single letter.
    if multiple_choice:
        m = re.search(r"\b([A-D])\b", text.upper())
        return m.group(1) if m else "A"

    return text if text else "N/A"

def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data
def build_answers(questions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    My agent loop:
    - 1 API call per question
    - no external tools
    - produces a string in {"output": "..."} format per question
    """
    # Initialize an empty list to store all generated model outputs
    answers: List[Dict[str, str]] = []
    start = time.time()

    # Iterate through each question object while keeping track of question index
    for idx, question in enumerate(questions, start=1):
        # Extract the question text from the JSON dictionary
        # Defaults to an empty string if "input" key is missing
        qtext = question.get("input", "")

        # Heuristic: most dataset MCQs contain "A." "B." "C." "D."
        # This simple rule helps the model distinguish between open-ended and MCQ formats
        multiple_choice = all(opt in qtext for opt in ["A.", "B.", "C.", "D."])

        # Call the LLM with the question text
        # The flag multiple_choice hints to the model how to format its reasoning/answering approach
        prediction = call_llm(qtext, multiple_choice=multiple_choice)

        # Store the model output in the required {"output": "..."} structure
        answers.append({"output": prediction})

        # Progress every 25 questions
        if idx % 25 == 0:
            elapsed = time.time() - start
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (len(questions) - idx) / rate if rate > 0 else float("inf")
            print(f"[{idx}/{len(questions)}] ~{rate:.2f} q/s, ETA ~{remaining/60:.1f} min")

    # Return the full list of predictions to be written into the output JSON
    return answers



def validate_results(
    questions: List[Dict[str, Any]], answers: List[Dict[str, Any]]
) -> None:
    if len(questions) != len(answers):
        raise ValueError(
            f"Mismatched lengths: {len(questions)} questions vs {len(answers)} answers."
        )
    for idx, answer in enumerate(answers):
        if "output" not in answer:
            raise ValueError(f"Missing 'output' field for answer index {idx}.")
        if not isinstance(answer["output"], str):
            raise TypeError(
                f"Answer at index {idx} has non-string output: {type(answer['output'])}"
            )
        if len(answer["output"]) >= 5000:
            raise ValueError(
                f"Answer at index {idx} exceeds 5000 characters "
                f"({len(answer['output'])} chars). Please make sure your answer does not include any intermediate results."
            )


def main() -> None:
    questions = load_questions(INPUT_PATH)
    print(f"Loaded {len(questions)} questions. Starting inference...", flush=True)

    answers = build_answers(questions)

    with OUTPUT_PATH.open("w") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)

    with OUTPUT_PATH.open("r") as fp:
        saved_answers = json.load(fp)
    validate_results(questions, saved_answers)
    print(
        f"Wrote {len(answers)} answers to {OUTPUT_PATH} "
        "and validated format successfully."
    )


if __name__ == "__main__":
    main()

