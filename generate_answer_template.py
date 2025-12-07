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

# ---- Provided API settings ----
# I keep these configurable via env vars so the TA can run it easily.
API_KEY = os.getenv("OPENAI_API_KEY", "cse476")
API_BASE = os.getenv("API_BASE", "http://10.4.58.53:41701/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "bens_model")

INPUT_PATH = Path("cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("cse_476_final_project_answers.json")

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
        else "Reply ONLY with the final answer (short). No explanation."
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
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

