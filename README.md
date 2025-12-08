CSE 476 Final Project — Agent Implementation README

This repository contains my implementation of a reasoning-time agent for the CSE 476 Final Project. The goal of the project is to design an inference-time strategy that can reliably convert diverse question formats into short, autograder-compatible final answers using only the course-provided LLM API.

My approach focuses on robust formatting, question-type classification, lightweight extraction heuristics, and safe LLM prompting with caching.

📁 Repository Structure

Only the essential files needed to reproduce my agent are included:
.

├  ── generate_answer_template.py   # Main agent implementation

  └── README.md                     # Project documentation

The following files are not included intentionally:

cse_476_final_project_test_data.json — provided by instructors, not part of repo

cse_476_final_project_answers.json — generated locally after running the script

.answer_cache.json — auto-created during long runs; not committed

This keeps the repository clean and avoids leaking unnecessary or temporary artifacts.

⚙️ Environment Setup

To recreate my environment from scratch:

python3 -m venv .venv
source .venv/bin/activate
pip install requests

No additional dependencies are needed.

🔑 API Configuration (Important)

This project does not include any API keys and does not require submitting them.

The script uses the following environment variables if available:

OPENAI_API_KEY

API_BASE

MODEL_NAME


The script safely returns to the default values specified in the code if you do not supply these variables.   How to Conduct Inference
 Considering that the test data file is located in the same directory:
 generate_answer_template.py in Python 3
 The result is cse_476_final_project_answers.json.

 comprising a single item for each question, each with:

 { "output" : "<final answer only>" }

 High-Level Overview of the Agent's Operation

 My agent employs a multi-phase inference-time approach:

 1. Classification of Question Types

 Every query falls into one of the following categories:

 McQ

 Yes or no

 context_qa

 math

 open

 This informs the model of the precise output format that is needed.

2. Strict System Prompts

Every question type receives a minimal, format-safe system instruction such as:

MCQ: “Output ONLY one letter: A, B, C, or D.”

Math: “Output ONLY the final numeric answer.”

Yes/No: “Output ONLY 'Yes' or 'No'."

This reduces hallucinations and prevents reasoning from leaking.

3. When feasible, Context-Based Deterministic Extraction

In certain questions, the answer is stated directly in the text.
The agent tries regex-based extraction for patterns like these before contacting the model:

"published by ___"

"constructed in 19XX"

" ___ is the highest reference hospital."

This increases accuracy and prevents needless API requests.

4. LLM Call with Cache and Retry

 A reliable function manages:

 Retry three times with exponential backoff

 global caching to prevent additional calls from being triggered by repeated prompts

 token-limited answers to avoid lengthy outputs

 5. Normalization of Output

 Final responses are cleaned, edited, validated, and limited to permitted tokens.

 6. Questionable Output Correction Pass

 A second "repair" call with more stringent guidelines is made if the model deviates from the desired format.
📌 Reproducibility Notes

Deterministic prompts and consistent system instructions ensure reproducibility.

Global caching dramatically speeds up repeated experiments.

No external LLMs or paid services are used.

The repository does not include any API keys or sensitive files.

🔗 GitHub Link
https://github.com/GoelUmang/Final_project

📝 Final Report

A separate PDF is submitted alongside this README. It describes:

How the agent works

Key functions and code blocks

Link to this GitHub repository

Reproduction instructions
