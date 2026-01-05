# A Survey on Mechanism Design Meets Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Auction Parameters](#auction-parameters)
4. [Reproducing Our Results](#reproducing-our-results)

---

## Overview

This repository reproduces the numerical results in our paper: A Survey on Mechanism Design Meets Large Language Models

---

## Getting Started

Our code supports **Python 3.9 – 3.11**.

```bash
# 1  Create and activate a virtual‑env
python -m venv venv
source venv/bin/activate

# 2  Install dependencies
pip install edsl

# 3  Configure your OpenAI key
cat > .env <<'EOF'
OPENAI_API_KEY= # your openAI key
EOF
```

---

## Auction Parameters

All experiments are launched via one of the three driver scripts below. Use the flags shown to configure auction mechanics.

| Auction type           | Driver script          | Required flags                                                                                           | Optional flags                                                                               |
| ---------------------- | ---------------------- | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Sealed‑bid**         | `main.py`              | `--seal_clock seal`                                                                                      | `--price_order {first,second,third,allpay}`<br>`--private_value {private,affiliated,common}` |


## Reproducing Our Results

The code is based on paper [Learning from Synthetic Labs: Language Models as Auction Participants](https://github.com/KeHang-Zhu/llm-auction).

### 1  Seeds and model settings

| Value regime        | Seeds used      |
| ------------------- | --------------- |
| Private             | **1299 – 1309** |
| Affiliated & Common | **1399 – 1409** |

All experiments use **GPT‑4**, `temperature = 0.5`.

### 2  Cached runs (default)

```python
results = survey.by(model).run(
    remote_inference_description="cache reuse", 
    remote_inference_visibility="public"         
)
```

The snippet above will *first look in the cache*; if a match is found, the result is loaded instantly.

### 3  Forcing a fresh run (optional)

To ignore the cache—for instance, when testing a new prompt—add `fresh=True`:

```python
results = survey.by(model).run(
    remote_inference_description="fresh run",
    remote_inference_visibility="public",
    fresh=True
)
```

### 4  Verifying cache hits

EDSL prints the **Job UUID** and whether it was served from cache. You can also inspect the universal cache via the web UI linked in the EDSL docs.

---


## 🔧 Dependencies
The main third-party package requirement are `openai` and `edsl`.
