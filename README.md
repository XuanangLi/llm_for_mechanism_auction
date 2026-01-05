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
| Private             | **1401**        |
| Affiliated & Common | **1401**        |

Our experiments use **GPT‑4o-mini**, `temperature = 0.5`; **GPT‑5.1**, `temperature = 0.5`; **Gemini-2.5-flash**, `temperature = 0.5`.

### 2  Drawing bid-value plot

```
python first_bid_value.py # second_bid_value.py | third_bid_value.py | all_pay_bid_value.py
```

### 3 Results

After running `main.py`, all turns results will saved to file path: llm-auction-main/{model_name}_{results}.
All plots will be included in the file: llm-auction-main/plots

---


## 🔧 Dependencies
The main third-party package requirement are `openai`, `gemini` and `edsl`.
