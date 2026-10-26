# 🛡️ Sentinel: Adaptive Time-Series Anomaly Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Hardware: RTX 5050](https://img.shields.io/badge/Hardware-NVIDIA_RTX_5050_(CUDA_13.2)-green.svg)](https://www.nvidia.com/)
[![Track: Utkrishta 2026](https://img.shields.io/badge/Hackathon-Utkrishta_2026_Challenge_2-purple.svg)](https://github.com/SatyaBoli/NEWW)

> **Utkrishta 2026 AI/ML Hackathon — Challenge 2 (Sentinel)**  
> **Theme:** Adaptive Streaming Time-Series Anomaly Detection  
> **Repository:** [https://github.com/SatyaBoli/NEWW](https://github.com/SatyaBoli/NEWW)  
> **Team:** Satya Boli & Team Sentinel  

---

## 📌 Executive Summary

**Sentinel** is an enterprise-grade, lightweight, causal, adaptive anomaly detection system designed to ingest streaming time-series metrics, identify anomalies in real time without lookahead bias, protect its baseline from contamination during prolonged incidents, and generate deterministic, machine-verifiable evidence for every flagged event.

Sentinel features a **Causal Temporal Convolutional Autoencoder (TCN-AE)** fine-tuned on the local **NVIDIA GeForce RTX 5050 Laptop GPU (CUDA 13.2, Blackwell CC 12.0)**, combined with statistical baselines (Rolling Median/MAD, Streaming EWMA), ADWIN drift detection, and a hysteresis event state machine.

---

## 🏗️ System Architecture

```
                    Streaming Ingestion (Point t)
                               │
                               ▼
               [Schema & Timestamp Validator]
                               │
                               ▼
          [Causal Feature Buffer (Past-Only Lags & Stats)]
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
      [B0: Rolling MAD]  [B1: EWMA]    [B3: Causal TCN-AE (RTX 5050)]
            │                  │                  │
            └──────────────────┼──────────────────┘
                               ▼
                   [Calibrated Score Fusion]
                               │
                               ▼
                   [Likelihood Transform (math.erfc)]
                               │
                               ▼
                   [Hysteresis State Machine (Enter: 3.2 / Exit: 1.8)]
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
   [Guarded Update Controller]        [Evidence Explainer]
              │                                 │
              ▼                                 ▼
   [ADWIN Drift Adaptation]           [Organizer Submission CSV/JSON]
```

---

## ⚡ Key Highlights & Innovation

1. **Strict Causality (Zero Future Leakage):**
   - Every score and feature at time $t$ depends strictly on observations prior to $t$.
   - The state machine classifies point $x_t$ *before* the baseline update is considered.
2. **Guarded Baseline Update (Anti-Contamination):**
   - Anomalous observations are quarantined from updating rolling reference buffers, preventing true prolonged anomalies from warping normal baselines.
3. **ADWIN Drift Adaptation:**
   - Detects genuine statistical regime shifts using Hoeffding-type bounds, triggering a controlled re-baselining without manual window guessing.
4. **Hysteresis Dual Thresholding & Cooldown Merging:**
   - Eliminates alert flapping through separate enter ($\tau_{\text{enter}} = 3.2$) and exit ($\tau_{\text{exit}} = 1.8$) thresholds.
   - Merges fragmented alarms into unified incident windows via configurable cooldown steps.
5. **Hardware-Accelerated Deep Learning:**
   - Fully optimized for NVIDIA RTX 5050 Laptop GPU with PyTorch CUDA 13.2 native Blackwell kernels.
   - Sub-millisecond inference latency (< 0.3 ms/point) and > 3,600 points/second streaming throughput.
6. **Deterministic Explainability:**
   - Generates machine-verifiable evidence records detailing observed value, expected baseline, residual delta, relative percent change, and contributing detectors.

---

## 📊 Benchmark Results (Utkrishta 2026 Evaluation Protocol)

Evaluated across representative subsets of the **Numenta Anomaly Benchmark (NAB)** using organizer-specified **Event-Based F1 Score** (range-based overlap matching without point-adjustment bias):

| Model / Configuration | Architecture | Event-F1 | Precision | Recall | Median Delay | Throughput |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **E1 (B0 Baseline)** | Rolling Median + MAD | 0.684 | 0.667 | 0.702 | 180s | 12,400 pts/s |
| **E2 (B1 Baseline)** | Streaming EWMA ($\alpha=0.15$) | 0.642 | 0.615 | 0.671 | 120s | 14,800 pts/s |
| **E3 (B2 Candidate)** | Windowed Isolation Forest | 0.667 | 0.650 | 0.685 | 240s | 1,850 pts/s |
| **E4 (B3 Deep Neural)** | Causal TCN-AE (RTX 5050) | 0.741 | 0.733 | 0.750 | 120s | 4,200 pts/s |
| **E5 (Sentinel Ensemble)**| **Hybrid Ensemble + ADWIN** | **0.783** | **0.769** | **0.800** | **90s** | **3,650 pts/s** |

---

## 🚀 Quickstart & Reproducibility

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/SatyaBoli/NEWW.git
cd NEWW

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Train Causal TCN-AE on Local RTX 5050 GPU
```bash
python scripts/train_gpu.py --epochs 10 --batch-size 128 --device cuda
```

### 3. Run Benchmark Bake-Off
```bash
python scripts/run_experiments.py
```

### 4. Run Streaming Pipeline & Generate Submission File
```bash
python scripts/run_sentinel.py --config configs/candidate.yaml --output-csv outputs/predictions.csv
```

### 5. Launch Interactive Replay Dashboard
```bash
python -m streamlit run app/dashboard.py
# Or on Windows simply run:
.\run_dashboard.bat
```

### 6. View Official 10-Slide Presentation Deck
Open `docs/presentation_deck.html` in any web browser or view `docs/presentation_deck.md`.

---

## 📁 Repository Structure

```
NEWW/
├── app/
│   └── dashboard.py               # Streamlit interactive replay dashboard
├── configs/
│   ├── baseline.yaml              # Fast statistical baseline config
│   └── candidate.yaml             # Hybrid Deep Ensemble config
├── docs/
│   ├── presentation_deck.md       # Official Utkrishta 10-slide deck content
│   └── presentation_deck.html     # Interactive browser-based presentation deck
├── models/
│   └── sentinel_tcn_best.pt       # Trained Causal TCN-AE weights (RTX 5050)
├── outputs/
│   ├── experiment_log.csv         # Multi-model evaluation benchmark log
│   └── predictions.csv            # Organizer submission formatted predictions
├── scripts/
│   ├── generate_slides.py         # Presentation deck generator
│   ├── run_experiments.py         # Multi-model bake-off runner
│   ├── run_sentinel.py            # Streaming pipeline CLI
│   └── train_gpu.py               # RTX 5050 GPU training launcher
├── src/sentinel/
│   ├── adaptation/                # ADWIN drift & guarded update controller
│   ├── data/                      # Streaming NAB data loader & schema audit
│   ├── detectors/                 # MAD, EWMA, iForest, TCN-AE & Ensemble
│   ├── evaluation/                # Official event-based F1 & delay evaluator
│   ├── events/                    # Hysteresis event state machine
│   ├── explanations/              # Deterministic evidence generator
│   ├── features/                  # Causal past-only lag/window extractor
│   ├── output/                    # Organizer submission adapter
│   ├── pipeline.py                # Single causal streaming entry point
│   ├── train.py                   # PyTorch CUDA training engine
│   └── contracts.py               # Data shapes & protocol definitions
├── tests/                         # Unit tests (state machine, causality, metrics)
├── pyproject.toml                 # Package configuration
└── README.md                      # Complete system documentation
```

---

## 👥 Authors & Team
- **Satya Boli** (Lead Architect & AI/ML Engineer) — [GitHub Profile](https://github.com/SatyaBoli)
- **Utkrishta 2026 AI/ML Hackathon** — Challenge 2 (Sentinel)
