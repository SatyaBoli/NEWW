# 🛡️ Sentinel: Adaptive Time-Series Anomaly Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Hardware: RTX 5050](https://img.shields.io/badge/Hardware-NVIDIA_RTX_5050_(CUDA_13.2)-green.svg)](https://www.nvidia.com/)
[![Pretrained Models: Included](https://img.shields.io/badge/Pretrained_Models-Included_(35KB_in_Repo)-success.svg)](models/)
[![Track: Utkrishta 2026](https://img.shields.io/badge/Hackathon-Utkrishta_2026_Challenge_2-purple.svg)](https://github.com/SatyaBoli/NEWW)

> **Utkrishta 2026 AI/ML Hackathon — Challenge 2 (Sentinel)**  
> **Theme:** Adaptive Streaming Time-Series Anomaly Detection  
> **Repository:** [https://github.com/SatyaBoli/NEWW](https://github.com/SatyaBoli/NEWW)  
> **Team:** Satya Boli & Team Sentinel  

---

## 📌 Executive Summary & Out-of-the-Box Setup

**Sentinel** is an enterprise-grade, lightweight, causal, adaptive anomaly detection system designed to ingest streaming time-series metrics, identify anomalies in real time without lookahead bias, protect its baseline from contamination during prolonged incidents, and generate deterministic, machine-verifiable evidence for every flagged event.

> [!TIP]
> **Zero-Download / Zero-Retrain Guarantee**: Both pre-trained neural network checkpoints are **committed directly into this git repository** (`models/sentinel_specialist_best.pt` and `models/sentinel_tcn_best.pt`, ~35 KB each). Anyone cloning this repository can **immediately run streaming inference and launch the dashboard without having to train models or download weights from external cloud buckets**.

---

## ⚔️ Dual-Model Architecture: Champion vs. Challenger

Sentinel implements a dual-model framework to demonstrate the empirical benefits of domain-specific inductive bias:

1. **Model A (Global Generalist)**: Trained across all Numenta Anomaly Benchmark (NAB) domains, including social media (Twitter) and ad exchange telemetry.
2. **Model B (Infrastructure Specialist - Recommended)**: Trained selectively on cloud infrastructure and server telemetry (`realAWSCloudwatch`, `realKnownCause`), learning the tight diurnal operational manifold of real production systems.

### 🏆 Empirical Bake-Off Scorecard (23,395 Streaming Points)

Evaluated under strict causal protocol (zero future lookahead, strictly $t-1 \dots t-W$ feature extraction):

| Model Architecture | Checkpoint | Event-F1 | Precision | Recall | False Positives | Throughput | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (Global TCN-AE)** | `models/sentinel_tcn_best.pt` | `0.0870` | `6.67%` | `12.5%` | 14 | 860 pts/s | 1.163 ms/pt |
| **Model B (Infra Specialist TCN)** | `models/sentinel_specialist_best.pt` | **`0.1290`** | **`8.70%`** | **`25.0%`** | 21 | **1,218 pts/s** | **0.821 ms/pt** |
| **Hybrid Ensemble (with Model A)** | `models/sentinel_tcn_best.pt` | `0.2857` | `33.33%` | `25.0%` | **4** | 1,106 pts/s | 0.904 ms/pt |
| **Hybrid Ensemble (with Model B)** | `models/sentinel_specialist_best.pt` | **`0.2857`** | **`33.33%`** | **`25.0%`** | **4** | **1,144 pts/s** | **0.874 ms/pt** |

**Key Takeaways**:
- **+48.3% F1 & 2x Higher Recall**: Standalone Model B achieves `0.1290` Event-F1 and `25.0%` Recall (vs. `0.0870` and `12.5%` for Model A).
- **Sub-Millisecond Speed**: Model B achieves `0.821 ms` inference latency on the NVIDIA RTX 5050 Laptop GPU (> 1,200 points/sec).
- **93.4% False Alarm Reduction**: In ensemble mode, both models suppress false positives from 61 down to **only 4** across 23,395 observations.

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
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
       [B0: Rolling MAD] [B1: EWMA]   [B3: Causal TCN-AE (RTX 5050)]
             │                 │                 │
             └─────────────────┼─────────────────┘
                               ▼
                    [Calibrated Score Fusion]
                               │
                               ▼
                    [Likelihood Transform (math.erfc)]
                               │
                               ▼
            [Hysteresis State Machine (Enter: 3.2 / Exit: 1.8)]
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
    [Guarded Update Controller]        [Evidence Explainer]
               │                               │
               ▼                               ▼
    [ADWIN Drift Adaptation]           [Organizer Submission CSV/JSON]
```

---

## ⚡ Core Engineering & Scientific Pillars

1. **Strict Causality (Zero Future Leakage)**:
   - Window statistics and autoencoder representations depend strictly on observations prior to $t$.
   - The state machine classifies point $x_t$ *before* any baseline buffer updates occur.
2. **Guarded Baseline Adaptation (Anti-Contamination)**:
   - Anomalous observations are quarantined from updating rolling reference buffers, preventing true prolonged anomalies from warping normal baselines.
3. **ADWIN Drift Detection**:
   - Detects genuine statistical regime shifts using Hoeffding-type bounds, triggering a controlled re-baselining without manual window guessing.
4. **Hysteresis Dual Thresholding & Cooldown Merging**:
   - Eliminates alert flapping through separate enter ($\tau_{\text{enter}} = 3.2$) and exit ($\tau_{\text{exit}} = 1.8$) thresholds.
   - Merges fragmented alarms into unified incident windows via configurable cooldown steps.
5. **Native GPU Acceleration**:
   - Optimized for NVIDIA GeForce RTX 5050 Laptop GPU (CUDA 13.2, Blackwell sm_120) with sub-millisecond inference latency (< 0.85 ms/point).
6. **Deterministic Evidence Trails**:
   - Generates machine-verifiable explanation records detailing observed value, expected baseline, residual delta, relative percent change, and contributing detectors.

---

## 🚀 Sentinel 2.0: Advanced Production Pathway

Live interactive demonstrations of Next-Gen features are included on branch `feature/dual-model-specialist`:

- **Conformalized Anomaly Detection (CAD)**: Finite-sample validity guarantee: $P(\text{False Alarm} \mid \text{Normal}) \le \alpha$.
- **Human-in-the-Loop Feedback Bank**: Causal cosine similarity matching for zero-retrain suppression of known operational false alarms (e.g., approved maintenance windows).
- **Automated Causal Periodicity Discovery**: Infers dominant seasonal cycles using autocorrelation on warm-up data without manual specification.
- **Edge Deployment Artifacts**: Exported TorchScript JIT and PyTorch 2.0 AOT compiled models ready for edge IoT gateways and Triton Inference Server.

---

## 📊 Full Multi-Baseline Benchmark Comparison

Evaluated across representative multi-domain benchmark streams (23,395 points) from the **Numenta Anomaly Benchmark (NAB)** under strict causal evaluation (range-based Event-F1 without point adjustment bias):

| Model / Configuration | Architecture | Event-F1 | Precision | Recall | False Positives | Throughput | Inference Latency |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **E1 (B0 Baseline)** | Rolling Median + MAD | 0.1333 | 8.11% | 37.50% | 34 | 3,397 pts/s | 0.294 ms/pt |
| **E2 (B1 Baseline)** | Streaming EWMA ($\alpha=0.15$) | 0.0833 | 4.69% | 37.50% | 61 | 27,815 pts/s | 0.036 ms/pt |
| **E3 (B2 Candidate)** | Windowed Isolation Forest | 0.0000 | 0.00% | 0.00% | 2 | 132 pts/s | 7.573 ms/pt |
| **E4 (B3 Deep Neural)** | Causal TCN-AE (RTX 5050 GPU) | 0.2105 | 18.18% | 25.00% | 9 | 619 pts/s | 1.615 ms/pt |
| **E5 (Sentinel Ensemble)**| **Hybrid Ensemble + Guarded ADWIN** | **0.3750** | **37.50%** | **37.50%** | **5** | **917 pts/s** | **1.091 ms/pt** |

> **False Alarm Suppression**: Sentinel's Hybrid Ensemble slashed false alarms from 61 down to **only 5** (**91.8% reduction** in false positives), while boosting Event-F1 to **0.3750** with balanced 37.5% precision and 37.5% recall. On single-series evaluation (`ec2_cpu`), false alarms were completely suppressed to **0 FP** with **100% precision**.

---

## 💻 Quickstart (No Training Required!)

### 1. Clone & Install
```bash
git clone -b feature/dual-model-specialist https://github.com/SatyaBoli/NEWW.git
cd NEWW

pip install -r requirements.txt
pip install -e .
```

### 2. Launch Interactive Replay Dashboard
```bash
python -m streamlit run app/dashboard.py
# Or on Windows:
.\run_dashboard.bat
```
Visit `http://localhost:8501` to view:
- **Tab 1: Live Streaming Replay**: Interactive streaming time-series visualization with dynamic threshold and evidence cards.
- **Tab 2: Model Benchmark Scorecard**: Side-by-side Champion vs. Challenger scorecard and multi-baseline metrics.
- **Tab 3: Architecture Guarantees**: Formal proof notes for causality and anti-contamination.
- **Tab 4: Sentinel 2.0 Pathway**: Live interactive demos of Conformal Risk Control, Feedback Bank, and Periodicity Discovery.

### 3. Run Benchmark Bake-Off
```bash
# Compare Model A vs. Model B:
python scripts/compare_models.py

# Run all 5 architectural baselines:
python scripts/run_experiments.py
```

### 4. (Optional) Re-Train Models on GPU
Both checkpoints are already included, but you can retrain anytime:
```bash
# Train Model B (Infrastructure Specialist):
python scripts/train_specialist_gpu.py --epochs 15 --batch-size 256 --device cuda

# Train Model A (Global Generalist):
python scripts/train_gpu.py --epochs 10 --batch-size 128 --device cuda
```

### 5. Run Streaming Pipeline & Generate Predictions
```bash
python scripts/run_sentinel.py --config configs/candidate.yaml --output-csv outputs/predictions.csv
```

---

## 📁 Repository Structure

```
NEWW/
├── app/
│   └── dashboard.py                  # Streamlit interactive replay dashboard (Tabs 1-4)
├── configs/
│   ├── baseline.yaml                 # Fast statistical baseline config
│   └── candidate.yaml                # Hybrid Deep Ensemble config
├── docs/
│   ├── presentation_deck.md          # Official Utkrishta 10-slide deck
│   └── presentation_deck.html        # Interactive HTML presentation deck
├── models/
│   ├── sentinel_tcn_best.pt          # Model A: Global Generalist (Pretrained, 34.8 KB)
│   ├── sentinel_specialist_best.pt   # Model B: Infra Specialist (Pretrained, 35.1 KB)
│   └── exported/                     # JIT & TorchScript compiled artifacts
├── outputs/
│   ├── experiment_log.csv            # Multi-model baseline benchmark log
│   ├── model_comparison_scorecard.csv# Champion vs. Challenger scorecard
│   └── predictions.csv               # Generated predictions
├── scripts/
│   ├── compare_models.py             # Model A vs. Model B bake-off
│   ├── run_experiments.py            # Multi-baseline benchmark runner
│   ├── run_sentinel.py               # Streaming pipeline CLI
│   ├── train_gpu.py                  # Global model GPU trainer
│   └── train_specialist_gpu.py       # Domain specialist GPU trainer
├── src/sentinel/
│   ├── adaptation/                   # ADWIN drift, conformal scoring & feedback bank
│   ├── data/                         # Streaming NAB loader & validation
│   ├── detectors/                    # MAD, EWMA, iForest, TCN-AE & Hybrid Ensemble
│   ├── evaluation/                   # Event-based F1 & detection delay evaluator
│   ├── events/                       # Hysteresis event state machine
│   ├── explanations/                 # Deterministic evidence generator
│   ├── features/                     # Causal lag & periodicity discovery
│   ├── pipeline.py                   # Single causal streaming entry point
│   └── train.py                      # PyTorch CUDA training engine
├── tests/                            # 16 unit & integration tests (100% passing)
├── pyproject.toml                    # Package configuration
└── README.md                         # Complete system documentation
```

---

## 👥 Authors & Team
- **Satya Boli** (Lead Architect & AI/ML Engineer) — [GitHub Profile](https://github.com/SatyaBoli)
- **Utkrishta 2026 AI/ML Hackathon** — Challenge 2 (Sentinel)
