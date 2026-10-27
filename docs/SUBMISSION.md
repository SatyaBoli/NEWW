# 🏆 Utkrishta 2026 Hackathon — Official Submission Document
## Challenge 2: Sentinel (Adaptive Time-Series Anomaly Detection)

---

### 📋 Submission Metadata

- **Track**: Challenge 2 — Sentinel: Adaptive Time-Series Anomaly Detection
- **Team Name**: Sentinel ML
- **Lead Developer**: Satya Boli ([GitHub Profile](https://github.com/SatyaBoli))
- **Repository**: [https://github.com/SatyaBoli/NEWW](https://github.com/SatyaBoli/NEWW)
- **Branch**: `feature/minimalist-gold-dashboard` (Backup: `feature/dual-model-specialist`)
- **Total Git Commits**: 35+ atomic commits (exceeds 20-commit requirement)
- **Local Hardware Accelerated**: NVIDIA GeForce RTX 5050 Laptop GPU (CUDA 13.2, sm_120 Blackwell Architecture)
- **Target Benchmark**: Numenta Anomaly Benchmark (NAB) — 58 streaming series, 365,558 observations

---

### 🎯 Problem Statement Fulfillment

| Organizer Requirement | Sentinel Implementation Status |
| :--- | :--- |
| **Strict Causal Streaming** | Verified $100\%$ past-only dependencies. Features and statistics at time $t$ access only $\{x_1, \dots, x_{t-1}\}$. Zero lookahead leakage mathematically verified by future-perturbation test. |
| **Local GPU Acceleration** | Built and fine-tuned `CausalTCNAEDetector` natively on the NVIDIA RTX 5050 GPU using PyTorch CUDA 13.2 dilated causal 1D convolutions. |
| **Contamination-Guarded Adaptation** | Anomalous points are quarantined from entering the rolling baseline buffer, preventing long-running anomalies from corrupting reference statistics. |
| **Distribution Drift Adaptation** | Integrated statistical ADWIN drift detector using Hoeffding bounds to trigger governed re-baselining upon true regime shifts. |
| **Hysteresis Dual Thresholding** | Decoupled entry ($\tau_{\text{enter}} = 6.5$) and exit ($\tau_{\text{exit}} = 2.5$) thresholds with persistence requirement ($k \ge 3$) to eliminate false alarm chatter. |
| **Event-Based F1 Evaluation** | Rigorous range-based interval intersection matching adhering to Kim et al. (AAAI 2022) / Tatbul et al. (NeurIPS 2018) without point-adjustment bias. |
| **Interpretable Evidence Trail** | Every flagged event outputs an auditable JSON record with observed value, expected baseline, residual delta, relative percent change, and contributing detectors. |
| **Interactive Dashboard & Slides** | Streamlit visual replay dashboard running live on `http://localhost:8501` + official 10-slide presentation deck in PowerPoint (.pptx) and Markdown. |

---

### 📊 Model Bake-Off Scorecard

Benchmark results across representative multi-domain test streams (23,395 observations):

| Model | Architecture | Event-F1 | Precision | Recall | False Positives | Throughput | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1 Baseline B0** | Rolling Median + MAD | `0.1333` | `8.11%` | `37.50%` | 34 | 4,556 pts/s | 0.219 ms/pt |
| **E2 Baseline B1** | Streaming EWMA ($\alpha=0.15$) | `0.0833` | `4.69%` | `37.50%` | 61 | 39,599 pts/s | 0.025 ms/pt |
| **E3 Candidate B2** | Causal Isolation Forest | `0.1818` | `33.33%` | `12.50%` | 2 | 355 pts/s | 2.813 ms/pt |
| **E4 Deep Neural B3** | Causal TCN-AE (RTX 5050 GPU) | `0.2105` | `18.18%` | `25.00%` | 9 | 1,252 pts/s | 0.799 ms/pt |
| **E5 Sentinel Ensemble** | **Hybrid Ensemble + ADWIN Guard** | **`0.3750`** | **`37.50%`** | **`37.50%`** | **5** | **898 pts/s** | **1.113 ms/pt** |

#### Key Performance Achievements:
1. **Event-F1 Score**: Sentinel's Hybrid Ensemble achieves **0.3750 Event-F1** (3 TP, 5 FP, 5 FN across 23,395 points), outperforming all individual statistical and neural baselines.
2. **False Alarm Suppression**: Slashed false alarms from 61 down to **only 5** (**91.8% reduction** in false positive alarms).
3. **AWS EC2 CPU Verification (Series 11)**: Achieved **0 False Positives**, **100% Precision**, and **0.6667 Event-F1** with clean capture of anomaly event `11_evt_0001` at `2014-02-26 22:15:00` (score 7.88).
4. **Near-Millisecond Inference**: Full multi-model consensus and neural GPU forward pass operates at **1.11 ms per observation** (~**900 points/second** on the RTX 5050 GPU).

---

### 📁 Generated Deliverables & File Locations

- **Submission Prediction File**: `outputs/predictions.csv` (validated against organizer schema: `series_id, timestamp, anomaly_score, label, threshold, event_id, detector_version`).
- **Evaluation Summary Metrics**: `outputs/evaluation_summary.json`
- **Multi-Model Benchmark Log**: `outputs/experiment_log.csv`
- **Architectural Ablation Log**: `outputs/ablation_log.csv`
- **Trained Deep Neural Models**: `models/sentinel_specialist_best.pt` & `models/sentinel_tcn_best.pt`
- **Interactive Presentation Deck (10 Slides)**:
  - PowerPoint Deck: `docs/Sentinel_Presentation_Deck.pptx` (complete 10-slide deck with embedded telemetry figures)
  - Markdown Slide Deck: `docs/presentation_deck.md`
- **Interactive Streamlit Dashboard**: `app/dashboard.py` (accessible at `http://localhost:8501` or via `run_dashboard.bat`).

---

### ⚡ Rapid Verification Guide for Evaluators

#### 1. Run Unit Tests (17/17 passing):
```bash
python -m pytest tests/
```

#### 2. Reproduce Multi-Model Bake-Off Scorecard:
```bash
python scripts/run_experiments.py
```

#### 3. Run Streaming Pipeline CLI:
```bash
python scripts/run_sentinel.py --config configs/candidate.yaml --output-csv outputs/predictions.csv
```

#### 4. Launch Live Interactive Dashboard:
```bash
python -m streamlit run app/dashboard.py
# Or on Windows:
.\run_dashboard.bat
```
