# 🏆 Utkrishta 2026 Hackathon — Official Submission Document
## Challenge 2: Sentinel (Adaptive Time-Series Anomaly Detection)

---

### 📋 Submission Metadata

- **Track**: Challenge 2 — Sentinel: Adaptive Time-Series Anomaly Detection
- **Team Name**: Sentinel ML
- **Lead Developer**: Satya Boli ([GitHub Profile](https://github.com/SatyaBoli))
- **Repository**: [https://github.com/SatyaBoli/NEWW](https://github.com/SatyaBoli/NEWW)
- **Branch**: `main`
- **Total Git Commits**: 31 atomic commits (exceeds 20-commit requirement)
- **Local Hardware Accelerated**: NVIDIA GeForce RTX 5050 Laptop GPU (CUDA 13.2, sm_120 Blackwell Architecture)
- **Target Benchmark**: Numenta Anomaly Benchmark (NAB) — 58 streaming series, 365,558 observations

---

### 🎯 Problem Statement Fulfillment

| Organizer Requirement | Sentinel Implementation Status |
| :--- | :--- |
| **Strict Causal Streaming** | Verified $100\%$ past-only dependencies. Features and statistics at time $t$ access only $\{x_1, \dots, x_{t-1}\}$. Zero lookahead leakage. |
| **Local GPU Acceleration** | Built and fine-tuned `CausalTCNAEDetector` natively on the NVIDIA RTX 5050 GPU using PyTorch CUDA 13.2 dilated causal 1D convolutions. |
| **Contamination-Guarded Adaptation** | Anomalous points are quarantined from entering the rolling baseline buffer, preventing long-running anomalies from corrupting reference statistics. |
| **Distribution Drift Adaptation** | Integrated statistical ADWIN drift detector using Hoeffding bounds to trigger governed re-baselining upon true regime shifts. |
| **Hysteresis Dual Thresholding** | Decoupled entry ($\tau_{\text{enter}} = 6.5$) and exit ($\tau_{\text{exit}} = 2.5$) thresholds with persistence requirement ($k \ge 3$) to eliminate false alarm chatter. |
| **Event-Based F1 Evaluation** | Rigorous range-based interval intersection matching adhering to Kim et al. (AAAI 2022) / Tatbul et al. (NeurIPS 2018) without point-adjustment bias. |
| **Interpretable Evidence Trail** | Every flagged event outputs an auditable JSON record with observed value, expected baseline, residual delta, relative percent change, and contributing detectors. |
| **Interactive Dashboard & Slides** | Streamlit visual replay dashboard running live on `http://localhost:8501` + official 10-slide presentation deck in HTML and Markdown. |

---

### 📊 Model Bake-Off Scorecard

Benchmark results across representative multi-domain test streams (23,395 observations):

| Model | Architecture | Event-F1 | Precision | Recall | False Positives | Throughput | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0 Baseline** | Rolling Median + MAD | `0.1111` | `10.00%` | `12.50%` | 9 | 9,554 pts/s | 0.105 ms/pt |
| **B1 Baseline** | Streaming EWMA ($\alpha=0.15$) | `0.0833` | `4.69%` | `37.50%` | 61 | 38,084 pts/s | 0.026 ms/pt |
| **B2 Candidate** | Windowed Isolation Forest | `0.0000` | `0.00%` | `0.00%` | 2 | 172 pts/s | 5.801 ms/pt |
| **B3 Deep Neural** | Causal TCN-AE (RTX 5050 GPU) | `0.1143` | `7.41%` | `25.00%` | 25 | 1,269 pts/s | 0.788 ms/pt |
| **B4 Sentinel Ensemble** | **Hybrid Ensemble + ADWIN Guard** | **`0.2857`** | **`33.33%`** | **`25.00%`** | **4** | **1,100 pts/s** | **0.910 ms/pt** |

#### Key Performance Achievements:
1. **Precision Boost**: Sentinel's Hybrid Ensemble achieved **33.33% precision**, representing a **+233% relative gain** over Rolling MAD and **+610%** over EWMA.
2. **False Alarm Suppression**: Slashed false alarms from 61 down to **only 4** (**93.4% reduction** in false positive incidents).
3. **Single-Series Verification (`ec2_cpu`)**: Neutralized daily scheduled 3:30 AM AWS backup spikes, achieving **0 False Positives**, **100% Precision**, and **0.6667 Event-F1**.
4. **Sub-Millisecond Inference**: Full multi-model consensus and neural GPU forward pass operates at **0.91 ms per observation** (over **1,100 points/second**).

---

### 📁 Generated Deliverables & File Locations

- **Submission Prediction File**: `outputs/predictions.csv` (validated against organizer schema: `series_id, timestamp, anomaly_score, label, threshold, event_id, detector_version`).
- **Evaluation Summary Metrics**: `outputs/evaluation_summary.json`
- **Multi-Model Benchmark Log**: `outputs/experiment_log.csv`
- **Trained Deep Neural Model**: `models/sentinel_tcn_best.pt`
- **Interactive Presentation Deck (10 Slides)**:
  - HTML Slide Deck: `docs/presentation_deck.html` (interactive deck with keyboard navigation)
  - Markdown Slide Deck: `docs/presentation_deck.md`
- **Interactive Streamlit Dashboard**: `app/dashboard.py` (accessible at `http://localhost:8501` or via `run_dashboard.bat`).

---

### ⚡ Rapid Verification Guide for Evaluators

#### 1. Run Unit Tests (13/13 passing):
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
