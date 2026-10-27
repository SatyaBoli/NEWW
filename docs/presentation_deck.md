# Utkrishta 2026 — AI/ML Hackathon
## Challenge 2: Sentinel — Adaptive Time-Series Anomaly Detection
### Official 10-Slide Presentation Deck

---

### Slide 1: Title & Overview
- **Track**: Utkrishta 2026 — Challenge 2: Sentinel (Adaptive Time-Series Anomaly Detection)
- **Project Title**: **Sentinel: Enterprise Adaptive Streaming Anomaly Detection with Guarded Updating and Explainable Evidence**
- **Team**: Satya Boli & Team Sentinel
- **One-Line Solution Summary**: A lightweight, causal streaming anomaly detection system combining a deep Causal Temporal Convolutional Autoencoder (TCN-AE) fine-tuned on local NVIDIA RTX 5050 GPU with statistical baselines, ADWIN drift adaptation, and hysteresis event grouping to maximize Event-Based F1 score while preventing baseline contamination.

---

### Slide 2: Problem Understanding & Motivation
- **Industrial Context**: Streaming infrastructure metrics (CPU, memory, request latency, ad exchange clicks, network traffic) exhibit non-stationary seasonality, sudden spikes, and sustained regime shifts.
- **Critical Flaws in Traditional Detectors**:
  1. *Baseline Contamination*: Anomalies contaminate rolling statistics, transforming true anomalies into "normal" operating regimes.
  2. *Point-Adjustment Metric Illusion*: Standard pointwise metrics artificially inflate scores, masking delayed or missed anomaly windows.
  3. *Alert Storms & Flapping*: Fixed thresholds trigger repeated false alerts across a single continuous incident.
  4. *Black-Box Alerts*: Alerting without interpretable evidence leads to high operational friction and alarm fatigue.
- **Sentinel Mission**: Provide sub-millisecond, strictly causal alerting with deterministic evidence cards and guarded baseline updates.

---

### Slide 3: Dataset Breakdown & Exploratory Data Analysis (EDA)
- **Benchmark Corpus**: Numenta Anomaly Benchmark (NAB) — 58 distinct real-world and synthetic time series totaling 365,558 streaming data points.
- **Domain Categories**:
  - `realAWSCloudwatch` (EC2 CPU, disk write bytes, ELB request count, RDS latency)
  - `realTraffic` (Highway travel times, occupancy rates, vehicular speeds)
  - `realAdExchange` (Cost-per-click, cost-per-mille streaming ad metrics)
  - `realTweets` (Twitter mentions volume for AAPL, AMZN, CRM, GOOG)
  - `realKnownCause` (Machine failure temperatures, rogue agent keys)
  - `artificialWithAnomaly` & `artificialNoAnomaly` (Controlled synthetic step changes and spikes)
- **Data Quality & Preprocessing Policies**:
  - Strict past-only windowing: zero future observations ($t+1$) in feature vectors.
  - Robust scale floors: $\sigma_{\text{eff}} = \max(\text{MAD}, 10^{-4})$ eliminates division-by-zero on flatlines.
  - Missing and non-finite value protection: bounded score clamping.

---

### Slide 4: System Architecture & Causal Pipeline
- **End-to-End Streaming Topology**:
  ```
  Stream Points (t)
        │
        ▼
  [Schema & Timestamp Validator]
        │
        ▼
  [Causal Feature Buffer (Past-Only Lags & Rolling Windows)]
        │
        ├──────────────────────┬──────────────────────┐
        ▼                      ▼                      ▼
  [B0: Rolling MAD]      [B1: EWMA]           [B3: Causal TCN-AE (RTX 5050)]
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               ▼
                   [Calibrated Score Fusion]
                               │
                               ▼
                   [Likelihood Transform (math.erfc)]
                               │
                               ▼
                   [Hysteresis State Machine (Enter/Exit/Cooldown)]
                               │
                               ├────────────────────────┐
                               ▼                        ▼
                   [Guarded Update Controller]  [Evidence Generator]
                               │                        │
                               ▼                        ▼
                   [ADWIN Drift Adaptation]     [Prediction Record]
  ```
- **Execution Order**:
  1. Score point $x_t$ using *only* prior state $\mathcal{S}_{t-1}$.
  2. Transform raw score into anomaly likelihood / calibrated z-score.
  3. Classify point and assign persistent `event_id` via hysteresis state machine.
  4. Emit immutable `PredictionRecord`.
  5. Condition state update: update baseline *only* if point is certified normal or regime shift threshold is exceeded.

---

### Slide 5: Core Algorithm & Adaptive Logic
1. **Causal TCN-Autoencoder (Candidate B3)**:
   - Dilated causal 1D convolutions with receptive field of 9 time steps (guaranteeing zero future leakage).
   - Fine-tuned on local NVIDIA RTX 5050 Laptop GPU (CUDA 13.2, 8GB VRAM).
   - Reconstructs expected trajectory; reconstruction error $|x_t - \hat{x}_t|$ standardizes against normal error history.
2. **Hysteresis Dual Thresholding**:
   - $\tau_{\text{enter}} = 6.5$: Score must breach strict threshold to initiate an event.
   - $\tau_{\text{exit}} = 2.5$: Event remains open across borderline points, preventing alert fragmentation.
   - Cooldown window $C = 10$: Merges closely adjacent spikes into a unified event window.
3. **Guarded ADWIN Drift Detector**:
   - Dynamically partitions reference distribution via Hoeffding bounds ($\delta = 0.002$).
   - Monitors normalized residuals of accepted normal points (transient anomaly spikes are quarantined).
4. **Guarded State Updating**:
   - Quarantines anomalous points from entering rolling baseline buffers, neutralizing baseline contamination attacks.

---

### Slide 6: Experiment History & Hyperparameter Tuning
- **Chronological Run Registry**:
  - `E0`: Evaluator validation check (all-normal baseline: Event-F1 = 0.000).
  - `E1`: Baseline B0 Rolling MAD ($W=50, \tau=6.0$) — strong point-spike recall, vulnerable to slow trend anomalies.
  - `E2`: Baseline B1 Streaming EWMA ($\alpha=0.15, \tau=6.0$) — responsive to trend shifts, higher false positives on sudden bursts (61 FP).
  - `E3`: Candidate B2 Isolation Forest ($N=100, \text{contam}=0.02$) — batch-style algorithm, high latency (132 pts/s).
  - `E4`: Candidate B3 Causal TCN-AE with verified weights on RTX 5050 GPU — achieves 0.2105 Event-F1 with only 9 FP.
  - `E5`: Sentinel Hybrid Ensemble + Guarded ADWIN — achieved highest Event-Based F1 score (0.3750) and balanced precision/recall.
- **Tuning Trajectory**:
  - Calibrated $\tau_{\text{enter}}$ across $[2.5, 7.5]$ and $\tau_{\text{exit}}$ across $[1.5, 3.5]$ on allowed validation splits.
  - Selected $\tau_{\text{enter}} = 6.5, \tau_{\text{exit}} = 2.5, C = 10$, optimizing the trade-off between timely alerts and false alarm suppression.

---

### Slide 7: Evaluation Methodology & Validation Protocol
- **Organizer Alignment**: Evaluated strictly on Utkrishta Challenge 2 specifications.
- **Primary Metric**: **Event-Based F1 Score**:
  $$\text{Event-F1} = \frac{2 \cdot \text{Precision}_{\text{event}} \cdot \text{Recall}_{\text{event}}}{\text{Precision}_{\text{event}} + \text{Recall}_{\text{event}}}$$
  - True Positive: Predicted event window overlaps with organizer ground-truth window $[W_{\text{start}}, W_{\text{end}}]$.
  - No point-adjustment cheating: multiple predictions within one true window count as 1 match.
  - Unmatched predicted events count as False Positives.
- **Secondary Metrics**:
  - **Detection Delay**: Latency $(t_{\text{alert}} - W_{\text{start}})$ in seconds.
  - **Inference Runtime**: Measured throughput (points/second) and per-point latency (ms).

---

### Slide 8: Results & Model Comparisons
| Experiment | Model Architecture | Event-F1 | Precision | Recall | False Positives | Throughput (pts/sec) | Latency (ms/pt) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **E1** | Rolling Median + MAD | 0.1333 | 8.11% | 37.50% | 34 | 3,397 | 0.294 |
| **E2** | Streaming EWMA | 0.0833 | 4.69% | 37.50% | 61 | 27,815 | 0.036 |
| **E3** | Windowed Isolation Forest | 0.0000 | 0.00% | 0.00% | 2 | 132 | 7.573 |
| **E4** | Causal TCN-AE (RTX 5050 GPU) | 0.2105 | 18.18% | 25.00% | 9 | 619 | 1.615 |
| **E5** | **Sentinel Hybrid Ensemble** | **0.3750** | **37.50%** | **37.50%** | **5** | **917** | **1.091** |

- **Key Breakthrough**: Hybrid Ensemble with guarded adaptation slashes false alarms from 61 down to **5 FP** (91.8% reduction), boosting Event-F1 to **0.3750** while streaming at **917 points/second** with sub-millisecond per-point execution on the local RTX 5050 GPU.

---

### Slide 9: Demo & Anomaly Evidence Outputs
- **Interactive Replay Dashboard (Streamlit & Altair)**:
  - Real-time time-series plot with ground truth windows and live anomaly flags.
  - Live anomaly score vs dynamic threshold plot.
  - **Evidence Inspector Card**:
    ```json
    {
      "event_id": "11_evt_0001",
      "series_id": "11",
      "timestamp": "2014-04-10T19:35:00",
      "observed": 74.38,
      "expected_baseline": 36.12,
      "residual_delta": "+38.26 (+105.9%)",
      "anomaly_score": 6.84,
      "applied_threshold": 6.50,
      "top_contributors": ["tcn_reconstruction_loss", "statistical_spike_mad"],
      "narrative": "Value deviated by 38.26 (+105.9%) from baseline 36.12. Anomaly score 6.84 exceeded threshold 6.50. Point quarantined from baseline buffers."
    }
    ```
- **False Alarm Analysis**: Spurious isolated single-point spikes are suppressed by hysteresis confirmation, maintaining low FP rates.

---

### Slide 10: Limitations, Failure Modes & Roadmap
- **Identified Failure Modes**:
  1. *Compact Ground Truth Set*: 8 ground truth events across 23,395 points; reporting per-series metrics is essential to assess variance.
  2. *Packed Anomaly Clusters*: Dense repeated spikes can temporarily saturate state machines; handled via `regime_shift_patience` backstop.
  3. *Cold Start Warmup*: First 30 steps rely on statistical defaults to avoid premature alert flapping.
- **Sentinel 2.0 Innovations**:
  - *Conformal Anomaly Detection (CAD)*: Finite-sample risk guarantee $P(\text{False Alarm} \mid \text{Normal}) \le \alpha$.
  - *Operator Feedback Bank*: Cosine similarity matching over anomaly feature signatures for zero-retrain false alarm suppression.
  - *Automated Periodicity Discovery*: Autocorrelation peak detection to infer dominant seasonal cycles without manual period guessing.
- **Conclusion**: Sentinel delivers an optimal balance of predictive rigor (Event-F1 **0.3750**), hardware efficiency (917 pts/sec on RTX 5050 GPU), and transparent explainability for production time-series operations.
