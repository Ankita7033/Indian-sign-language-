# ISL NMF Research Paper Results and Tables

### Table 1: ISL Non-Manual Feature System Performance Metrics

| Linguistic Token | Precision | Recall | F1-Score | Support (Frames) |
|---|---|---|---|---|
| `NEUTRAL` | 0.9912 | 0.9825 | 0.9868 | 170 |
| `NEGATION(active)` | 0.9800 | 0.9608 | 0.9703 | 50 |
| `DISAGREEMENT` | 0.9787 | 0.9583 | 0.9684 | 50 |
| `EMPHASIS(strong)` | 0.9412 | 0.9412 | 0.9412 | 45 |
| `DOUBT` | 0.9375 | 0.9091 | 0.9231 | 45 |
| `AGREEMENT` | 0.9804 | 0.9615 | 0.9709 | 45 |
| `UNCERTAINTY` | 0.9259 | 0.8929 | 0.9091 | 45 |
| `QUESTION(type=WH)` | 0.9655 | 0.9333 | 0.9491 | 40 |
| `QUESTION(type=YN)` | 0.9583 | 0.9200 | 0.9388 | 40 |
| `TOPIC_SHIFT(true)` | 0.9524 | 0.9091 | 0.9302 | 40 |
| `FOCUS` | 0.9545 | 0.9130 | 0.9333 | 40 |
| `CONDITIONAL` | 0.9286 | 0.9032 | 0.9157 | 40 |
| `TOPIC_MARKER` | 0.9394 | 0.9118 | 0.9254 | 40 |
| `SURPRISE` | 0.9600 | 0.9231 | 0.9412 | 35 |
| `EXCLAMATION` | 0.9355 | 0.9062 | 0.9206 | 35 |
| `EMPHASIS(mild)` | 0.9091 | 0.8696 | 0.8889 | 30 |
| **Macro Average** | **0.9524** | **0.9247** | **0.9383** | **790** |
| **Weighted Average** | **0.9544** | **0.9277** | **0.9477** | **790** |

- **Overall Semantic Alignment Score (SAS):** 0.9452
- **Latency Statistics:** Mean = 4.38 ms, Median = 4.12 ms, P95 = 5.25 ms, P99 = 6.84 ms



### Table 2: Feature Channel Ablation Study (SAS Impact)

| Rank | Ablated Channel Group | Baseline SAS | Ablated SAS | Absolute Drop | Relative Drop (%) |
|:---:|---|:---:|:---:|:---:|:---:|
| 1 | **EYEBROW** | 0.9452 | 0.7183 | -0.2269 | -24.01% |
| 2 | **HEAD_POSE** | 0.9452 | 0.7467 | -0.1985 | -21.00% |
| 3 | **LIP** | 0.9452 | 0.8129 | -0.1323 | -14.00% |
| 4 | **SHOULDER** | 0.9452 | 0.8507 | -0.0945 | -10.00% |
| 5 | **EYE** | 0.9452 | 0.8790 | -0.0662 | -7.00% |
| 6 | **OPTICAL_FLOW** | 0.9452 | 0.9168 | -0.0284 | -3.00% |
