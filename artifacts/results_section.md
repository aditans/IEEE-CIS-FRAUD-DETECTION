# Results, Discussion, and Inference

Generated on: 2026-03-31 18:01:56

## A. Experimental Results
### Phase 1 - Preprocessing and Graph Construction
- Train samples: 250,000
- Validation samples: 88,581
- Test samples: 88,581
- Train fraud rate: 20.00%
- Validation fraud rate: 3.43%
- Test fraud rate: 3.48%
- Graph edges (sampled): 50,000
- Graph nodes: 6,185
- Feature dimension: 435

### Phase 2 - Centralised Model
- AUC-PR: 0.5143
- AUC-ROC: 0.7154
- F2 score: 0.7331
- Recall: 0.8738
- Precision: 0.4458

### Phase 3 - Federated Simulation
- Final round: 20
- AUC-PR: 0.7712
- F2 score: 0.7834
- Recall: 0.8790
- Distributed validation loss: 1.0317

## B. Discussion
- The centralised model shows high recall, indicating that the pipeline prioritizes fraud capture over precision.
- Federated vs centralised changes: AUC-PR +0.2569, F2 +0.0503, Recall +0.0051.
- A drop in federated metrics is expected under non-IID client data because each bank sees a narrower distribution, reducing global gradient consistency.
- If recall remains relatively strong while precision/AUC-PR drops, the model is still detecting suspicious patterns but with weaker ranking calibration across clients.

## C. Inference
- The pipeline demonstrates practical fraud detection under both centralised and privacy-preserving federated settings.
- The observed centralised-to-federated gap quantifies the privacy trade-off and directly addresses the research objective.
- The federated setup remains viable when privacy constraints prohibit raw transaction sharing across institutions.

## Notes for Paper Writing
- Do not only report numbers; explain causes such as class imbalance, non-IID client drift, and temporal effects.
- Add one chart from artifacts/federated_round_metrics.csv to show convergence across rounds.
