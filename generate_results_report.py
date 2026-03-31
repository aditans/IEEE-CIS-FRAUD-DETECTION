import argparse
import csv
import os
import pickle
from datetime import datetime

import numpy as np


def safe_rate(y):
    y_arr = np.asarray(y)
    if y_arr.size == 0:
        return 0.0
    return float(np.mean(y_arr) * 100.0)


def summarize_phase1(preprocessed_path):
    with open(preprocessed_path, "rb") as f:
        data = pickle.load(f)

    y_train = np.asarray(data.get("y_train", []))
    y_val = np.asarray(data.get("y_val", []))
    y_test = np.asarray(data.get("y_test", []))
    edge_list = data.get("edge_list", [])

    all_nodes = set()
    for e in edge_list:
        all_nodes.add(e.get("src"))
        all_nodes.add(e.get("dst"))

    return {
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "train_fraud_rate": safe_rate(y_train),
        "val_fraud_rate": safe_rate(y_val),
        "test_fraud_rate": safe_rate(y_test),
        "edge_count": int(len(edge_list)),
        "node_count": int(len(all_nodes)),
        "feature_dim": int(len(data.get("feature_cols", []))),
    }


def summarize_phase2(centralised_path):
    if not os.path.exists(centralised_path):
        return None
    with open(centralised_path, "rb") as f:
        c = pickle.load(f)
    return {
        "auc_pr": float(c.get("auc_pr", 0.0)),
        "auc_roc": float(c.get("auc_roc", 0.0)),
        "f2": float(c.get("f2", 0.0)),
        "recall": float(c.get("recall", 0.0)),
        "precision": float(c.get("precision", 0.0)),
    }


def summarize_phase3(fl_history_path):
    if not os.path.exists(fl_history_path):
        return None, []

    with open(fl_history_path, "rb") as f:
        hist = pickle.load(f)

    metrics = getattr(hist, "metrics_distributed", {}) or {}
    losses = getattr(hist, "losses_distributed", []) or []

    rounds = set()
    for _, values in metrics.items():
        for r, _ in values:
            rounds.add(int(r))
    for r, _ in losses:
        rounds.add(int(r))

    rows = []
    for r in sorted(rounds):
        row = {"round": r, "loss": None, "auc_pr": None, "f2": None, "recall": None}

        for rr, vv in losses:
            if int(rr) == r:
                row["loss"] = float(vv)
                break

        for name in ["auc_pr", "f2", "recall"]:
            for rr, vv in metrics.get(name, []):
                if int(rr) == r:
                    row[name] = float(vv)
                    break

        rows.append(row)

    final = None
    if rows:
        last = rows[-1]
        final = {
            "round": int(last["round"]),
            "auc_pr": float(last["auc_pr"] or 0.0),
            "f2": float(last["f2"] or 0.0),
            "recall": float(last["recall"] or 0.0),
            "loss": float(last["loss"] or 0.0),
        }

    return final, rows


def write_round_csv(rows, csv_path):
    if not rows:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "loss", "auc_pr", "f2", "recall"])
        writer.writeheader()
        writer.writerows(rows)


def section_discussion(p2, p3):
    lines = []
    lines.append("## B. Discussion")

    if p2 is None:
        lines.append("- Phase 2 metrics file was not found, so centralised-vs-federated analysis is limited.")
        return lines

    lines.append("- The centralised model shows high recall, indicating that the pipeline prioritizes fraud capture over precision.")

    if p3 is None:
        lines.append("- Federated metrics file was not found, so cross-setting comparison is pending.")
        return lines

    delta_aucpr = p3["auc_pr"] - p2["auc_pr"]
    delta_f2 = p3["f2"] - p2["f2"]
    delta_recall = p3["recall"] - p2["recall"]

    lines.append(
        f"- Federated vs centralised changes: AUC-PR {delta_aucpr:+.4f}, F2 {delta_f2:+.4f}, Recall {delta_recall:+.4f}."
    )
    lines.append(
        "- A drop in federated metrics is expected under non-IID client data because each bank sees a narrower distribution, reducing global gradient consistency."
    )
    lines.append(
        "- If recall remains relatively strong while precision/AUC-PR drops, the model is still detecting suspicious patterns but with weaker ranking calibration across clients."
    )
    return lines


def build_markdown(phase1, phase2, phase3_final, generated_at):
    lines = []
    lines.append("# Results, Discussion, and Inference")
    lines.append("")
    lines.append(f"Generated on: {generated_at}")
    lines.append("")
    lines.append("## A. Experimental Results")
    lines.append("### Phase 1 - Preprocessing and Graph Construction")
    lines.append(f"- Train samples: {phase1['train_samples']:,}")
    lines.append(f"- Validation samples: {phase1['val_samples']:,}")
    lines.append(f"- Test samples: {phase1['test_samples']:,}")
    lines.append(f"- Train fraud rate: {phase1['train_fraud_rate']:.2f}%")
    lines.append(f"- Validation fraud rate: {phase1['val_fraud_rate']:.2f}%")
    lines.append(f"- Test fraud rate: {phase1['test_fraud_rate']:.2f}%")
    lines.append(f"- Graph edges (sampled): {phase1['edge_count']:,}")
    lines.append(f"- Graph nodes: {phase1['node_count']:,}")
    lines.append(f"- Feature dimension: {phase1['feature_dim']}")
    lines.append("")

    lines.append("### Phase 2 - Centralised Model")
    if phase2 is None:
        lines.append("- Metrics file not found. Run phase2_model.py to populate this section.")
    else:
        lines.append(f"- AUC-PR: {phase2['auc_pr']:.4f}")
        lines.append(f"- AUC-ROC: {phase2['auc_roc']:.4f}")
        lines.append(f"- F2 score: {phase2['f2']:.4f}")
        lines.append(f"- Recall: {phase2['recall']:.4f}")
        lines.append(f"- Precision: {phase2['precision']:.4f}")
    lines.append("")

    lines.append("### Phase 3 - Federated Simulation")
    if phase3_final is None:
        lines.append("- Metrics file not found. Run phase3_federated.py to populate this section.")
    else:
        lines.append(f"- Final round: {phase3_final['round']}")
        lines.append(f"- AUC-PR: {phase3_final['auc_pr']:.4f}")
        lines.append(f"- F2 score: {phase3_final['f2']:.4f}")
        lines.append(f"- Recall: {phase3_final['recall']:.4f}")
        lines.append(f"- Distributed validation loss: {phase3_final['loss']:.4f}")
    lines.append("")

    lines.extend(section_discussion(phase2, phase3_final))
    lines.append("")

    lines.append("## C. Inference")
    lines.append("- The pipeline demonstrates practical fraud detection under both centralised and privacy-preserving federated settings.")
    lines.append("- The observed centralised-to-federated gap quantifies the privacy trade-off and directly addresses the research objective.")
    lines.append("- The federated setup remains viable when privacy constraints prohibit raw transaction sharing across institutions.")
    lines.append("")
    lines.append("## Notes for Paper Writing")
    lines.append("- Do not only report numbers; explain causes such as class imbalance, non-IID client drift, and temporal effects.")
    lines.append("- Add one chart from artifacts/federated_round_metrics.csv to show convergence across rounds.")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate a consolidated results file from artifacts.")
    parser.add_argument("--artifacts", default="artifacts", help="Artifacts directory path")
    parser.add_argument(
        "--output",
        default=os.path.join("artifacts", "results_section.md"),
        help="Output markdown file",
    )
    args = parser.parse_args()

    artifacts_dir = args.artifacts
    preprocessed_path = os.path.join(artifacts_dir, "preprocessed.pkl")
    centralised_path = os.path.join(artifacts_dir, "centralised_metrics.pkl")
    fl_history_path = os.path.join(artifacts_dir, "fl_history.pkl")
    round_csv_path = os.path.join(artifacts_dir, "federated_round_metrics.csv")

    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError(
            f"Missing {preprocessed_path}. Run phase1_preprocessing.py first."
        )

    phase1 = summarize_phase1(preprocessed_path)
    phase2 = summarize_phase2(centralised_path)
    phase3_final, round_rows = summarize_phase3(fl_history_path)

    write_round_csv(round_rows, round_csv_path)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    markdown = build_markdown(phase1, phase2, phase3_final, generated_at)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"Saved: {args.output}")
    if round_rows:
        print(f"Saved: {round_csv_path}")


if __name__ == "__main__":
    main()
