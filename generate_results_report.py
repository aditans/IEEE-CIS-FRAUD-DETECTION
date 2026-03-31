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
    with open(preprocessed_path, 'rb') as f:
        data = pickle.load(f)

    y_train = np.asarray(data.get('y_train', []))
    y_val = np.asarray(data.get('y_val', []))
    y_test = np.asarray(data.get('y_test', []))
    edge_list = data.get('edge_list', [])

    all_nodes = set()
    for e in edge_list:
        all_nodes.add(e.get('src'))
        all_nodes.add(e.get('dst'))

    return {
        'train_samples': int(len(y_train)),
        'val_samples': int(len(y_val)),
        'test_samples': int(len(y_test)),
        'train_fraud_rate': safe_rate(y_train),
        'val_fraud_rate': safe_rate(y_val),
        'test_fraud_rate': safe_rate(y_test),
        'edge_count': int(len(edge_list)),
        'node_count': int(len(all_nodes)),
        'feature_dim': int(len(data.get('feature_cols', []))),
    }


def summarize_phase2(centralised_path):
    if not os.path.exists(centralised_path):
        return None

    with open(centralised_path, 'rb') as f:
        c = pickle.load(f)

    return {
        'auc_pr': float(c.get('auc_pr', 0.0)),
        'auc_roc': float(c.get('auc_roc', 0.0)),
        'f2': float(c.get('f2', 0.0)),
        'recall': float(c.get('recall', 0.0)),
        'precision': float(c.get('precision', 0.0)),
        'fpr': float(c.get('fpr', 0.0)),
        'tn': int(c.get('tn', 0)),
        'fp': int(c.get('fp', 0)),
        'fn': int(c.get('fn', 0)),
        'tp': int(c.get('tp', 0)),
    }


def _history_to_dict(history_obj):
    out = {'metrics_distributed': {}, 'losses_distributed': [], 'metrics_centralized': {}, 'losses_centralized': []}

    out['metrics_distributed'] = getattr(history_obj, 'metrics_distributed', {}) or {}
    out['losses_distributed'] = getattr(history_obj, 'losses_distributed', []) or []
    out['metrics_centralized'] = getattr(history_obj, 'metrics_centralized', {}) or {}
    out['losses_centralized'] = getattr(history_obj, 'losses_centralized', []) or []
    return out


def summarize_phase3(fl_history_path, federated_metrics_path):
    history = None
    fed_metrics = None

    if os.path.exists(fl_history_path):
        with open(fl_history_path, 'rb') as f:
            history = _history_to_dict(pickle.load(f))

    if os.path.exists(federated_metrics_path):
        with open(federated_metrics_path, 'rb') as f:
            fed_metrics = pickle.load(f)

    if history is None and fed_metrics is None:
        return None, [], None

    rows = []
    rounds = set()

    if history is not None:
        for _, vals in history['metrics_distributed'].items():
            for r, _ in vals:
                rounds.add(int(r))
        for _, vals in history['metrics_centralized'].items():
            for r, _ in vals:
                rounds.add(int(r))
        for r, _ in history['losses_distributed']:
            rounds.add(int(r))
        for r, _ in history['losses_centralized']:
            rounds.add(int(r))

    for r in sorted(rounds):
        row = {
            'round': r,
            'loss_distributed': None,
            'loss_global': None,
            'auc_pr_distributed': None,
            'auc_pr_global': None,
            'f2_global': None,
            'recall_global': None,
            'precision_global': None,
            'auc_roc_global': None,
            'fpr_global': None,
        }

        if history is not None:
            for rr, vv in history['losses_distributed']:
                if int(rr) == r:
                    row['loss_distributed'] = float(vv)
                    break
            for rr, vv in history['losses_centralized']:
                if int(rr) == r:
                    row['loss_global'] = float(vv)
                    break

            for rr, vv in history['metrics_distributed'].get('auc_pr', []):
                if int(rr) == r:
                    row['auc_pr_distributed'] = float(vv)
                    break

            for key in ['auc_pr_global', 'f2_global', 'recall_global', 'precision_global', 'auc_roc_global', 'fpr_global']:
                for rr, vv in history['metrics_centralized'].get(key, []):
                    if int(rr) == r:
                        row[key] = float(vv)
                        break

        rows.append(row)

    final = None
    if fed_metrics and fed_metrics.get('global_eval_final'):
        g = fed_metrics['global_eval_final']
        final = {
            'round': int(rows[-1]['round']) if rows else 0,
            'auc_pr': float(g.get('auc_pr_global', 0.0)),
            'auc_roc': float(g.get('auc_roc_global', 0.0)),
            'f2': float(g.get('f2_global', 0.0)),
            'recall': float(g.get('recall_global', 0.0)),
            'precision': float(g.get('precision_global', 0.0)),
            'fpr': float(g.get('fpr_global', 0.0)),
            'tn': int(g.get('tn_global', 0)),
            'fp': int(g.get('fp_global', 0)),
            'fn': int(g.get('fn_global', 0)),
            'tp': int(g.get('tp_global', 0)),
            'source': 'global_holdout',
        }
    elif rows:
        last = rows[-1]
        final = {
            'round': int(last['round']),
            'auc_pr': float(last['auc_pr_global'] or last['auc_pr_distributed'] or 0.0),
            'auc_roc': float(last['auc_roc_global'] or 0.0),
            'f2': float(last['f2_global'] or 0.0),
            'recall': float(last['recall_global'] or 0.0),
            'precision': float(last['precision_global'] or 0.0),
            'fpr': float(last['fpr_global'] or 0.0),
            'tn': 0,
            'fp': 0,
            'fn': 0,
            'tp': 0,
            'source': 'history_fallback',
        }

    return final, rows, fed_metrics


def write_round_csv(rows, csv_path):
    if not rows:
        return
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'round',
                'loss_distributed',
                'loss_global',
                'auc_pr_distributed',
                'auc_pr_global',
                'f2_global',
                'recall_global',
                'precision_global',
                'auc_roc_global',
                'fpr_global',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def make_figures(artifacts_dir, rows, phase2, phase3_final, psi_path, preprocessed_path):
    figs = {}

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        return figs

    sns.set_theme(style='whitegrid')
    fig_dir = os.path.join(artifacts_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    if rows:
        rounds = [r['round'] for r in rows]
        auc_global = [r['auc_pr_global'] if r['auc_pr_global'] is not None else np.nan for r in rows]
        auc_dist = [r['auc_pr_distributed'] if r['auc_pr_distributed'] is not None else np.nan for r in rows]
        loss_global = [r['loss_global'] if r['loss_global'] is not None else np.nan for r in rows]
        loss_dist = [r['loss_distributed'] if r['loss_distributed'] is not None else np.nan for r in rows]

        fig, ax1 = plt.subplots(figsize=(11, 6))
        ax2 = ax1.twinx()

        ax1.plot(rounds, loss_global, marker='o', linewidth=2, label='Global Holdout Loss', color='#0d3b66')
        ax1.plot(rounds, loss_dist, marker='o', linewidth=1.5, linestyle='--', label='Client-Aggregated Loss', color='#3d5a80')
        ax2.plot(rounds, auc_global, marker='s', linewidth=2, label='Global Holdout AUC-PR', color='#ef476f')
        ax2.plot(rounds, auc_dist, marker='s', linewidth=1.5, linestyle='--', label='Client-Avg AUC-PR', color='#ff8fab')

        ax1.set_xlabel('Communication Round')
        ax1.set_ylabel('Validation Loss')
        ax2.set_ylabel('AUC-PR')
        ax1.set_title('Federated Convergence Across Rounds')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

        out = os.path.join(fig_dir, 'federated_convergence.png')
        fig.tight_layout()
        fig.savefig(out, dpi=220)
        plt.close(fig)
        figs['convergence'] = out

    if phase2 and phase3_final:
        metrics = ['auc_pr', 'f2', 'recall', 'precision', 'auc_roc', 'fpr']
        central_vals = [phase2.get(m, 0.0) for m in metrics]
        fed_vals = [phase3_final.get(m, 0.0) for m in metrics]

        x = np.arange(len(metrics))
        width = 0.37

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(x - width / 2, central_vals, width, label='Centralised', color='#118ab2')
        ax.bar(x + width / 2, fed_vals, width, label='Federated (Global Holdout)', color='#f4a261')

        ax.set_xticks(x)
        ax.set_xticklabels(['AUC-PR', 'F2', 'Recall', 'Precision', 'AUC-ROC', 'FPR'])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Score')
        ax.set_title('Centralised vs Federated Metric Comparison')
        ax.legend()

        out = os.path.join(fig_dir, 'centralised_vs_federated_metrics.png')
        fig.tight_layout()
        fig.savefig(out, dpi=220)
        plt.close(fig)
        figs['comparison'] = out

    if os.path.exists(psi_path):
        with open(psi_path, 'rb') as f:
            psi_data = pickle.load(f)

        b1 = np.asarray(psi_data.get('bank1_feature0', []), dtype=np.float32)
        b2 = np.asarray(psi_data.get('bank2_feature0', []), dtype=np.float32)
        psi_val = float(psi_data.get('psi', 0.0))

        if b1.size > 0 and b2.size > 0:
            q_upper = float(np.quantile(np.concatenate([b1, b2]), 0.995))
            q_upper = max(q_upper, 1.0)
            bins = np.linspace(0.0, q_upper, 40)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
            fig.suptitle(f'Feature Drift (Proxy: Feature[0]) | PSI={psi_val:.4f}', fontsize=13)

            ax1.hist(np.clip(b1, 0.0, q_upper), bins=bins, alpha=0.8, color='#1d3557', density=True)
            ax1.set_title('Bank 1 Distribution')
            ax1.set_xlabel('Feature Value (clipped at 99.5th pct)')
            ax1.set_ylabel('Density')

            ax2.hist(np.clip(b2, 0.0, q_upper), bins=bins, alpha=0.8, color='#e76f51', density=True)
            ax2.set_title('Bank 2 Distribution')
            ax2.set_xlabel('Feature Value (clipped at 99.5th pct)')

            out = os.path.join(fig_dir, 'psi_histogram.png')
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.savefig(out, dpi=220)
            plt.close(fig)
            figs['psi'] = out

    # Optional SHAP summary using a surrogate tabular classifier.
    # This keeps explainability available even though the core model is a GNN.
    try:
        import pandas as pd
        import shap
        from sklearn.ensemble import RandomForestClassifier

        with open(preprocessed_path, 'rb') as f:
            p = pickle.load(f)

        X_train = p.get('X_train')
        y_train = np.asarray(p.get('y_train', []))
        if X_train is not None and len(y_train) > 500:
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train)

            X_num = X_train.select_dtypes(include=[np.number]).copy()
            if X_num.shape[1] > 0:
                n_fit = min(len(X_num), 5000)
                n_bg = min(len(X_num), 400)
                n_shap = min(len(X_num), 1200)

                X_fit = X_num.iloc[:n_fit]
                y_fit = y_train[:n_fit]
                X_bg = X_num.iloc[:n_bg]
                X_eval = X_num.iloc[:n_shap]

                rf = RandomForestClassifier(
                    n_estimators=180,
                    max_depth=10,
                    class_weight='balanced_subsample',
                    random_state=42,
                    n_jobs=-1,
                )
                rf.fit(X_fit, y_fit)

                explainer = shap.TreeExplainer(rf)
                shap_values = explainer.shap_values(X_eval)

                # Normalize SHAP output shape across SHAP/sklearn versions.
                if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                    sv = shap_values[:, :, 1]
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
                    sv = shap_values
                elif isinstance(shap_values, list) and len(shap_values) > 1:
                    sv = shap_values[1]
                elif isinstance(shap_values, list):
                    sv = shap_values[0]
                else:
                    sv = shap_values

                shap.summary_plot(
                    sv,
                    X_eval,
                    show=False,
                    plot_type='dot',
                    max_display=15,
                )
                fig = plt.gcf()
                fig.set_size_inches(12, 7)
                fig.subplots_adjust(left=0.30, right=0.98, top=0.92, bottom=0.12)
                out = os.path.join(fig_dir, 'shap_summary.png')
                fig.tight_layout()
                fig.savefig(out, dpi=220)
                plt.close(fig)
                figs['shap'] = out
    except Exception:
        pass

    return figs


def section_discussion(phase2, phase3):
    lines = []
    lines.append('## B. Discussion')

    if phase2 is None:
        lines.append('- Phase 2 metrics file was not found, so centralised-vs-federated analysis is limited.')
        return lines

    lines.append('- High Recall is expected because preprocessing increased minority presence (SMOTE/quick_balance) and the loss used positive weighting (pos_weight=3.0), which shifts the decision boundary toward catching fraud positives.')
    lines.append('- In class-weighted BCE, positive-class gradient magnitude is scaled by pos_weight, so false negatives are penalized more strongly. This raises TP at the expense of more FP, which explains strong Recall with moderate Precision.')

    if phase3 is None:
        lines.append('- Federated metrics file was not found, so cross-setting comparison is pending.')
        return lines

    delta_aucpr = phase3['auc_pr'] - phase2['auc_pr']
    delta_f2 = phase3['f2'] - phase2['f2']
    delta_recall = phase3['recall'] - phase2['recall']

    lines.append(
        f"- Using the shared global holdout view, federated vs centralised deltas are: AUC-PR {delta_aucpr:+.4f}, F2 {delta_f2:+.4f}, Recall {delta_recall:+.4f}."
    )
    lines.append(
        '- If federated remains stronger after this correction, the likely cause is regularization via decentralized updates and reduced overfitting to a single pooled distribution.'
    )
    lines.append(
        '- If federated drops after correction (common under non-IID splits), this indicates client gradient disagreement and weaker global ranking calibration.'
    )
    lines.append(
        '- Prior anomalous comparisons often happen when federated AUC-PR is reported as weighted average of per-client AUC-PR, which is not directly equivalent to pooled/global AUC-PR.'
    )

    return lines


def build_markdown(phase1, phase2, phase3_final, generated_at, fig_paths):
    def rel_to_results(path):
        rel = os.path.relpath(path, os.path.dirname(os.path.join('artifacts', 'results_section.md')))
        return rel.replace('\\', '/')

    lines = []
    lines.append('# Results, Discussion, and Inference')
    lines.append('')
    lines.append(f'Generated on: {generated_at}')
    lines.append('')

    lines.append('## A. Experimental Results')
    lines.append('### Phase 1 - Preprocessing and Graph Construction')
    lines.append(f"- Train samples: {phase1['train_samples']:,}")
    lines.append(f"- Validation samples: {phase1['val_samples']:,}")
    lines.append(f"- Test samples: {phase1['test_samples']:,}")
    lines.append(f"- Train fraud rate: {phase1['train_fraud_rate']:.2f}%")
    lines.append(f"- Validation fraud rate: {phase1['val_fraud_rate']:.2f}%")
    lines.append(f"- Test fraud rate: {phase1['test_fraud_rate']:.2f}%")
    lines.append(f"- Graph edges (sampled): {phase1['edge_count']:,}")
    lines.append(f"- Graph nodes: {phase1['node_count']:,}")
    lines.append(f"- Feature dimension: {phase1['feature_dim']}")
    lines.append('')

    lines.append('### Phase 2 - Centralised Model')
    if phase2 is None:
        lines.append('- Metrics file not found. Run phase2_model.py to populate this section.')
    else:
        lines.append(f"- AUC-PR: {phase2['auc_pr']:.4f}")
        lines.append(f"- AUC-ROC: {phase2['auc_roc']:.4f}")
        lines.append(f"- F2 score: {phase2['f2']:.4f}")
        lines.append(f"- Recall: {phase2['recall']:.4f}")
        lines.append(f"- Precision: {phase2['precision']:.4f}")
        lines.append(f"- FPR: {phase2['fpr']:.4f}")
        lines.append(f"- Confusion matrix: TN={phase2['tn']} FP={phase2['fp']} FN={phase2['fn']} TP={phase2['tp']}")
    lines.append('')

    lines.append('### Phase 3 - Federated Simulation')
    if phase3_final is None:
        lines.append('- Metrics file not found. Run phase3_federated.py to populate this section.')
    else:
        lines.append(f"- Final round: {phase3_final['round']}")
        lines.append(f"- Metric source: {phase3_final.get('source', 'unknown')}")
        lines.append(f"- AUC-PR: {phase3_final['auc_pr']:.4f}")
        lines.append(f"- AUC-ROC: {phase3_final['auc_roc']:.4f}")
        lines.append(f"- F2 score: {phase3_final['f2']:.4f}")
        lines.append(f"- Recall: {phase3_final['recall']:.4f}")
        lines.append(f"- Precision: {phase3_final['precision']:.4f}")
        lines.append(f"- FPR: {phase3_final['fpr']:.4f}")
        lines.append(f"- Confusion matrix: TN={phase3_final['tn']} FP={phase3_final['fp']} FN={phase3_final['fn']} TP={phase3_final['tp']}")
    lines.append('')

    lines.append('### Generated Figures')
    if fig_paths.get('convergence'):
        lines.append(f"- Federated convergence: ![]({rel_to_results(fig_paths['convergence'])})")
    if fig_paths.get('comparison'):
        lines.append(f"- Centralised vs Federated: ![]({rel_to_results(fig_paths['comparison'])})")
    if fig_paths.get('psi'):
        lines.append(f"- PSI drift histogram: ![]({rel_to_results(fig_paths['psi'])})")
    if fig_paths.get('shap'):
        lines.append(f"- SHAP summary plot: ![]({rel_to_results(fig_paths['shap'])})")
    if not fig_paths:
        lines.append('- No figures were generated (missing plotting dependencies or artifacts).')
    lines.append('')

    lines.extend(section_discussion(phase2, phase3_final))
    lines.append('')

    lines.append('## C. Inference')
    lines.append('- The hybrid STGNN (GraphSAGE + GRU) is viable for fraud detection under centralized and privacy-preserving federated regimes when evaluated with a consistent benchmark.')
    lines.append('- The architecture addresses practical research gaps: privacy-preserving collaboration, concept-drift visibility through PSI, and transparent explanation via SHAP-style feature attribution.')
    lines.append('- For regulatory framing (DORA/EU AI Act), keep this report with confusion matrix, FPR, and explainability plots as model-risk evidence.')
    lines.append('')

    lines.append('## Notes for Paper Writing')
    lines.append('- Do not only report numbers; always explain causes (imbalance handling, weighting, drift, and non-IID client effects).')
    lines.append('- Prefer global holdout federated metrics for centralised-vs-federated comparison; keep client-weighted AUC-PR only as a training trend.')

    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description='Generate consolidated results + figures from artifacts.')
    parser.add_argument('--artifacts', default='artifacts', help='Artifacts directory path')
    parser.add_argument('--output', default=os.path.join('artifacts', 'results_section.md'), help='Output markdown file')
    args = parser.parse_args()

    artifacts_dir = args.artifacts
    preprocessed_path = os.path.join(artifacts_dir, 'preprocessed.pkl')
    centralised_path = os.path.join(artifacts_dir, 'centralised_metrics.pkl')
    fl_history_path = os.path.join(artifacts_dir, 'fl_history.pkl')
    federated_metrics_path = os.path.join(artifacts_dir, 'federated_metrics.pkl')
    psi_path = os.path.join(artifacts_dir, 'psi_details.pkl')
    round_csv_path = os.path.join(artifacts_dir, 'federated_round_metrics.csv')

    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError(f'Missing {preprocessed_path}. Run phase1_preprocessing.py first.')

    phase1 = summarize_phase1(preprocessed_path)
    phase2 = summarize_phase2(centralised_path)
    phase3_final, round_rows, _ = summarize_phase3(fl_history_path, federated_metrics_path)

    write_round_csv(round_rows, round_csv_path)

    fig_paths = make_figures(
        artifacts_dir=artifacts_dir,
        rows=round_rows,
        phase2=phase2,
        phase3_final=phase3_final,
        psi_path=psi_path,
        preprocessed_path=preprocessed_path,
    )

    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    markdown = build_markdown(phase1, phase2, phase3_final, generated_at, fig_paths)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(markdown)

    print(f'Saved: {args.output}')
    if round_rows:
        print(f'Saved: {round_csv_path}')
    for _, p in fig_paths.items():
        print(f'Saved: {p}')


if __name__ == '__main__':
    main()
