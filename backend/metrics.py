from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from models import CanonicalSpan


def _overlap(a: CanonicalSpan, b: CanonicalSpan) -> int:
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def _iou(a: CanonicalSpan, b: CanonicalSpan) -> float:
    inter = _overlap(a, b)
    if inter == 0:
        return 0.0
    union = (a.end - a.start) + (b.end - b.start) - inter
    return inter / union if union > 0 else 0.0


def match_spans(
    gold: list[CanonicalSpan],
    pred: list[CanonicalSpan],
    mode: str = "exact",
    overlap_threshold: float = 0.5,
) -> tuple[
    list[tuple[CanonicalSpan, CanonicalSpan]], list[CanonicalSpan], list[CanonicalSpan]
]:
    """Return (matched_pairs, unmatched_gold, unmatched_pred).

    Uses the Hungarian algorithm for optimal 1:1 bipartite matching.
    """
    if not gold or not pred:
        return [], list(gold), list(pred)

    n_gold = len(gold)
    n_pred = len(pred)

    # Build cost matrix: higher score = better match, invert for minimization
    # Use -score so that linear_sum_assignment minimizes cost (= maximizes score)
    cost = np.full((n_gold, n_pred), 1e9)

    for i, g in enumerate(gold):
        for j, p in enumerate(pred):
            if mode == "exact":
                if g.start == p.start and g.end == p.end and g.label == p.label:
                    cost[i, j] = 0.0  # perfect match
            else:  # overlap
                if g.label == p.label:
                    iou = _iou(g, p)
                    if iou >= overlap_threshold:
                        cost[i, j] = 1.0 - iou  # lower cost = better match

    row_ind, col_ind = linear_sum_assignment(cost)

    matched: list[tuple[CanonicalSpan, CanonicalSpan]] = []
    matched_gold: set[int] = set()
    matched_pred: set[int] = set()

    for i, j in zip(row_ind, col_ind):
        if cost[i, j] < 1e9:  # actually a valid match
            matched.append((gold[i], pred[j]))
            matched_gold.add(i)
            matched_pred.add(j)

    unmatched_gold = [g for i, g in enumerate(gold) if i not in matched_gold]
    unmatched_pred = [p for j, p in enumerate(pred) if j not in matched_pred]
    return matched, unmatched_gold, unmatched_pred


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_metrics(
    gold: list[CanonicalSpan],
    pred: list[CanonicalSpan],
    mode: str = "exact",
    overlap_threshold: float = 0.5,
) -> dict:
    matched, unmatched_gold, unmatched_pred = match_spans(
        gold, pred, mode, overlap_threshold
    )

    tp = len(matched)
    fp = len(unmatched_pred)
    fn = len(unmatched_gold)

    micro = _prf(tp, fp, fn)

    # Per-label metrics
    all_labels = sorted(set(s.label for s in gold) | set(s.label for s in pred))
    per_label: dict[str, dict] = {}
    for label in all_labels:
        lg = [s for s in gold if s.label == label]
        lp = [s for s in pred if s.label == label]
        lm, lug, lup = match_spans(lg, lp, mode, overlap_threshold)
        metrics = _prf(len(lm), len(lup), len(lug))
        metrics["support"] = len(lg)
        per_label[label] = metrics

    # Macro averages
    if per_label:
        macro_p = sum(v["precision"] for v in per_label.values()) / len(per_label)
        macro_r = sum(v["recall"] for v in per_label.values()) / len(per_label)
        macro_f1 = sum(v["f1"] for v in per_label.values()) / len(per_label)
    else:
        macro_p = macro_r = macro_f1 = 0.0

    # Confusion matrix
    confusion = _build_confusion(
        gold, pred, matched, unmatched_gold, unmatched_pred, all_labels
    )
    confusion_matrix = _confusion_to_grid(confusion)

    # Cohen's kappa (character-level)
    kappa = _cohens_kappa_spans(
        gold,
        pred,
        max(
            max((s.end for s in gold), default=0),
            max((s.end for s in pred), default=0),
        ),
    )

    # Mean IoU for matched spans
    if matched:
        mean_iou = sum(_iou(g, p) for g, p in matched) / len(matched)
    else:
        mean_iou = 0.0

    return {
        "micro": micro,
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "per_label": per_label,
        "confusion_matrix": confusion_matrix,
        "false_positives": unmatched_pred,
        "false_negatives": unmatched_gold,
        # Backward-compatible alias.
        "confusion": confusion,
        "cohens_kappa": kappa,
        "mean_iou": mean_iou,
    }


def _build_confusion(
    gold: list[CanonicalSpan],
    pred: list[CanonicalSpan],
    matched: list[tuple[CanonicalSpan, CanonicalSpan]],
    unmatched_gold: list[CanonicalSpan],
    unmatched_pred: list[CanonicalSpan],
    labels: list[str],
) -> dict:
    all_labels = labels + ["O"]
    matrix: dict[str, dict[str, int]] = {
        g: {p: 0 for p in all_labels} for g in all_labels
    }
    for g, p in matched:
        matrix[g.label][p.label] += 1
    for g in unmatched_gold:
        matrix[g.label]["O"] += 1
    for p in unmatched_pred:
        matrix["O"][p.label] += 1
    return {"labels": all_labels, "matrix": matrix}


def _confusion_to_grid(confusion: dict) -> dict:
    labels: list[str] = confusion["labels"]
    matrix_dict: dict[str, dict[str, int]] = confusion["matrix"]
    grid = [
        [matrix_dict[row_label][col_label] for col_label in labels]
        for row_label in labels
    ]
    return {"labels": labels, "matrix": grid}


def _cohens_kappa_spans(
    gold: list[CanonicalSpan], pred: list[CanonicalSpan], text_len: int
) -> float:
    """Compute Cohen's kappa at character level."""
    if text_len == 0:
        return 1.0 if not gold and not pred else 0.0

    # Build character-level label arrays
    gold_labels = ["O"] * text_len
    pred_labels = ["O"] * text_len
    for s in gold:
        for i in range(s.start, min(s.end, text_len)):
            gold_labels[i] = s.label
    for s in pred:
        for i in range(s.start, min(s.end, text_len)):
            pred_labels[i] = s.label

    all_labels = sorted(set(gold_labels) | set(pred_labels))
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    n = len(all_labels)

    conf = [[0] * n for _ in range(n)]
    for g, p in zip(gold_labels, pred_labels):
        conf[label_to_idx[g]][label_to_idx[p]] += 1

    total = text_len
    po = sum(conf[i][i] for i in range(n)) / total
    pe = sum(
        sum(conf[i][j] for j in range(n)) * sum(conf[j][i] for j in range(n))
        for i in range(n)
    ) / (total * total)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)
