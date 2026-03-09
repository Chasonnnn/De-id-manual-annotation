import pytest
from models import CanonicalSpan
from metrics import compute_metrics, match_spans, _iou


def _span(start, end, label="NAME", text="x"):
    return CanonicalSpan(start=start, end=end, label=label, text=text)


# ---------------------------------------------------------------------------
# Exact matching
# ---------------------------------------------------------------------------


class TestExactMatch:
    def test_perfect_match(self):
        gold = [_span(0, 5, "NAME"), _span(10, 15, "EMAIL")]
        pred = [_span(0, 5, "NAME"), _span(10, 15, "EMAIL")]
        m = compute_metrics(gold, pred, mode="exact")
        assert m["micro"]["precision"] == 1.0
        assert m["micro"]["recall"] == 1.0
        assert m["micro"]["f1"] == 1.0

    def test_no_match_different_offsets(self):
        gold = [_span(0, 5, "NAME")]
        pred = [_span(6, 10, "NAME")]
        m = compute_metrics(gold, pred, mode="exact")
        assert m["micro"]["precision"] == 0.0
        assert m["micro"]["recall"] == 0.0

    def test_no_match_different_labels(self):
        gold = [_span(0, 5, "NAME")]
        pred = [_span(0, 5, "EMAIL")]
        m = compute_metrics(gold, pred, mode="exact")
        assert m["micro"]["tp"] == 0
        assert m["micro"]["fp"] == 1
        assert m["micro"]["fn"] == 1

    def test_partial_match(self):
        gold = [_span(0, 5, "NAME"), _span(10, 15, "EMAIL")]
        pred = [_span(0, 5, "NAME")]
        m = compute_metrics(gold, pred, mode="exact")
        assert m["micro"]["tp"] == 1
        assert m["micro"]["fn"] == 1
        assert m["micro"]["fp"] == 0
        assert m["micro"]["recall"] == 0.5
        assert m["micro"]["precision"] == 1.0


class TestBoundaryMatch:
    def test_trims_trailing_space_and_period(self):
        gold = [_span(10, 15, "NAME", "David"), _span(30, 33, "NAME", "Ana")]
        pred = [_span(10, 16, "NAME", "David "), _span(30, 34, "NAME", "Ana.")]
        m = compute_metrics(gold, pred, mode="boundary")
        assert m["micro"]["precision"] == 1.0
        assert m["micro"]["recall"] == 1.0
        assert m["micro"]["f1"] == 1.0

    def test_does_not_trim_internal_punctuation(self):
        gold = [_span(0, 8, "NAME", "St. John")]
        pred = [_span(0, 7, "NAME", "St John")]
        m = compute_metrics(gold, pred, mode="boundary")
        assert m["micro"]["tp"] == 0
        assert m["micro"]["fp"] == 1
        assert m["micro"]["fn"] == 1


class TestExactNameAffixTolerant:
    def test_co_primary_matches_supported_name_boundary_affixes(self):
        gold = [
            _span(0, 12, "NAME", "Mr. Muhammad"),
            _span(20, 26, "NAME", "David,"),
            _span(30, 34, "NAME", "Ana."),
            _span(40, 47, "NAME", "Javier "),
            _span(50, 61, "NAME", "Sebastian's"),
        ]
        pred = [
            _span(4, 12, "NAME", "Muhammad"),
            _span(20, 25, "NAME", "David"),
            _span(30, 33, "NAME", "Ana"),
            _span(40, 46, "NAME", "Javier"),
            _span(50, 59, "NAME", "Sebastian"),
        ]

        m = compute_metrics(gold, pred, mode="exact")

        assert m["micro"]["f1"] == 0.0
        tolerant = m["co_primary_metrics"]["exact_name_affix_tolerant"]
        assert tolerant["micro"]["precision"] == 1.0
        assert tolerant["micro"]["recall"] == 1.0
        assert tolerant["micro"]["f1"] == 1.0

    def test_co_primary_does_not_allow_multi_token_name_to_single_token(self):
        gold = [_span(0, 14, "NAME", "Michael Myers")]
        pred = [_span(0, 7, "NAME", "Michael")]

        m = compute_metrics(gold, pred, mode="exact")

        tolerant = m["co_primary_metrics"]["exact_name_affix_tolerant"]
        assert tolerant["micro"]["tp"] == 0
        assert tolerant["micro"]["fp"] == 1
        assert tolerant["micro"]["fn"] == 1

    def test_co_primary_keeps_non_name_labels_exact(self):
        gold = [_span(0, 6, "URL", "abc.com")]
        pred = [_span(0, 5, "URL", "abc.c")]

        m = compute_metrics(gold, pred, mode="exact")

        tolerant = m["co_primary_metrics"]["exact_name_affix_tolerant"]
        assert tolerant["micro"]["tp"] == 0
        assert tolerant["micro"]["fp"] == 1
        assert tolerant["micro"]["fn"] == 1


# ---------------------------------------------------------------------------
# Overlap (bipartite) matching
# ---------------------------------------------------------------------------


class TestOverlapMatch:
    def test_basic_overlap(self):
        gold = [_span(0, 10, "NAME")]
        pred = [_span(2, 10, "NAME")]
        matched, ug, up = match_spans(gold, pred, mode="overlap", overlap_threshold=0.5)
        assert len(matched) == 1

    def test_overlap_below_threshold(self):
        """Spans overlap by 1 char out of 10 -- IoU ~0.05, below 0.5 threshold."""
        gold = [_span(0, 10, "NAME")]
        pred = [_span(9, 19, "NAME")]
        matched, ug, up = match_spans(gold, pred, mode="overlap", overlap_threshold=0.5)
        assert len(matched) == 0
        assert len(ug) == 1
        assert len(up) == 1

    def test_bipartite_optimal_matching(self):
        """Hungarian algorithm should find optimal 1:1 matching.

        Gold: A=[0,10], B=[10,20]
        Pred: X=[0,10], Y=[8,20]

        Greedy (left-to-right): A->X, B->Y (both match)
        But what if pred order is reversed? Y=[8,20] could greedily match A.
        The Hungarian algorithm should still find A->X, B->Y as optimal.
        """
        gold = [_span(0, 10, "NAME"), _span(10, 20, "NAME")]
        pred = [_span(8, 20, "NAME"), _span(0, 10, "NAME")]  # reversed order
        matched, ug, up = match_spans(gold, pred, mode="overlap", overlap_threshold=0.1)
        assert len(matched) == 2
        # The optimal matching should pair gold[0] with pred[1] (both [0,10])
        # and gold[1] with pred[0] (both ~[10,20])

    def test_different_labels_no_match(self):
        gold = [_span(0, 10, "NAME")]
        pred = [_span(0, 10, "EMAIL")]
        matched, ug, up = match_spans(gold, pred, mode="overlap")
        assert len(matched) == 0


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------


class TestIoU:
    def test_perfect_iou(self):
        assert _iou(_span(0, 10), _span(0, 10)) == 1.0

    def test_no_overlap_iou(self):
        assert _iou(_span(0, 5), _span(10, 15)) == 0.0

    def test_partial_iou(self):
        # [0,10] and [5,15]: overlap=5, union=15
        iou = _iou(_span(0, 10), _span(5, 15))
        assert abs(iou - 5 / 15) < 1e-9

    def test_contained_iou(self):
        # [2,8] inside [0,10]: overlap=6, union=10
        iou = _iou(_span(0, 10), _span(2, 8))
        assert abs(iou - 6 / 10) < 1e-9


# ---------------------------------------------------------------------------
# Per-label metrics
# ---------------------------------------------------------------------------


class TestPerLabel:
    def test_per_label_breakdown(self):
        gold = [_span(0, 5, "NAME"), _span(10, 15, "EMAIL")]
        pred = [_span(0, 5, "NAME")]  # Missing EMAIL
        m = compute_metrics(gold, pred, mode="exact")
        assert m["per_label"]["NAME"]["f1"] == 1.0
        assert m["per_label"]["EMAIL"]["recall"] == 0.0
        assert m["per_label"]["EMAIL"]["precision"] == 0.0

    def test_per_label_false_positive(self):
        gold = [_span(0, 5, "NAME")]
        pred = [_span(0, 5, "NAME"), _span(10, 15, "PHONE")]
        m = compute_metrics(gold, pred, mode="exact")
        assert m["per_label"]["NAME"]["f1"] == 1.0
        assert m["per_label"]["PHONE"]["tp"] == 0
        assert m["per_label"]["PHONE"]["fp"] == 1


# ---------------------------------------------------------------------------
# Macro averages
# ---------------------------------------------------------------------------


class TestMacro:
    def test_macro_average(self):
        gold = [_span(0, 5, "NAME"), _span(10, 15, "EMAIL")]
        pred = [_span(0, 5, "NAME")]
        m = compute_metrics(gold, pred, mode="exact")
        # NAME: P=1, R=1, F1=1. EMAIL: P=0, R=0, F1=0
        # Macro: P=0.5, R=0.5, F1=0.5
        assert abs(m["macro"]["precision"] - 0.5) < 1e-9
        assert abs(m["macro"]["recall"] - 0.5) < 1e-9
        assert abs(m["macro"]["f1"] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


class TestConfusion:
    def test_confusion_perfect(self):
        gold = [_span(0, 5, "NAME")]
        pred = [_span(0, 5, "NAME")]
        m = compute_metrics(gold, pred, mode="exact")
        cm = m["confusion"]
        cm_grid = m["confusion_matrix"]
        assert "NAME" in cm["labels"]
        assert "O" in cm["labels"]
        assert cm["matrix"]["NAME"]["NAME"] == 1
        assert cm["matrix"]["NAME"]["O"] == 0
        assert len(cm_grid["labels"]) == len(cm_grid["matrix"])

    def test_confusion_mismatch(self):
        gold = [_span(0, 5, "NAME")]
        pred = [_span(10, 15, "EMAIL")]
        m = compute_metrics(gold, pred, mode="exact")
        cm = m["confusion"]
        assert cm["matrix"]["NAME"]["O"] == 1  # gold NAME unmatched
        assert cm["matrix"]["O"]["EMAIL"] == 1  # pred EMAIL unmatched

    def test_confusion_label_swap(self):
        """Matched by position in overlap mode but labels differ -- not matched."""
        gold = [_span(0, 10, "NAME")]
        pred = [_span(0, 10, "EMAIL")]
        m = compute_metrics(gold, pred, mode="overlap")
        cm = m["confusion"]
        # Different labels means no match
        assert cm["matrix"]["NAME"]["O"] == 1
        assert cm["matrix"]["O"]["EMAIL"] == 1


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------


class TestCohensKappa:
    def test_perfect_agreement(self):
        gold = [_span(0, 5, "NAME")]
        pred = [_span(0, 5, "NAME")]
        m = compute_metrics(gold, pred, mode="exact")
        assert m["cohens_kappa"] == 1.0

    def test_no_agreement(self):
        gold = [_span(0, 5, "NAME")]
        pred = [_span(5, 10, "EMAIL")]
        m = compute_metrics(gold, pred, mode="exact")
        assert m["cohens_kappa"] < 0.5

    def test_partial_agreement(self):
        gold = [_span(0, 5, "NAME"), _span(10, 15, "EMAIL")]
        pred = [_span(0, 5, "NAME"), _span(10, 15, "PHONE")]
        m = compute_metrics(gold, pred, mode="exact")
        # NAME chars match, but EMAIL/PHONE chars disagree
        kappa = m["cohens_kappa"]
        assert 0.0 < kappa < 1.0


# ---------------------------------------------------------------------------
# Mean IoU
# ---------------------------------------------------------------------------


class TestMeanIoU:
    def test_perfect_iou_in_metrics(self):
        gold = [_span(0, 10, "NAME")]
        pred = [_span(0, 10, "NAME")]
        m = compute_metrics(gold, pred, mode="exact")
        assert m["mean_iou"] == 1.0

    def test_partial_iou_in_overlap_mode(self):
        gold = [_span(0, 10, "NAME")]
        pred = [_span(3, 10, "NAME")]
        m = compute_metrics(gold, pred, mode="overlap", overlap_threshold=0.3)
        # IoU = 7 / 10 = 0.7
        assert abs(m["mean_iou"] - 0.7) < 1e-9

    def test_no_matches_zero_iou(self):
        gold = [_span(0, 5, "NAME")]
        pred = [_span(20, 25, "EMAIL")]
        m = compute_metrics(gold, pred, mode="exact")
        assert m["mean_iou"] == 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_spans(self):
        m = compute_metrics([], [], mode="exact")
        assert m["micro"]["tp"] == 0
        assert m["micro"]["fp"] == 0
        assert m["micro"]["fn"] == 0

    def test_empty_gold_with_pred(self):
        m = compute_metrics([], [_span(0, 5, "NAME")], mode="exact")
        assert m["micro"]["tp"] == 0
        assert m["micro"]["fp"] == 1
        assert m["micro"]["precision"] == 0.0

    def test_empty_pred_with_gold(self):
        m = compute_metrics([_span(0, 5, "NAME")], [], mode="exact")
        assert m["micro"]["tp"] == 0
        assert m["micro"]["fn"] == 1
        assert m["micro"]["recall"] == 0.0

    def test_single_char_span(self):
        gold = [_span(5, 6, "NAME")]
        pred = [_span(5, 6, "NAME")]
        matched, ug, up = match_spans(gold, pred, mode="exact")
        assert len(matched) == 1

    def test_false_positive_and_negative_lists_present(self):
        gold = [_span(0, 5, "NAME")]
        pred = [_span(10, 15, "EMAIL")]
        m = compute_metrics(gold, pred, mode="exact")
        assert len(m["false_positives"]) == 1
        assert len(m["false_negatives"]) == 1
