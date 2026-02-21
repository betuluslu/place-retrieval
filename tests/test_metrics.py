from place_retrieval.metrics import average_precision, mean_average_precision, recall_at_k


def test_average_precision_single_positive():
    # One relevant item exists, ranked at position 2
    ranked = ["x", "A", "y", "z"]
    true = ["A"]

    ap = average_precision(ranked, true)
    # AP = precision at the hit position = 1/2
    assert abs(ap - 0.5) < 1e-9


def test_metrics_multi_positive_query():
    # Two relevant items: A and B
    ranked = ["A", "x", "B", "y"]
    true = ["A", "B"]

    ap = average_precision(ranked, true)
    # Hits at k=1 (P=1/1=1.0) and k=3 (P=2/3)
    # AP = (1.0 + 2/3) / 2 = 0.833333...
    assert abs(ap - (1.0 + 2.0 / 3.0) / 2.0) < 1e-9

    # Also test recall@1 and recall@2 behavior for multi-positive ground truth
    ranked_lists = [ranked]
    true_lists = [true]
    assert recall_at_k(ranked_lists, true_lists, k=1) == 1.0  # top1 hits A
    assert recall_at_k(ranked_lists, true_lists, k=2) == 1.0  # still hit