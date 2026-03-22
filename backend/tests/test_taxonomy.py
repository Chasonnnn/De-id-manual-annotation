from taxonomy import CANONICAL_LABELS, canonicalize_label


def test_canonical_labels_include_custom_project_extensions():
    assert "CUSTOMIZED_FIELD" in CANONICAL_LABELS
    assert "OTHER_LOCATIONS_IDENTIFIED" in CANONICAL_LABELS


def test_canonicalize_label_preserves_new_project_extensions():
    assert canonicalize_label("CUSTOMIZED_FIELD") == "CUSTOMIZED_FIELD"
    assert canonicalize_label("OTHER_LOCATIONS_IDENTIFIED") == "OTHER_LOCATIONS_IDENTIFIED"
