import pytest

from sam3_masking.prompts import parse_prompt_catalog


def test_prompt_catalog_builds_canonical_categories_in_query_order():
    catalog = parse_prompt_catalog(
        " car, truck, traffic sign ",
        "vehicle:car,truck;sign:traffic sign",
    )

    assert catalog.prompts == ("car", "truck", "traffic sign")
    assert catalog.synonym_to_canonical == {
        "car": "vehicle",
        "truck": "vehicle",
        "traffic sign": "sign",
    }
    assert catalog.categories == ("vehicle", "sign")
    assert catalog.normalized_synonyms() == {
        "vehicle": ["car", "truck"],
        "sign": ["traffic sign"],
    }


@pytest.mark.parametrize(
    ("prompts", "synonyms", "message"),
    [
        ("car,Car", "", "duplicate prompt"),
        ("car", "vehicle:truck", "not present"),
        ("car", "vehicle:car;automobile:car", "belongs to both"),
        ("car", "vehicle", "expected canonical"),
        ("car", "vehicle:", "members must be nonempty"),
    ],
)
def test_prompt_catalog_rejects_ambiguous_specs(prompts, synonyms, message):
    with pytest.raises(ValueError, match=message):
        parse_prompt_catalog(prompts, synonyms)
