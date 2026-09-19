from scripts.prepare_mix import (
    AG_ASSERTIONS,
    cape_example,
    convert_mnli,
)


def test_cape_example():
    example = cape_example(
        context="hello",
        assertion="This is a greeting.",
        label=1,
        source="test",
    )

    assert example == {
        "context": "hello",
        "assertion": "This is a greeting.",
        "label": 1,
        "source": "test",
    }


def test_ag_assertions_exist():
    assert set(
        AG_ASSERTIONS.keys()
    ) == {0, 1, 2, 3}


def test_labels_are_binary():
    positive = cape_example(
        "x",
        "y",
        1,
        "test",
    )

    negative = cape_example(
        "x",
        "z",
        0,
        "test",
    )

    assert positive["label"] == 1
    assert negative["label"] == 0


def test_mnli_conversion_maps_entailment_and_contradiction():
    rows = [
        {"premise": "p1", "hypothesis": "h1", "label": 0},
        {"premise": "p2", "hypothesis": "h2", "label": 1},
        {"premise": "p3", "hypothesis": "h3", "label": 2},
        {"premise": "p4", "hypothesis": "h4", "label": -1},
    ]

    assert convert_mnli(rows) == [
        {
            "context": "p1",
            "assertion": "h1",
            "label": 1,
            "source": "mnli",
        },
        {
            "context": "p3",
            "assertion": "h3",
            "label": 0,
            "source": "mnli",
        },
    ]
