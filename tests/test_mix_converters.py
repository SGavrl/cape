from scripts.prepare_mix import (
    AG_ASSERTIONS,
    cape_example,
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
