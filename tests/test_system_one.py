from cape.system_one import evaluate_tasks


class StubAdapter:
    name = "stub"

    def judge(self, state, assertion):
        return float("supported" in state or "yes" in assertion)

    def judge_many(self, state, assertions):
        return [self.judge(state, assertion) for assertion in assertions]

    def choose(
        self,
        state,
        choices,
        assertion_template,
        minimum_probability=None,
    ):
        del state, assertion_template, minimum_probability
        scores = {choice: float(index == 1) for index, choice in enumerate(choices)}
        return choices[1], scores

    def score(self, state, levels):
        del state, levels
        raise NotImplementedError


def test_generic_harness_covers_boolean_choice_fanout_and_unsupported_score():
    tasks = [
        {
            "type": "boolean",
            "state": "supported fact",
            "assertion": "claim",
            "label": 1,
        },
        {
            "type": "choice",
            "state": "state",
            "choices": ["a", "b"],
            "answer": "b",
        },
        {
            "type": "fanout",
            "state": "state",
            "judgments": [
                {"assertion": "yes", "label": 1},
                {"assertion": "no", "label": 0},
            ],
        },
        {
            "type": "score",
            "state": "state",
            "levels": ["low", "high"],
            "answer_index": 1,
        },
    ]
    report = evaluate_tasks(StubAdapter(), tasks)
    assert report["boolean"]["accuracy"] == 1.0
    assert report["choice"]["accuracy"] == 1.0
    assert report["unsupported"] == 1
    assert report["malformed"] == 0
