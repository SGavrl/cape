from scripts.prepare_workflows import generate_workflow_tasks


def test_workflow_generation_is_deterministic_and_covers_domains():
    first = generate_workflow_tasks()
    second = generate_workflow_tasks()
    assert first == second
    assert {task["workflow"] for task in first} == {
        "customer_support",
        "code_review",
        "retrieval",
        "agent_trace",
        "incident",
    }
    assert all(task["type"] == "fanout" for task in first)
