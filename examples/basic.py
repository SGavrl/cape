from cape import CAPE


cape = CAPE.from_checkpoint(
    "checkpoints/cape_mix_v1.pt"
)

print(cape.info())

context = (
    "The customer says their card was "
    "charged twice."
)

print(
    cape.judge(
        context,
        "This is a billing issue.",
    )
)

assertions = [
    "This is a billing issue.",
    "This is a networking issue.",
    "The customer experienced a duplicate charge.",
]

for assertion, probability in zip(
    assertions,
    cape.judge_many(
        context,
        assertions,
    ),
    strict=True,
):
    print(
        f"{probability:7.2%}  "
        f"{assertion}"
    )

choice = cape.choose(
    context,
    [
        "billing",
        "networking",
        "card delivery",
        "cash withdrawal",
    ],
    assertion_template=(
        "This customer request is about {choice}."
    ),
)

print()
print("choice:", choice.choice)
print(
    "probability:",
    f"{choice.probability:.2%}",
)
print("scores:", choice.scores)
