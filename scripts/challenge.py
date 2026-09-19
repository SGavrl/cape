import argparse

from cape import CAPE


CHECKPOINT = "checkpoints/cape_mix_v1.pt"


CASES = {
    "paraphrase_and_entailment": [
        (
            "A woman is swimming across the lake.",
            "A person is in the water.",
        ),
        (
            "A woman is swimming across the lake.",
            "Someone is swimming.",
        ),
        (
            "A woman is swimming across the lake.",
            "The woman is dry.",
        ),
        (
            "John purchased a vehicle yesterday.",
            "John bought something.",
        ),
        (
            "John purchased a vehicle yesterday.",
            "John bought a car.",
        ),
        (
            "The dog sprinted through the yard.",
            "An animal was moving quickly.",
        ),
    ],

    "negation": [
        (
            "The server did not restart.",
            "The server restarted.",
        ),
        (
            "The server did not restart.",
            "The server remained running.",
        ),
        (
            "No users reported an outage.",
            "Users reported an outage.",
        ),
        (
            "Not every request failed.",
            "Every request failed.",
        ),
        (
            "Not every request failed.",
            "At least one request did not fail.",
        ),
    ],

    "uncertainty": [
        (
            "The application became slower after several hours.",
            "The application definitely has a memory leak.",
        ),
        (
            "The application became slower after several hours.",
            "The application may have a performance problem.",
        ),
        (
            "CPU usage increased slightly after the deployment.",
            "The deployment caused a severe performance regression.",
        ),
        (
            "Three users reported intermittent errors.",
            "There may be a reliability problem.",
        ),
        (
            "Three users reported intermittent errors.",
            "The service is definitely broken for everyone.",
        ),
    ],

    "code_semantics": [
        (
            "The allocator retains references to objects after they are freed.",
            "This may cause a use-after-free bug.",
        ),
        (
            "The allocator retains references to objects after they are freed.",
            "This may create a memory safety issue.",
        ),
        (
            "A pointer is dereferenced after the memory it points to has been freed.",
            "This code may access invalid memory.",
        ),
        (
            "Two threads modify the same shared variable without synchronization.",
            "This code may contain a race condition.",
        ),
        (
            "Two threads modify the same shared variable without synchronization.",
            "This code concerns concurrency.",
        ),
        (
            "Every worker holds one lock while waiting for another worker's lock.",
            "The program may be deadlocked.",
        ),
        (
            "A loop performs a database query once for every item.",
            "This code may have a performance problem.",
        ),
        (
            "The patch only changes a comment.",
            "This change may alter runtime behavior.",
        ),
    ],

    "performance": [
        (
            "The patch adds several if statements.",
            "This change may affect performance.",
        ),
        (
            "The patch adds hundreds of branches inside a loop executed millions of times per second.",
            "This change may affect performance.",
        ),
        (
            "The patch adds validation checks to a settings page opened once per month.",
            "This change is likely to cause a major performance regression.",
        ),
        (
            "A function now performs an O(n^2) search instead of using a hash table.",
            "This change may reduce performance.",
        ),
        (
            "A frequently called function now sleeps for one second before returning.",
            "This change is likely to hurt performance.",
        ),
    ],

    "retrieval": [
        (
            "Question: How does TLS encrypt network traffic?\n"
            "Document: TLS establishes encrypted communication using cryptographic protocols.",
            "The document is relevant to the question.",
        ),
        (
            "Question: How does TLS encrypt network traffic?\n"
            "Document: TLS commonly uses certificates to authenticate endpoints.",
            "The document is relevant to the question.",
        ),
        (
            "Question: How does TLS encrypt network traffic?\n"
            "Document: Encryption protects information from unauthorized readers.",
            "The document is relevant to the question.",
        ),
        (
            "Question: How does TLS encrypt network traffic?\n"
            "Document: Basketball was invented in 1891.",
            "The document is relevant to the question.",
        ),
        (
            "Question: How does TLS encrypt network traffic?\n"
            "Document: Michael Jordan won six NBA championships.",
            "The document is relevant to the question.",
        ),
        (
            "Question: Why is the sky blue?\n"
            "Document: Rayleigh scattering affects shorter wavelengths more strongly than longer wavelengths.",
            "The document helps answer the question.",
        ),
    ],

    "causal_reasoning": [
        (
            "The database server lost power.",
            "The database may become unavailable.",
        ),
        (
            "The database server lost power.",
            "The database definitely lost all stored data.",
        ),
        (
            "The disk is completely full.",
            "New writes may fail.",
        ),
        (
            "The CPU temperature reached 105 degrees Celsius.",
            "The machine may have a thermal problem.",
        ),
        (
            "A cable was unplugged from the network switch.",
            "Network connectivity may be affected.",
        ),
    ],

    "temporal_reasoning": [
        (
            "Alice arrived before Bob. Bob arrived before Charlie.",
            "Alice arrived before Charlie.",
        ),
        (
            "Alice arrived before Bob. Bob arrived before Charlie.",
            "Charlie arrived before Alice.",
        ),
        (
            "The backup completed before the deployment started.",
            "The backup was completed when deployment began.",
        ),
        (
            "The service failed after the configuration change.",
            "The configuration change happened before the failure.",
        ),
    ],

    "coreference": [
        (
            "Sarah gave Emma the laptop because she no longer needed it.",
            "Sarah previously possessed the laptop.",
        ),
        (
            "Tom called James after he arrived home.",
            "James definitely arrived home.",
        ),
        (
            "The server rejected the client because it had an invalid certificate.",
            "The client may have had an invalid certificate.",
        ),
    ],

    "quantifiers": [
        (
            "Every server in the cluster is online.",
            "At least one server is online.",
        ),
        (
            "Some requests failed.",
            "Every request failed.",
        ),
        (
            "Some requests failed.",
            "At least one request failed.",
        ),
        (
            "No transactions were rejected.",
            "At least one transaction was rejected.",
        ),
    ],

    "numbers": [
        (
            "The old latency was 100 ms. The new latency is 200 ms.",
            "Latency increased.",
        ),
        (
            "The old latency was 100 ms. The new latency is 200 ms.",
            "Latency doubled.",
        ),
        (
            "The price fell from $100 to $80.",
            "The price decreased by 20 percent.",
        ),
        (
            "There are 20 failed requests out of 100 total requests.",
            "The failure rate is 20 percent.",
        ),
    ],

    "commonsense": [
        (
            "The glass fell onto concrete and shattered.",
            "The glass is broken.",
        ),
        (
            "The glass fell onto concrete and shattered.",
            "The glass is intact.",
        ),
        (
            "Heavy rain continued all night.",
            "The ground may be wet.",
        ),
        (
            "A person has not slept for two days.",
            "The person may be tired.",
        ),
        (
            "The freezer has been unplugged for three days.",
            "The food inside may no longer be frozen.",
        ),
    ],

    "weird_ood": [
        (
            "The customer says everything feels strange today.",
            "This is a networking issue.",
        ),
        (
            "The company announced a mysterious new product.",
            "The product uses artificial intelligence.",
        ),
        (
            "A developer added several new functions.",
            "The code is better.",
        ),
        (
            "The server emitted an unfamiliar warning.",
            "The server is compromised.",
        ),
        (
            "Revenue decreased this month.",
            "The company is going bankrupt.",
        ),
    ],
}


def main():
    parser = argparse.ArgumentParser(
        description="Run the frozen CAPE hard challenge."
    )
    parser.add_argument(
        "--checkpoint",
        default=CHECKPOINT,
    )
    args = parser.parse_args()

    cape = CAPE.from_checkpoint(
        args.checkpoint
    )

    print()
    print("CAPE-Mix-v1 hard challenge")
    print("=" * 100)
    print(cape.info())

    number = 1

    for category, cases in CASES.items():
        print()
        print()
        print("#" * 100)
        print(category.upper())
        print("#" * 100)
        print()

        for context, assertion in cases:
            probability = cape.judge(
                context,
                assertion,
            )

            print(
                f"[{number:03d}] "
                f"{probability:7.2%} | "
                f"{assertion}"
            )

            print(
                f"      {context}"
            )

            print()

            number += 1


if __name__ == "__main__":
    main()
