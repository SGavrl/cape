from __future__ import annotations

import random


SPLIT_SEEDS = {
    "train": 120240,
    "validation": 120241,
    "test": 120242,
    "holdout": 120243,
}

ONTOLOGIES = {
    "train": [
        ("orchid", "flower", "plant"),
        ("salmon", "fish", "animal"),
        ("violin", "instrument", "artifact"),
        ("chemist", "scientist", "professional"),
    ],
    "validation": [
        ("sparrow", "bird", "animal"),
        ("harpsichord", "keyboard instrument", "instrument"),
    ],
    "test": [
        ("granite", "rock", "material"),
        ("falcon", "raptor", "bird"),
    ],
    "holdout": [
        ("sapphire", "gemstone", "mineral"),
        ("mangrove", "tree", "plant"),
    ],
}

LEXICAL_TEMPLATES = {
    "train": (
        "The inventory identifies {item} as a {kind}.",
        "The inventory supports that {item} is a {claim}.",
    ),
    "validation": (
        "According to the registry, {item} belongs to the class {kind}.",
        "The registry warrants classifying {item} as a {claim}.",
    ),
    "test": (
        "A field report categorized {item} as a {kind}.",
        "The report establishes that {item} is a {claim}.",
    ),
    "holdout": (
        "For specimen {item}, the recorded type is {kind}.",
        "The record justifies calling specimen {item} a {claim}.",
    ),
}

TEMPORAL_TEMPLATES = {
    "train": (
        "{a} concluded prior to {b}. {b} was followed by {c}.",
        "{first} occurred before {last}.",
    ),
    "validation": (
        "Before {b} began, {a} had already ended. Later, {c} started.",
        "{first} happened earlier than {last}.",
    ),
    "test": (
        "Once {a} was complete, {b} commenced; subsequently {c} began.",
        "{last} took place after {first}.",
    ),
    "holdout": (
        "By the time {b} started, {a} was over. {c} did not start until {b} ended.",
        "{first} preceded {last}.",
    ),
}

EVENTS = {
    "train": ("intake", "inspection", "packaging"),
    "validation": ("survey", "analysis", "publication"),
    "test": ("registration", "screening", "orientation"),
    "holdout": ("excavation", "cataloguing", "exhibition"),
}

NEGATION_CASES = {
    "train": [
        ("The parcel did not clear customs.", "The parcel cleared customs.", "contradiction"),
        ("The parcel did not clear customs.", "Customs confiscated the parcel.", "neutral"),
        ("The sensor did not activate.", "The sensor activated.", "contradiction"),
        ("The sensor did not activate.", "The sensor was defective.", "neutral"),
    ],
    "validation": [
        ("The permit was not approved.", "The permit was approved.", "contradiction"),
        ("The permit was not approved.", "The permit was denied.", "neutral"),
    ],
    "test": [
        ("The payment did not settle.", "The payment settled.", "contradiction"),
        ("The payment did not settle.", "The payment was fraudulent.", "neutral"),
    ],
    "holdout": [
        ("The alarm was not armed.", "The alarm was armed.", "contradiction"),
        ("The alarm was not armed.", "The alarm was broken.", "neutral"),
    ],
}

PERFORMANCE_CASES = {
    "train": [
        (
            "Each imported record is serialized three times before the next record is handled.",
            "The repeated serialization may reduce import throughput.",
            1.0,
            "possible",
        ),
        (
            "A computed lookup result is now reused for later requests with the same key.",
            "The reuse may reduce repeated computation.",
            1.0,
            "possible",
        ),
        (
            "A spelling correction changes only a disabled help message.",
            "The correction is likely to cause a major throughput regression.",
            0.0,
            "likely",
        ),
    ],
    "validation": [
        (
            "Every telemetry packet is compressed synchronously before transmission.",
            "Compression work may add processing latency per packet.",
            1.0,
            "possible",
        ),
        (
            "A connection pool now reuses established connections across operations.",
            "The change may avoid some connection setup work.",
            1.0,
            "possible",
        ),
    ],
    "test": [
        (
            "A regular expression is compiled again for each line of a large input.",
            "Repeated compilation may reduce processing throughput.",
            1.0,
            "possible",
        ),
        (
            "An index was added for the field used by the most frequent lookup.",
            "The index may improve that lookup's performance.",
            1.0,
            "possible",
        ),
        (
            "A one-time migration prints an additional status message.",
            "The message will certainly degrade request latency in production.",
            0.0,
            "certain",
        ),
    ],
    "holdout": [
        (
            "Every journal entry forces storage synchronization before the next entry is accepted.",
            "The synchronous journal writes may limit ingestion throughput.",
            1.0,
            "possible",
        ),
        (
            "An offline archival task now allocates one small metadata object per archive.",
            "The change will certainly cause a major regression in interactive requests.",
            0.0,
            "certain",
        ),
    ],
}

SYSTEMS_CASES = {
    "train": [
        (
            "The message queue has reached its configured byte limit.",
            "Publishing another message may be rejected until capacity is freed.",
            1.0,
            "possible",
        ),
        (
            "A process exhausted its permitted file descriptors.",
            "Opening another file may fail.",
            1.0,
            "possible",
        ),
        (
            "A cache entry expired.",
            "The underlying database was certainly destroyed.",
            0.0,
            "certain",
        ),
    ],
    "validation": [
        (
            "The service account lacks permission to read the requested object.",
            "The read may return an authorization error.",
            1.0,
            "possible",
        ),
        (
            "A retry budget was exhausted after repeated timeouts.",
            "The operation may stop retrying.",
            1.0,
            "possible",
        ),
    ],
    "test": [
        (
            "The receive buffer cannot accept additional bytes until its consumer catches up.",
            "The producer may experience backpressure.",
            1.0,
            "possible",
        ),
        (
            "A replica is temporarily unreachable while two other replicas remain healthy.",
            "All stored records were certainly erased.",
            0.0,
            "certain",
        ),
    ],
    "holdout": [
        (
            "The process reached its memory limit while requesting another allocation.",
            "The new allocation may fail.",
            1.0,
            "possible",
        ),
        (
            "One health check timed out during a network interruption.",
            "Every dependent service is certainly permanently unavailable.",
            0.0,
            "certain",
        ),
    ],
}

UNSUPPORTED_CASES = {
    "train": [
        ("The studio released two additional episodes.", "The new episodes are higher quality."),
        ("A clinic extended its opening hours.", "The clinic hired a particular physician."),
        ("The manufacturer changed the package color.", "The product now contains a new ingredient."),
    ],
    "validation": [
        ("The museum opened a second entrance.", "The museum is more popular than before."),
        ("A retailer revised its logo.", "The retailer intends to sell the company."),
    ],
    "test": [
        ("The publisher announced another book.", "The book will become a bestseller."),
        ("A laboratory installed new shelves.", "The laboratory discovered a new medicine."),
    ],
    "holdout": [
        ("The restaurant introduced a shorter menu.", "Its food now tastes better."),
        ("A school repainted its classrooms.", "Student examination scores increased."),
    ],
}

MODALITY_CASES = {
    "train": [
        ("Inspection found that vibration could loosen the fitting.", "The fitting may loosen.", 1.0, "possible"),
        ("Inspection found that vibration could loosen the fitting.", "The fitting certainly loosened.", 0.0, "certain"),
        ("Repeated trials showed the treatment usually succeeds.", "The treatment is likely to succeed.", 1.0, "likely"),
        ("Repeated trials showed the treatment usually succeeds.", "The treatment always succeeds.", 0.0, "certain"),
    ],
    "validation": [
        ("The forecast says frost is possible overnight.", "Frost may occur overnight.", 1.0, "possible"),
        ("The forecast says frost is possible overnight.", "Frost will definitely occur overnight.", 0.0, "certain"),
    ],
    "test": [
        ("The evidence makes contamination probable but not conclusive.", "Contamination is likely.", 1.0, "likely"),
        ("The evidence makes contamination probable but not conclusive.", "Contamination is certain.", 0.0, "certain"),
    ],
    "holdout": [
        ("The assessment identifies flooding as a possible outcome.", "Flooding could occur.", 1.0, "possible"),
        ("The assessment identifies flooding as a possible outcome.", "Flooding is guaranteed.", 0.0, "certain"),
    ],
}


def _row(split, capability, index, context, assertion, target, **labels):
    row = {
        "id": f"v1.2-{split}-{capability}-{index:06d}",
        "context": context,
        "assertion": assertion,
        "target": float(target),
        "source": "synthetic_v1_2",
        "capability": capability,
        "split_protocol": "template_ood" if split in {"test", "holdout"} else "iid",
    }
    row.update(labels)
    return row


def _lexical(split, index, rng):
    specific, middle, broad = ONTOLOGIES[split][index % len(ONTOLOGIES[split])]
    context_template, assertion_template = LEXICAL_TEMPLATES[split]
    item = f"item-{rng.randrange(10_000_000):07d}"
    supported = index % 2 == 0
    kind, claim = (specific, broad) if supported else (broad, specific)
    return _row(
        split,
        "lexical_directionality",
        index,
        context_template.format(item=item, kind=kind),
        assertion_template.format(item=item, claim=claim),
        float(supported),
        nli_label="entailment" if supported else "neutral",
    )


def _temporal(split, index, rng):
    a, b, c = EVENTS[split]
    context_template, assertion_template = TEMPORAL_TEMPLATES[split]
    supported = index % 2 == 0
    first, last = (a, c) if supported else (c, a)
    sequence = f"Sequence {rng.randrange(10_000_000):07d}: "
    return _row(
        split,
        "temporal",
        index,
        sequence + context_template.format(a=a, b=b, c=c),
        assertion_template.format(first=first, last=last),
        float(supported),
        nli_label="entailment" if supported else "contradiction",
    )


def _from_cases(split, capability, index, cases, rng, *, unsupported=False):
    case = cases[split][index % len(cases[split])]
    reference = f"Observation {rng.randrange(10_000_000):07d}. "
    if unsupported:
        context, assertion = case
        return _row(
            split,
            capability,
            index,
            reference + context,
            assertion,
            0.0,
            nli_label="neutral",
        )
    if capability == "negation":
        context, assertion, nli_label = case
        return _row(
            split,
            capability,
            index,
            reference + context,
            assertion,
            0.0,
            nli_label=nli_label,
        )
    context, assertion, target, strength = case
    return _row(
        split,
        capability,
        index,
        reference + context,
        assertion,
        target,
        nli_label="entailment" if target else "neutral",
        strength_label=strength,
    )


def _numeric(split, index, rng):
    start = rng.randint(40, 900)
    percent = rng.choice((10, 20, 25, 50))
    actual_increase = (index // 2) % 2 == 0
    supported = index % 2 == 0
    claimed_increase = actual_increase if supported else not actual_increase
    finish = start * (100 + percent if actual_increase else 100 - percent) / 100
    if split == "train":
        context = f"A reading changed from {start} units to {finish:g} units."
        assertion = f"The reading {'increased' if claimed_increase else 'decreased'} by {percent} percent."
    elif split == "validation":
        context = f"Initially the count was {start}; afterward it was {finish:g}."
        assertion = f"The later count is {percent} percent {'higher' if claimed_increase else 'lower'}."
    elif split == "test":
        context = f"The before and after measurements are {start} and {finish:g}, respectively."
        assertion = f"Relative to the first measurement, the second changed by {percent} percent {'upward' if claimed_increase else 'downward'}."
    else:
        context = f"Baseline: {start} points. Final observation: {finish:g} points."
        assertion = f"The final observation represents a {percent} percent {'gain' if claimed_increase else 'loss'} from baseline."
    return _row(
        split,
        "numeric",
        index,
        context,
        assertion,
        float(supported),
        nli_label="entailment" if supported else "contradiction",
    )


def _quantifier(split, index, rng):
    nouns = {
        "train": ("dossier", "stamped"),
        "validation": ("lantern", "lit"),
        "test": ("orchard", "irrigated"),
        "holdout": ("ballot", "verified"),
    }
    noun, property_word = nouns[split]
    group = f"group-{rng.randrange(10_000_000):07d}"
    exact = rng.randint(2, 7)
    cases = (
        (
            f"Every {noun} in {group} is {property_word}, and {group} contains at least one {noun}.",
            f"At least one {noun} in {group} is {property_word}.",
            1.0,
            "entailment",
        ),
        (
            f"Some {noun} in {group} is {property_word}.",
            f"Every {noun} in {group} is {property_word}.",
            0.0,
            "neutral",
        ),
        (
            f"No {noun} in {group} is {property_word}.",
            f"At least one {noun} in {group} is {property_word}.",
            0.0,
            "contradiction",
        ),
        (
            f"Not every {noun} in {group} is {property_word}.",
            f"At least one {noun} in {group} is not {property_word}.",
            1.0,
            "entailment",
        ),
        (
            f"Exactly {exact} {noun}s in {group} are {property_word}.",
            f"At least {exact} {noun}s in {group} are {property_word}.",
            1.0,
            "entailment",
        ),
        (
            f"Exactly {exact} {noun}s in {group} are {property_word}.",
            f"At least {exact + 1} {noun}s in {group} are {property_word}.",
            0.0,
            "contradiction",
        ),
    )
    context, assertion, target, nli_label = cases[index % len(cases)]
    return _row(
        split,
        "quantifiers",
        index,
        context,
        assertion,
        target,
        nli_label=nli_label,
    )


GENERATORS = {
    "lexical_directionality": _lexical,
    "temporal": _temporal,
    "numeric": _numeric,
    "quantifiers": _quantifier,
}


def generate_v1_2_synthetic(split, count_per_capability, seed=None):
    if split not in SPLIT_SEEDS:
        raise ValueError(f"unknown split: {split}")
    if count_per_capability < 0:
        raise ValueError("count_per_capability must be non-negative")
    rng = random.Random(SPLIT_SEEDS[split] if seed is None else seed)
    rows = []
    for capability, generator in GENERATORS.items():
        rows.extend(
            generator(split, index, rng)
            for index in range(count_per_capability)
        )
    case_groups = (
        ("negation", NEGATION_CASES, False),
        ("performance", PERFORMANCE_CASES, False),
        ("systems", SYSTEMS_CASES, False),
        ("unsupported_inference", UNSUPPORTED_CASES, True),
        ("modality", MODALITY_CASES, False),
    )
    for capability, cases, unsupported in case_groups:
        rows.extend(
            _from_cases(
                split,
                capability,
                index,
                cases,
                rng,
                unsupported=unsupported,
            )
            for index in range(count_per_capability)
        )
    rng.shuffle(rows)
    return rows
