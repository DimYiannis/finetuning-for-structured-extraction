from data.schema import ExtractionResult
from src.evaluate import score, to_triples, validate


def _result(*triples: tuple[str, str, str]) -> ExtractionResult:
    return ExtractionResult(
        entities=[],
        relationships=[
            {"subject": s, "relation": r, "target": t} for s, r, t in triples
        ],
    )


GOLD = to_triples(
    _result(
        ("resolve", "CALLS", "validate"),
        ("LoRAResolver", "DEFINED_IN", "resolver"),
    )
)


def test_score_counts_overlap():
    pred = to_triples(
        _result(("resolve", "CALLS", "validate"), ("resolve", "CALLS", "print"))
    )
    assert score(pred, GOLD) == (1, 1, 1)


def test_invalid_output_misses_every_gold_triple():
    assert validate('{"entities": [{"name": "enc') is None  # truncated
    assert score(set(), GOLD) == (0, 0, 2)


def test_parsed_model_output_matches_hand_written_gold():
    # relation comes back as a RelationType enum; it must still match the
    # plain-string triples a hand-written reference produces
    raw = (
        '{"entities": [], "relationships": '
        '[{"subject": "resolve", "relation": "CALLS", "target": "validate"}]}'
    )
    assert to_triples(validate(raw)) == {("resolve", "CALLS", "validate")}
