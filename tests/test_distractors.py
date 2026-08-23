"""Distractor tests: the scale experiment is only credible if padding is plausible."""

from hotset.corpus.distractors import pad_catalog
from hotset.corpus.harvest import load

REAL = load()


def test_pads_to_target_sizes():
    for target in (150, 300, 500):
        assert len(pad_catalog(REAL, target)) == target


def test_names_stay_unique():
    padded = pad_catalog(REAL, 500)
    assert len({t.name for t in padded}) == len(padded)


def test_real_tools_are_preserved_unmodified():
    padded = pad_catalog(REAL, 300)
    assert [t for t in padded if not t.synthetic] == REAL


def test_seed_makes_corpus_reproducible():
    assert [t.name for t in pad_catalog(REAL, 300)] == [t.name for t in pad_catalog(REAL, 300)]
    assert [t.name for t in pad_catalog(REAL, 300)] != [
        t.name for t in pad_catalog(REAL, 300, seed=7)
    ]


def test_distractors_diverge_from_their_source():
    """Identical descriptions would leave no unique right answer for the labels."""
    real_desc = {t.description for t in REAL}
    synthetic = [t for t in pad_catalog(REAL, 500) if t.synthetic]
    assert all(t.description not in real_desc for t in synthetic)


def test_distractors_carry_realistic_token_weight():
    """A giveaway marker would collapse index cost and invalidate token accounting."""
    padded = pad_catalog(REAL, 500)
    synth = [t for t in padded if t.synthetic]
    avg = sum(len(t.description) for t in synth) / len(synth)
    assert avg > 60
