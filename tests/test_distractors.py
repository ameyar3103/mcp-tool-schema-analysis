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


def test_every_distractor_names_its_source():
    from hotset.corpus.distractors import pad_catalog
    from hotset.corpus.harvest import load

    base = load()
    real = {t.name for t in base}
    padded = pad_catalog(base, 200, seed=0)
    synthetic = [t for t in padded if t.synthetic]
    assert synthetic
    assert all(t.twin_of in real for t in synthetic)


def test_corpus_versions_differ_and_both_reproduce():
    from hotset.corpus.distractors import pad_catalog
    from hotset.corpus.harvest import load

    base = load()
    v1 = pad_catalog(base, 200, seed=0, version=1)
    v2 = pad_catalog(base, 200, seed=0, version=2)
    assert [t.description for t in v1] != [t.description for t in v2]
    assert [t.description for t in v1] == [
        t.description for t in pad_catalog(base, 200, seed=0, version=1)
    ]
