"""The replay behind docs/corpus.md: twin visibility, counted without calling a model."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from twin_exposure import exposure


def tool(name):
    return SimpleNamespace(name=name)


def session(*pairs):
    return SimpleNamespace(turns=[SimpleNamespace(user=u, tool=t) for u, t in pairs])


class Fixed:
    """A policy that always returns the same layers, so counting is the only variable."""

    def __init__(self, hot=(), tail=()):
        self.hot, self.tail = [tool(n) for n in hot], [tool(n) for n in tail]

    def plan(self, catalog, history, query):
        return SimpleNamespace(hot=self.hot, tail=self.tail)


TWINS = {"read_file": "read_file_alt"}


def test_only_turns_whose_target_has_a_twin_are_counted():
    """A target with no synthetic twin cannot be twin-confused, so it is not a trial."""
    s = [session(("a", "read_file"), ("b", "write_file"), ("c", "read_file"))]
    assert exposure(Fixed(), [], s, TWINS)["n"] == 2


def test_a_twin_in_the_tail_is_exposure_even_though_it_is_not_cached():
    """Layer C is ephemeral but still in the prompt; the model can pick from it."""
    s = [session(("a", "read_file"))]
    got = exposure(Fixed(tail=["read_file_alt"]), [], s, TWINS)
    assert (got["hot"], got["tail"], got["either"]) == (0, 1, 1)


def test_either_does_not_double_count_a_twin_in_both_layers():
    s = [session(("a", "read_file"))]
    got = exposure(Fixed(hot=["read_file_alt"], tail=["read_file_alt"]), [], s, TWINS)
    assert (got["hot"], got["tail"], got["either"]) == (1, 1, 1)


def test_the_target_is_tracked_separately_from_its_twin():
    """static-hot-set's immunity hypothesis rests on this pair being independent."""
    s = [session(("a", "read_file"))]
    got = exposure(Fixed(hot=["read_file"]), [], s, TWINS)
    assert (got["either"], got["target"]) == (0, 1)


def test_an_index_only_style_arm_shows_no_schemas_at_all():
    """Names-only arms expose neither, which is why exposure cannot explain their rate."""
    s = [session(("a", "read_file"))]
    got = exposure(Fixed(), [], s, TWINS)
    assert (got["either"], got["target"], got["n"]) == (0, 0, 1)
