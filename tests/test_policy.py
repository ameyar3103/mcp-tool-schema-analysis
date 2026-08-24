"""Policy tests: retrieval quality and the layer shapes each baseline asks for."""

from hotset.corpus.harvest import load
from hotset.layout.prompt import assemble
from hotset.policy.baselines import FullCatalog, LazyDiscovery, RagOverTools, StaticHotSet
from hotset.policy.retrieval import BM25, terms

CATALOG = load()


def test_terms_split_snake_and_camel():
    """Tool names are identifiers; queries are prose. Retrieval fails unless they meet."""
    assert terms("get_current_time") == ["get", "current", "time"]
    assert terms("browserTakeScreenshot") == ["browser", "take", "screenshot"]


def test_bm25_ranks_the_obvious_tool_first():
    """A strawman retriever would make baseline 2 meaningless."""
    bm = BM25(CATALOG)
    assert bm.top_k("what time is it in Tokyo", 1)[0].name == "get_current_time"
    assert bm.top_k("click a button on a web page", 1)[0].name == "browser_click"


def test_bm25_is_order_independent():
    """Catalog order must not move results, or arms differ for a bookkeeping reason."""
    a = BM25(CATALOG).top_k("read a file", 5)
    b = BM25(list(reversed(CATALOG))).top_k("read a file", 5)
    assert [t.name for t in a] == [t.name for t in b]


def test_baseline_layer_shapes():
    """Each arm is defined by which layers it populates."""
    q = "read a file"
    assert len(FullCatalog().plan(CATALOG, [], q).hot) == len(CATALOG)
    assert len(RagOverTools(k=8).plan(CATALOG, [], q).hot) == 8
    lazy = LazyDiscovery().plan(CATALOG, [], q)
    assert not lazy.hot and not lazy.index and len(lazy.extra_tools) == 1
    assert len(StaticHotSet(CATALOG[:8]).plan(CATALOG, [], q).index) == len(CATALOG)


def test_lazy_discovery_serves_its_own_tool():
    """Baseline 3's registry is answered by the policy, not the environment."""
    lazy = LazyDiscovery()
    assert lazy.serves("search_tools") and not lazy.serves("read_file")
    assert "get_current_time" in lazy.serve(CATALOG, {"query": "current time", "limit": 3})


def test_only_lazy_discovery_adds_native_tools():
    """Extra tools change the uncached upstream field, so no other arm may add any."""
    for arm in (FullCatalog(), RagOverTools(), StaticHotSet(CATALOG[:4])):
        assert len(assemble(arm.plan(CATALOG, [], "x"), [])["tools"]) == 1
