"""Tier 1 retrieval-quality improvements (v0.4.0).

Covers: stemming, CamelCase expansion, CJK bigrams, enriched corpus with
field boosts (name/tags/category), env-configurable k1/b, relevance floor,
chit-chat skip, and word-boundary description truncation.
"""

import importlib
import sys
from pathlib import Path

import pytest

import bm25_retriever as br

PLUGIN_DIR = Path(__file__).resolve().parent.parent


def _load_plugin_module():
    """Import the plugin package __init__ as a module (no Hermes required)."""
    spec = importlib.util.spec_from_file_location(
        "skill_retrieval_plugin_t1", PLUGIN_DIR / "__init__.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── Stemming ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("word,stem", [
    ("skills", "skill"),
    ("searching", "search"),
    ("searches", "search"),
    ("optimizing", "optimiz"),
    ("optimized", "optimiz"),
    ("studies", "study"),
    ("plugins", "plugin"),
])
def test_stem_collapses_variants(word, stem):
    assert br._stem(word) == stem


@pytest.mark.parametrize("word", ["bm25", "git", "docs", "yes", "is", "html5"])
def test_stem_leaves_short_or_alnum_tokens(word):
    # < 5 chars or non-alpha tokens are never stemmed
    assert br._stem(word) == word


def test_stemming_matches_morphological_variants():
    index = br.BM25Index()
    index.build(
        ["search", "travel", "image"],
        ["multi engine web searching", "plan trips flights", "generate pictures"],
    )
    assert index.retrieve("searches", top_k=3)[0][0] == "search"


# ─── CamelCase ───────────────────────────────────────────────────────────────

def test_camel_expansion_keeps_original_and_adds_parts():
    toks = br._normalize_tokens("skillView")
    assert "skillview" in toks
    assert "skill" in toks and "view" in toks


def test_camel_expansion_handles_acronyms():
    toks = br._normalize_tokens("HTMLParser")
    assert "html" in toks and "parser" in toks


def test_camel_expansion_skips_short_fragments():
    # "widgetB" must NOT emit a bare "b" token (would match unrelated docs)
    toks = br._normalize_tokens("widgetB")
    assert toks == ["widgetb"]


def test_camel_query_matches_split_doc():
    index = br.BM25Index()
    index.build(
        ["sv", "x", "y"],
        ["skill view loader", "calendar planner", "travel flights"],
    )
    assert index.retrieve("skillView", top_k=3)[0][0] == "sv"


# ─── CJK ─────────────────────────────────────────────────────────────────────

def test_cjk_emits_run_and_bigrams():
    toks = br._normalize_tokens("图片生成")
    assert "图片生成" in toks
    assert {"图片", "片生", "生成"} <= set(toks)


def test_cjk_mixed_with_latin():
    toks = br._normalize_tokens("用Python生成图片")
    assert "python" in toks
    assert "生成" in toks and "图片" in toks


def test_cjk_query_matches_partial_phrase():
    index = br.BM25Index()
    index.build(
        ["img", "trip", "code"],
        ["图片生成 image generation", "旅行计划 travel planning", "代码审查 code review"],
    )
    # Query shares only the bigram 图片 with the doc's run 图片生成
    assert index.retrieve("帮我做图片", top_k=3)[0][0] == "img"


def test_non_cjk_text_is_untouched_by_cjk_split():
    assert br._split_cjk("plain ascii text") == "plain ascii text"


# ─── Enriched corpus / field boosts ──────────────────────────────────────────

def test_build_index_text_boosts_name_and_tags():
    text = br._build_index_text("web-search", "Find pages.", ["exa", "brave"], "research")
    assert text.count("web-search") == 3
    assert text.count("exa") == 2
    assert "research" in text
    assert text.endswith("Find pages.")


def test_build_index_text_skips_general_category():
    assert "general" not in br._build_index_text("n", "d", [], "general")


@pytest.mark.parametrize("raw,expected", [
    (None, []),
    ("", []),
    ("a, b; c", ["a", "b", "c"]),
    (["x", None, " y ", ""], ["x", "y"]),
    (("t",), ["t"]),
])
def test_normalize_tags(raw, expected):
    assert br._normalize_tags(raw) == expected


def test_parse_skill_md_full_reads_tags(tmp_path):
    md = tmp_path / "SKILL.md"
    md.write_text("---\nname: s\ndescription: d\ntags: [bm25, retrieval]\n---\n")
    assert br._parse_skill_md_full(md) == ("s", "d", ["bm25", "retrieval"])
    # Back-compat 2-tuple API unchanged
    assert br._parse_skill_md(md) == ("s", "d")


def test_tags_make_skill_retrievable(tmp_path, monkeypatch):
    """A skill whose description lacks the query term is found via its tags."""
    skills_root = tmp_path / "skills"
    for rel, name, desc, tags in [
        ("a/fusion", "fusion", "Merge ranked lists.", "[bm25, reranking]"),
        ("b/trip", "trip", "Plan travel itineraries.", "[flights]"),
        ("c/pics", "pics", "Make pictures.", "[images]"),
    ]:
        d = skills_root / rel
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {desc}\ntags: {tags}\n---\n"
        )
    (tmp_path / "plugins").mkdir()
    cfg = tmp_path / "config.yaml"
    cfg.write_text("skills:\n  disabled: []\n")
    monkeypatch.setattr(br, "SKILLS_ROOT", skills_root)
    monkeypatch.setattr(br, "PLUGINS_ROOT", tmp_path / "plugins")
    monkeypatch.setattr(br, "CONFIG_PATH", cfg)
    br._index = None
    br._skills_by_id = {}
    try:
        index = br.get_index()
        assert index.retrieve("reranking", top_k=3)[0][0] == "a/fusion"
        assert br.get_skill_info("a/fusion")["tags"] == ["bm25", "reranking"]
    finally:
        br._index = None
        br._skills_by_id = {}


def test_record_skill_includes_category_and_tags():
    skills, seen = [], set()
    br._record_skill(
        skills, seen,
        {"category": "devops", "skill_name": "s", "frontmatter_name": "s", "description": "d"},
        tags=["docker"],
    )
    assert skills[0]["tags"] == ["docker"]
    assert "devops" in skills[0]["text"]
    assert skills[0]["text"].count("docker") == 2


# ─── Env-configurable k1 / b ─────────────────────────────────────────────────

def test_env_float_parsing(monkeypatch):
    monkeypatch.setenv("SR_TEST_F", "1.2")
    assert br._env_float("SR_TEST_F", 9.0) == 1.2
    for bad in ("abc", "-1", "0", ""):
        monkeypatch.setenv("SR_TEST_F", bad)
        assert br._env_float("SR_TEST_F", 9.0) == 9.0
    monkeypatch.delenv("SR_TEST_F")
    assert br._env_float("SR_TEST_F", 9.0) == 9.0


# ─── Relevance floor ─────────────────────────────────────────────────────────

def test_relevance_floor_drops_weak_tail():
    mod = _load_plugin_module()
    results = [("a", 10.0), ("b", 5.0), ("c", 1.4), ("d", 0.5)]
    assert mod._apply_relevance_floor(results, 0.15) == [("a", 10.0), ("b", 5.0)]


def test_relevance_floor_keeps_top1_and_handles_edge_cases():
    mod = _load_plugin_module()
    assert mod._apply_relevance_floor([("a", 0.01)], 0.5) == [("a", 0.01)]
    assert mod._apply_relevance_floor([], 0.5) == []
    r = [("a", 3.0), ("b", 0.1)]
    assert mod._apply_relevance_floor(r, 0.0) == r


@pytest.mark.parametrize("raw,expected", [
    (None, 0.15), ("", 0.15), ("0.3", 0.3), ("0", 0.0),
    ("1.0", 0.15), ("-0.1", 0.15), ("junk", 0.15),
])
def test_parse_min_score_ratio(raw, expected):
    mod = _load_plugin_module()
    assert mod._parse_min_score_ratio(raw) == expected


# ─── Chit-chat skip ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("msg", [
    "thanks", "Thanks!", "ok", "OK.", "yes", "got it", "👍", "  ", "", "cool!!",
])
def test_is_chatty_true(msg):
    mod = _load_plugin_module()
    assert mod._is_chatty(msg)


@pytest.mark.parametrize("msg", [
    "implement tier 1",
    "search papers on arxiv",
    "yes, deploy the docker container now",
    "fix it",
    "docker",
])
def test_is_chatty_false(msg):
    mod = _load_plugin_module()
    assert not mod._is_chatty(msg)


def test_is_chatty_non_str_is_false():
    mod = _load_plugin_module()
    assert mod._is_chatty([{"text": "thanks"}]) is False


def test_hook_skips_chatty_before_touching_index(monkeypatch):
    mod = _load_plugin_module()

    def boom(*a, **k):
        raise AssertionError("get_index must not be called for chit-chat")

    monkeypatch.setattr(mod, "get_index", boom)
    assert mod._on_pre_llm_call("", "thanks!") is None


# ─── Truncation ──────────────────────────────────────────────────────────────

def test_truncate_desc_word_boundary():
    mod = _load_plugin_module()
    desc = ("word " * 60).strip()
    out = mod._truncate_desc(desc, 50)
    assert len(out) <= 50
    assert out.endswith("…")
    assert not out[:-1].endswith(" ")
    assert out[:-1].split()[-1] == "word"  # no mid-word cut


def test_truncate_desc_short_and_whitespace_collapse():
    mod = _load_plugin_module()
    assert mod._truncate_desc("short one") == "short one"
    assert mod._truncate_desc("folded\n  yaml   text") == "folded yaml text"
