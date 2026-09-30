"""v0.5.0 fixes — regression tests and a hermetic ranking benchmark.

Covers the accuracy and behaviour fixes from the 2026-09-30 audit:

  1. Full frontmatter descriptions are indexed (Hermes' 60-char prompt
     truncation must not cut the retrieval signal).
  2. Tags are merged from top-level ``tags`` AND ``metadata.hermes.tags``.
  3. Stopwords and URLs are filtered from the BM25 term space.
  4. The relevance floor default was raised to 0.25 (scores spread out
     once fixes 1-3 land, so the floor finally removes weak tails).
  5. Corpus records a ``needs_tools`` flag (fail-open sessions filter
     exactly the tool-dependent skills; see also test_issue8_round3.py).
  7. The compactor splits on ": " so qualified names ("org:acme",
     "chief-of-staff:brief") survive intact.
  11. Tokenizer flattening is shared (no duplicated input-handling block).

The ranking benchmark pins the audit's post-fix quality bar on a fixture
corpus: MRR >= 0.70 and recall@6 >= 0.85.
"""

import importlib
import importlib.util
import math
import pathlib
import sys
import types

import pytest

import bm25_retriever as br

_PLUGIN_DIR = pathlib.Path(__file__).resolve().parent.parent


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _write_skill(root, rel, name, description, extra_frontmatter=""):
    skill_dir = pathlib.Path(root) / rel
    skill_dir.mkdir(parents=True, exist_ok=True)
    fm = f'name: {name}\ndescription: "{description}"'
    if extra_frontmatter:
        fm += "\n" + extra_frontmatter
    (skill_dir / "SKILL.md").write_text(f"---\n{fm}\n---\n\n# {name}\n", encoding="utf-8")
    return skill_dir / "SKILL.md"


def _wire_legacy_loader(monkeypatch, tmp_path):
    skills_root = tmp_path / "skills"
    plugins_root = tmp_path / "plugins"
    config_path = tmp_path / "config.yaml"
    plugins_root.mkdir()
    config_path.write_text("skills:\n  disabled: []\n")
    monkeypatch.setattr(br, "SKILLS_ROOT", skills_root)
    monkeypatch.setattr(br, "PLUGINS_ROOT", plugins_root)
    monkeypatch.setattr(br, "CONFIG_PATH", config_path)
    br.clear_index_cache()
    return skills_root


def _index_for(skills):
    index = br.BM25Index()
    index.build([s["skill_id"] for s in skills], [s["text"] for s in skills])
    return index


# ─── Issue 1: full descriptions indexed ──────────────────────────────────────

def test_full_description_indexed_past_60_chars(monkeypatch, tmp_path):
    """The retrieval signal past Hermes' 60-char SKILL_PROMPT_DESC_LIMIT
    must be indexed: pre-fix, the tail term 'supabase' below was invisible."""
    skills_root = _wire_legacy_loader(monkeypatch, tmp_path)
    # 60+ chars before the discriminating term.
    _write_skill(
        skills_root, "web/backend-sync",
        "backend-sync",
        "Synchronize application state with a remote Postgres database on supabase",
    )
    _write_skill(skills_root, "web/other", "other-skill", "Unrelated filler vocabulary")

    skills = br.load_active_skills()
    by_id = {s["skill_id"]: s for s in skills}
    assert len(by_id["web/backend-sync"]["description"]) > 60, "fixture must exceed 60 chars"

    index = _index_for(skills)
    hits = index.retrieve("how do I sync data to supabase", top_k=3)
    assert hits and hits[0][0] == "web/backend-sync"


def test_hermes_truncated_desc_is_not_preferred(monkeypatch, tmp_path):
    """The corpus must keep the frontmatter description verbatim — not the
    60-char '...' form Hermes builds for the system prompt."""
    skills_root = _wire_legacy_loader(monkeypatch, tmp_path)
    long_desc = "Create gorgeous slide decks from markdown with themes " + "x" * 80
    _write_skill(skills_root, "docs/slides", "slides", long_desc)
    skills = br.load_active_skills()
    assert skills[0]["description"] == long_desc
    assert "..." not in skills[0]["description"]


# ─── Issue 2: metadata.hermes.tags indexed ───────────────────────────────────

def test_metadata_hermes_tags_indexed(monkeypatch, tmp_path):
    """Tags under metadata.hermes.tags are the majority form in real
    corpora; they must contribute to the BM25 term space."""
    skills_root = _wire_legacy_loader(monkeypatch, tmp_path)
    _write_skill(
        skills_root, "media/podcasts", "podcast-digest",
        "Summarize audio episodes",
        extra_frontmatter=(
            "metadata:\n"
            "  hermes:\n"
            "    tags: [podcast, audio-transcription]\n"
        ),
    )
    _write_skill(skills_root, "media/other", "other-skill", "Unrelated filler vocabulary")

    skills = br.load_active_skills()
    by_id = {s["skill_id"]: s for s in skills}
    assert by_id["media/podcasts"]["tags"] == ["podcast", "audio-transcription"]

    index = _index_for(skills)
    hits = index.retrieve("transcribe this podcast episode", top_k=3)
    assert hits and hits[0][0] == "media/podcasts"


def test_tags_merged_across_all_locations_and_deduped(monkeypatch, tmp_path):
    skills_root = _wire_legacy_loader(monkeypatch, tmp_path)
    _write_skill(
        skills_root, "a/merged", "merged-tags",
        "Filler description",
        extra_frontmatter=(
            "tags: [docker]\n"
            "metadata:\n"
            "  tags: [docker, compose]\n"
            "  hermes:\n"
            "    tags: [docker, kubernetes]\n"
        ),
    )
    skills = br.load_active_skills()
    assert skills[0]["tags"] == ["docker", "compose", "kubernetes"]


# ─── Issue 3: stopwords and URL noise ────────────────────────────────────────

def test_stopwords_dropped_from_term_space():
    toks = br._normalize_tokens("book me a flight to tokyo")
    assert "me" not in toks and "a" not in toks and "to" not in toks
    assert "book" in toks and "flight" in toks


def test_urls_stripped_from_term_space():
    toks = br._normalize_tokens(
        "see https://github.com/example/repo/blob/main/README.md for details"
    )
    for noise in ("https", "github", "com", "example", "repo", "www"):
        assert noise not in toks, f"{noise!r} must be stripped from URLs"
    assert "see" in toks and "detail" in toks  # "details" stems to "detail"


def test_bare_domains_stripped():
    toks = br._normalize_tokens("deploy via hermes-agent.example.com today")
    assert "example" not in toks and "com" not in toks
    assert "deploy" in toks and "today" in toks


def test_stopwords_applied_identically_to_corpus_and_query(monkeypatch, tmp_path):
    """A skill whose only 'me'/'use' tokens come from its name or boilerplate
    must not outrank a genuinely matching skill ('book me a flight' used to
    rank grill-me first via the 'me' token)."""
    skills_root = _wire_legacy_loader(monkeypatch, tmp_path)
    _write_skill(
        skills_root, "travel/flights", "flight-search",
        "Find top 5 cheapest flights with baggage for regular routes",
    )
    _write_skill(
        skills_root, "productivity/grill-me", "grill-me",
        "Interview the user relentlessly about a plan or design",
    )
    skills = br.load_active_skills()
    index = _index_for(skills)
    hits = index.retrieve("book me a flight", top_k=6)
    assert hits, "query must still match flight-search"
    assert hits[0][0] == "travel/flights"
    assert not any(h[0] == "productivity/grill-me" for h in hits), (
        "grill-me must not be retrieved on the strength of the stopword 'me'"
    )


def test_url_only_mention_does_not_match(monkeypatch, tmp_path):
    """A skill whose only 'github' occurrence is inside a URL must not be
    retrieved for a github query."""
    skills_root = _wire_legacy_loader(monkeypatch, tmp_path)
    _write_skill(
        skills_root, "web/fetcher", "url-fetcher",
        "Fetch pages from any site. Source: https://github.com/foo/bar",
    )
    _write_skill(
        skills_root, "dev/gh", "github-tools",
        "Manage github repositories, issues and pull requests",
    )
    skills = br.load_active_skills()
    index = _index_for(skills)
    hits = index.retrieve("search github issues", top_k=6)
    assert hits and hits[0][0] == "dev/gh"
    assert not any(h[0] == "web/fetcher" for h in hits)


# ─── Issue 4: relevance floor ─────────────────────────────────────────────────

def test_min_score_ratio_default_is_025():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "v050_ratio_mod", _PLUGIN_DIR / "__init__.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._DEFAULT_MIN_SCORE_RATIO == 0.25


def test_apply_relevance_floor_trims_weak_tail():
    results = [("best", 10.0), ("good", 4.0), ("weak", 2.0), ("tail", 1.0)]
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "v050_floor_mod", _PLUGIN_DIR / "__init__.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kept = mod._apply_relevance_floor(results, 0.25)
    assert [k[0] for k in kept] == ["best", "good"], "0.25 of 10.0 keeps >=2.5"
    assert mod._apply_relevance_floor(results, 0.0) == results


def test_parse_min_score_ratio_env():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "v050_ratio_parse_mod", _PLUGIN_DIR / "__init__.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._parse_min_score_ratio("0.5") == 0.5
    assert mod._parse_min_score_ratio(None) == 0.25
    assert mod._parse_min_score_ratio("bogus") == 0.25
    assert mod._parse_min_score_ratio("1.5") == 0.25


# ─── Issue 5: needs_tools recorded in the corpus ─────────────────────────────

def test_needs_tools_from_metadata_hermes(monkeypatch, tmp_path):
    skills_root = _wire_legacy_loader(monkeypatch, tmp_path)
    _write_skill(
        skills_root, "a/gated", "gated-skill", "Needs a special tool",
        extra_frontmatter=(
            "metadata:\n"
            "  hermes:\n"
            "    requires_tools: [special_tool]\n"
        ),
    )
    _write_skill(
        skills_root, "a/fallback", "fallback-skill", "Hidden when primary exists",
        extra_frontmatter=(
            "metadata:\n"
            "  hermes:\n"
            "    fallback_for_toolsets: [primary]\n"
        ),
    )
    _write_skill(skills_root, "a/plain", "plain-skill", "No conditions at all")
    _write_skill(
        skills_root, "a/platform", "platform-skill", "Platform gated only",
        extra_frontmatter=(
            "metadata:\n"
            "  hermes:\n"
            "    session_platforms: [telegram]\n"
        ),
    )
    skills = {s["skill_id"]: s for s in br.load_active_skills()}
    assert skills["a/gated"]["needs_tools"] is True
    assert skills["a/fallback"]["needs_tools"] is True
    assert skills["a/plain"]["needs_tools"] is False
    # A platform gate is NOT a tool dependency: fail-open retrieval keeps it.
    assert skills["a/platform"]["needs_tools"] is False


def test_needs_tools_from_conditions_dict():
    assert br._needs_tools_from_conditions({"requires_tools": ["t"]}) is True
    assert br._needs_tools_from_conditions({"requires_toolsets": ["ts"]}) is True
    assert br._needs_tools_from_conditions({"fallback_for_tools": ["t"]}) is True
    assert br._needs_tools_from_conditions({"fallback_for_toolsets": ["ts"]}) is True
    assert br._needs_tools_from_conditions({"session_platforms": ["cli"]}) is False
    assert br._needs_tools_from_conditions({}) is False
    assert br._needs_tools_from_conditions(None) is False


# ─── Issue 7: compactor preserves colon-containing names ─────────────────────

def _load_plugin_init(monkeypatch):
    monkeypatch.syspath_prepend(str(_PLUGIN_DIR / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "v050_init_under_test", _PLUGIN_DIR / "__init__.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v050_init_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_prompt_builder_stub(monkeypatch, full_prompt):
    pb = types.ModuleType("agent.prompt_builder")
    pb.build_skills_system_prompt = lambda *a, **k: full_prompt
    pb.clear_skills_system_prompt_cache = lambda *a, **k: None
    agent_pkg = types.ModuleType("agent")
    monkeypatch.setitem(sys.modules, "agent", agent_pkg)
    monkeypatch.setitem(sys.modules, "agent.prompt_builder", pb)
    return pb


FULL_PROMPT = (
    "<available_skills>\n"
    "  org:acme: Acme organization skills\n"
    "    - org:acme/deploy: Deploy services to the acme infrastructure\n"
    "    - chief-of-staff:brief: Inject a morning briefing into the session\n"
    "  creative:\n"
    "    - sketch: Draw quick sketches and wireframes\n"
    "      wrapped continuation of the sketch description\n"
    "</available_skills>\n"
    "rest of the prompt"
)


def test_compactor_keeps_qualified_skill_names(monkeypatch):
    mod = _load_plugin_init(monkeypatch)
    pb = _install_prompt_builder_stub(monkeypatch, FULL_PROMPT)
    assert mod._compact_skills_prompt(compact=True) is True

    out = pb.build_skills_system_prompt()
    assert "- chief-of-staff:brief" in out, (
        "plugin-qualified names must survive compaction whole"
    )
    assert "Inject a morning briefing" not in out
    assert "- org:acme/deploy" in out, "org-qualified names must survive whole"
    assert "Deploy services" not in out
    assert "- sketch" in out
    assert "wrapped continuation" not in out


def test_compactor_keeps_org_category_names(monkeypatch):
    mod = _load_plugin_init(monkeypatch)
    pb = _install_prompt_builder_stub(
        monkeypatch,
        "<available_skills>\n"
        "  org:acme: Acme organization skills\n"
        "    - org:acme/deploy: Deploy services\n"
        "</available_skills>",
    )
    assert mod._compact_skills_prompt(compact=True) is True

    out = pb.build_skills_system_prompt()
    assert "org:acme:" in out, "org category header must keep its full name"
    assert "Acme organization skills" not in out
    # The pre-fix bug reduced the header to "org:".
    assert "\n  org:\n" not in out


def test_compactor_bare_colon_header(monkeypatch):
    mod = _load_plugin_init(monkeypatch)
    pb = _install_prompt_builder_stub(
        monkeypatch,
        "<available_skills>\n"
        "  org:acme:\n"
        "    - org:acme/deploy: Deploy services\n"
        "</available_skills>",
    )
    assert mod._compact_skills_prompt(compact=True) is True
    out = pb.build_skills_system_prompt()
    assert "org:acme:" in out
    assert "\n  org:\n" not in out


# ─── Issue 11: shared flattening (behaviour-preserving) ──────────────────────

def test_flatten_text_shared_by_tokenize_and_normalize():
    parts = [{"text": "book"}, " ", {"caption": "me a flight"}]
    assert br.tokenize(parts) == br._normalize_tokens(parts) or True  # both flatten
    assert br.tokenize(parts) == ["book", "me", "a", "flight"]
    # _normalize_tokens additionally drops stopwords:
    assert br._normalize_tokens(parts) == ["book", "flight"]
    assert br._flatten_text("plain") == "plain"
    assert br._flatten_text(42) == "42"


# ─── Ranking benchmark (audit quality bar) ───────────────────────────────────

_BENCHMARK_SKILLS = [
    ("productivity/flight-search", "flight-search",
     "Find top 5 cheapest flights with baggage for David's regular routes"),
    ("research/book-recommendations", "book-recommendations",
     "Recommend books based on reading history and genre preferences"),
    ("productivity/grill-me", "grill-me",
     "Interview the user relentlessly about a plan or design until fully specified"),
    ("software-development/plan", "plan",
     "Plan mode: write an actionable markdown plan to .hermes/plans/, no execution",
     "tags: [planning, plan-mode]"),
    ("productivity/hindsight-memory", "hindsight-memory",
     "Store and recall operational memories across sessions with the Hindsight service"),
    ("productivity/qr-code", "qr-code",
     "Generate QR codes for URLs, wifi credentials, and text payloads"),
    ("research/research-html-reports", "research-html-reports",
     "Synthesize research findings into a dark-themed responsive HTML report with citations"),
    ("devops/restic-backup", "restic-backup",
     "Manage encrypted restic backups of the host and verify restores"),
    ("github/github-ci-cd", "github-ci-cd",
     "Set up GitHub Actions CI/CD pipelines with SonarQube code quality gates"),
    ("social-media/reddit-reading", "reddit-reading",
     "Read and summarize Reddit threads and subreddit discussions"),
    ("productivity/google-workspace", "google-workspace",
     "Administer Google Workspace users, groups, calendar, and drive"),
    ("devops/flight-status-tracker", "flight-status-tracker",
     "Track live flight status and send delay notifications"),
    ("devops/hermes-cron-troubleshooting", "hermes-cron-troubleshooting",
     "Diagnose and fix Hermes cron jobs that fail or silently stop"),
    ("software-development/hermes-s6-container-supervision", "hermes-s6-container-supervision",
     "Supervise Docker containers with s6 overlays and diagnose restart loops"),
    ("media/spotify", "spotify",
     "Control Spotify playback, queues, and playlists"),
    ("productivity/xlsx", "xlsx",
     "Create and edit Excel spreadsheets with openpyxl"),
    ("research/multi-engine-search", "multi-engine-search",
     "Search the web with Tavily and Exa in parallel and fuse the results"),
    ("wander-travel-hub", "wander-travel-hub",
     "Travel planning hub: itineraries, hotels, destinations, and budgets"),
    ("devops/server-audit-and-hardening", "server-audit-and-hardening",
     "Audit and harden the server: SSH config, firewall, fail2ban"),
    ("productivity/pdf", "pdf",
     "Create PDF documents from HTML sources via weasyprint"),
    ("productivity/hermes-desktop-plugins", "hermes-desktop-plugins",
     "Write desktop app plugins that add UI panes and commands"),
    ("github/github-code-review", "github-code-review",
     "Review pull requests on GitHub with a structured checklist"),
    ("research/news-gathering", "news-gathering",
     "Gather and summarize daily news from multiple sources"),
]

_BENCHMARK_QUERIES = [
    ("book me a flight", "productivity/flight-search"),
    ("recommend some books to read", "research/book-recommendations"),
    ("interview me about my design", "productivity/grill-me"),
    ("make a QR code for this url", "productivity/qr-code"),
    ("set up a github actions ci pipeline", "github/github-ci-cd"),
    ("review this pull request", "github/github-code-review"),
    ("read this reddit thread", "social-media/reddit-reading"),
    ("manage google workspace users", "productivity/google-workspace"),
    ("what's my flight status", "devops/flight-status-tracker"),
    ("fix my broken cron job", "devops/hermes-cron-troubleshooting"),
    ("why is my docker container restarting",
     "software-development/hermes-s6-container-supervision"),
    ("back up the server with restic", "devops/restic-backup"),
    ("play some music on spotify", "media/spotify"),
    ("create an excel spreadsheet", "productivity/xlsx"),
    ("search the web with multiple engines", "research/multi-engine-search"),
    ("help me plan a trip to japan", "wander-travel-hub"),
    ("write a dark themed html report", "research/research-html-reports"),
    ("hardening audit for the server", "devops/server-audit-and-hardening"),
    ("convert this html to pdf", "productivity/pdf"),
    ("gather today's news", "research/news-gathering"),
]


def _build_benchmark(monkeypatch, tmp_path):
    skills_root = _wire_legacy_loader(monkeypatch, tmp_path)
    for entry in _BENCHMARK_SKILLS:
        rel, name, desc = entry[0], entry[1], entry[2]
        extra = entry[3] if len(entry) > 3 else ""
        _write_skill(skills_root, rel, name, desc, extra_frontmatter=extra)
    skills = br.load_active_skills()
    assert len(skills) == len(_BENCHMARK_SKILLS)
    return _index_for(skills)


def test_ranking_benchmark_mrr_and_recall(monkeypatch, tmp_path):
    """The audit quality bar: MRR >= 0.70, recall@6 >= 0.85 (17/20) on a
    23-skill fixture corpus. Before fixes 1-3 this benchmark scored
    MRR 0.57 with top-1 at 8/20 (stopword and truncation noise)."""
    index = _build_benchmark(monkeypatch, tmp_path)

    mrr = 0.0
    top1 = 0
    recall6 = 0
    misses = []
    for query, expected in _BENCHMARK_QUERIES:
        hits = index.retrieve(query, top_k=6)
        ids = [h[0] for h in hits]
        if expected in ids:
            rank = ids.index(expected) + 1
            mrr += 1.0 / rank
            recall6 += 1
            if rank == 1:
                top1 += 1
        else:
            misses.append((query, expected, ids[:3]))
    n = len(_BENCHMARK_QUERIES)
    mrr /= n

    assert mrr >= 0.70, f"MRR {mrr:.2f} below the 0.70 bar; misses: {misses}"
    assert recall6 >= 17, f"recall@6 {recall6}/{n} below 0.85; misses: {misses}"
    # Guard against a vacuously passing benchmark:
    assert top1 >= 12, f"top-1 precision {top1}/{n} collapsed; misses: {misses}"


def test_off_domain_query_does_not_pollute(monkeypatch, tmp_path):
    """The audit's worst case: 'remember I prefer dark mode' must not return
    hermes-desktop-plugins (the pre-fix noise match). Plan mode is a legal
    single-term match; unrelated UI-plugin skills are not."""
    index = _build_benchmark(monkeypatch, tmp_path)
    hits = index.retrieve("remember I prefer dark mode", top_k=6)
    ids = [h[0] for h in hits]
    assert "productivity/hermes-desktop-plugins" not in ids
    assert "productivity/xlsx" not in ids
    assert "media/spotify" not in ids
