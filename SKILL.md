---
name: skill-retrieval
description: >-
  BM25-based skill retrieval plugin for Hermes Agent. Replaces the full skill
  list in the system prompt with a names-only compact view (~2K tokens) and
  injects top-K relevant skill descriptions per turn via BM25 retrieval (~300
  tokens). Saves ~9K tokens/turn. Use when system prompt token overhead from
  skills is a concern, or when skill discovery quality matters.
license: MIT
metadata:
  version: 0.5.0
  author: Green-Needle-Tech
  platforms: [linux, macos, windows]
  tags: [bm25, skill-retrieval, system-prompt, token-optimization, plugin]
  hermes:
    plugin_type: hook
    hooks: [pre_llm_call]
---

# Skill Retrieval

This is a **Hermes Agent** plugin. It is not a Claude Code plugin and will
not load in Claude Code — that runtime has no `pre_llm_call` event, no Python
`register()` entry point, and reads `.claude-plugin/plugin.json` rather than
`plugin.yaml`. Requires Hermes Agent >=0.21.1 (`requires_hermes` in
`plugin.yaml`).

BM25-based progressive disclosure for Hermes Agent skills. Instead of dumping
every skill description into the system prompt (~11.5K tokens), this plugin
keeps a compact names-only index and injects only the top-K relevant
descriptions per turn.

## What it does

Two-phase progressive disclosure:

1. **Phase 1 — System prompt compaction** (session start): Monkey-patches
   `build_skills_system_prompt` so the `<available_skills>` block lists skill
   names only (descriptions stripped). All skills remain discoverable by name
   (~2K tokens instead of ~11.5K).

2. **Phase 2 — Per-turn BM25 retrieval** (`pre_llm_call` hook): Tokenizes the
   user message, ranks active skill descriptions with BM25 Okapi, and injects
   the top-K matches (~300 tokens) as context appended after the user
   message.

## Architecture

```
Session start
    │
    ▼
Phase 1: patch build_skills_system_prompt
    └── <available_skills> → names only (~2K tokens)

Each turn (pre_llm_call)
    │
    ▼
Phase 2: BM25Index.retrieve(user_message, top_k)
    └── inject "## Retrieved Skills ..." into user message (~300 tokens)
```

The BM25 index is built lazily on the first turn from standalone skills
(`~/.hermes/skills`) and plugin-bundled skills (`~/.hermes/plugins/*/skills`),
then cached per Hermes home, tool capability snapshot, session platform and
resolved disabled-skill set (the same inputs Hermes keys its own skills prompt
cache on). The cache is rebuilt without a restart when Hermes clears its skills
prompt cache (`skill_manage` create/patch/delete, hub install, skill toggles)
or when a cheap on-disk manifest changes (skill root and category dir mtimes,
top-level `SKILL.md` files, `config.yaml`).
Retrieval uses a pure-stdlib inverted index (term → posting list of
precomputed BM25 weights) and is sub-millisecond for ~200 skills.

## Token savings

| Stage | Tokens (approx.) |
|-------|------------------|
| Before (full skill list in system prompt) | ~11.5K |
| After — names-only system prompt | ~2.0K |
| After — per-turn top-K descriptions | ~0.3K |
| **Net per turn** | **~2.3K** (~9K saved) |

Measured on a Hermes install with ~300 skills; savings scale with skill count.

## Installation

This repository IS the plugin directory. Copy or symlink it into the Hermes
plugins folder:

```bash
# From a checkout of this repo (repo root = plugin root)
ln -s "$(pwd)" ~/.hermes/plugins/skill-retrieval

# Or copy
cp -r . ~/.hermes/plugins/skill-retrieval
```

User plugins are opt-in: Hermes discovers the directory but does not load it
until it is enabled (this adds `skill-retrieval` to `plugins.enabled` in
`~/.hermes/config.yaml`):

```bash
hermes plugins enable skill-retrieval
```

Restart the agent session so `register()` runs — it patches the system prompt
and registers the `pre_llm_call` hook.

Dependencies (install into the Hermes Python env if missing):

```bash
pip install pyyaml
```

## Configuration

All settings are environment variables (read once at plugin import; restart
the session to apply):

| Setting | Default | How to set |
|---------|---------|------------|
| `TOP_K` | `6` | `SKILL_RETRIEVAL_TOP_K` |
| System prompt compaction | enabled | `SKILL_RETRIEVAL_COMPACT=0` disables compaction while keeping BM25 retrieval injection |
| BM25 `k1` | `1.5` | `SKILL_RETRIEVAL_K1` (must be > 0) |
| BM25 `b` | `0.75` | `SKILL_RETRIEVAL_B` (must be > 0) |
| Relevance floor | `0.25` | `SKILL_RETRIEVAL_MIN_SCORE_RATIO` (in `[0, 1)`; results scoring below this fraction of the top score are dropped) |

```bash
export SKILL_RETRIEVAL_TOP_K=8
export SKILL_RETRIEVAL_COMPACT=0
export SKILL_RETRIEVAL_K1=1.2
export SKILL_RETRIEVAL_B=0.7
export SKILL_RETRIEVAL_MIN_SCORE_RATIO=0.35
```

Invalid values (non-numeric, out of range) log a warning and fall back to
the default.

## Verify it's working

Phase 1 silently no-ops outside a full Hermes runtime, and the BM25 index can
silently empty. After restart, check the Hermes logs.

**Healthy start — look for these log lines:**

- `BM25 index built: N docs …`
- `Skill retrieval plugin registered (top_k=…, compact=true)`

**Degraded — these warnings mean it's not working:**

- `Cannot locate prompt_builder — compaction skipped` (Phase 1 failed; Phase 2
  still runs — sessions without a recorded capability snapshot fall back to
  fail-open retrieval with tool-dependent skills filtered out)
- `No active skills found for BM25 index` (index is empty — zero retrieval injection)

## How it works

**Corpus.** Each skill becomes one BM25 document built from its SKILL.md
frontmatter with field boosts: the name is repeated 3x (users and the model
reference skills by name), tags 2x (the capability vocabulary queries
actually use), the category path once, and the FULL description once —
Hermes truncates prompt descriptions to 60 characters, and the index
deliberately bypasses that cut. Tags are merged from top-level `tags`,
`metadata.tags` AND `metadata.hermes.tags` (the majority form in real
corpora). Disabled skills and plugin skills gated off for the session are
skipped; every recorded skill carries a `needs_tools` flag derived from its
`requires_tools` / `requires_toolsets` / `fallback_for_*` conditions.

**Term pipeline** (applied identically to the corpus and every query):
URLs and bare domains are stripped whole; CamelCase compounds are split
(`skillView` → `skillView skill View`, keeping the original); Chinese and
Japanese text is segmented into runs plus overlapping bigrams; common
English words and web fragments (`me`, `the`, `use`, `https`, `com`, …)
are dropped as stopwords; a conservative suffix stripper stems the rest
(`skills` → `skill`, `studies` → `study`).

**Index.** BM25 Okapi TF saturation + Lucene IDF
`log(1 + (N-df+0.5)/(df+0.5))` (always positive, so small corpora and
common terms still score), stored as an inverted index:
``dict[str, list[tuple[int, float]]]`` mapping each term to a posting list
of (doc_index, precomputed BM25 weight).

**Retrieve.** For each unique query token present in the index, walk its
posting list and accumulate scores; sort by descending score (score > 0
only); then drop results scoring below `SKILL_RETRIEVAL_MIN_SCORE_RATIO`
(default 0.25) of the top score, so off-domain messages don't inject a
top-K of noise.

## Performance

- Index built on first use and cached (~8 ms for 200 skills on a CPU-only
  VM). Each later turn only stats the skill roots, their immediate
  subdirectories and `config.yaml` to validate the cache. A zero-skill
  install caches the empty result too (one warning, no rescans).
- Retrieval is sub-millisecond (~0.03 ms mean for 200 skills). The inverted
  index touches only documents that share a query term — no full-corpus scan.
- No compiled dependencies. The plugin uses only the Python standard library
  (plus pyyaml for config/frontmatter parsing). This removes a 154 MB
  numpy/scipy install and a ~573 ms import cost, which matters for subprocess
  spawning paths (e.g. a Claude Code `UserPromptSubmit` variant).
- Failures in the hook return `None` (no injection) so the agent keeps working.

## Dependencies

- `pyyaml`

## Limitations

- BM25 is **lexical**, not semantic. Paraphrased queries that share few tokens
  with a skill's description may rank poorly even when the intent matches.
- Descriptions longer than 200 characters are truncated in the injected block;
  use `skill_view(name)` for the full skill body.
- Compaction requires Hermes's `agent.prompt_builder` module; if it cannot be
  imported, Phase 1 is skipped.
- A named session whose system prompt was never built in this process (after
  a gateway restart, or when resuming a session) has no recorded capability
  snapshot. Since v0.5.0 the hook keeps retrieving anyway — fail-open — and
  filters out only the skills whose activation depends on tool capabilities
  (`requires_tools` / `requires_toolsets` / `fallback_for_*`). Skills with
  platform or gateway-channel gates are evaluated as usual.
- Each turn's injected "Retrieved Skills" block stays in the conversation
  history, so the token saving is gradually consumed in long sessions
  (roughly 8–9 turns of top-6 injections on a 100+ skill install). For
  long-running sessions, lower `SKILL_RETRIEVAL_TOP_K` or set
  `SKILL_RETRIEVAL_COMPACT=0` and rely on `skill_view(name)` alone.
- A content-only edit of a nested `SKILL.md` (`category/skill/SKILL.md`)
  made outside Hermes (e.g. in an editor) changes no directory mtime, so it is
  picked up only after Hermes clears its skills prompt cache or the agent
  restarts. Edits through `skill_manage`, and any added/removed skill or
  config change, are picked up on the next turn.
- Phase 1 depends on Hermes internals (`agent.prompt_builder`) and can break
  on a Hermes upgrade.
- BM25 top-1 precision is soft: the best-matching skill is often not rank 1,
  though it usually lands within the first few results. v0.5.0 materially
  improved ranking (full-length descriptions, merged tags, stopword/URL
  filtering, a 0.25 relevance floor): on a 23-skill benchmark with 20
  real-world-style queries, top-1 is 18/20, recall@6 19/20, MRR 0.92
  (previously MRR 0.57). Ranking still depends entirely on your own corpus
  and how its descriptions are worded, so `TOP_K` below ~5 is not
  recommended.
- The stdlib index computes in float64 (the previous scipy version used
  float32). Equal-scoring skills may order differently than before. This is
  harmless — the scores are genuine ties (~1e-6 difference) — but it is a real
  behaviour delta from the scipy version.
