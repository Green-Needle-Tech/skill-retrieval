# skill-retrieval

BM25-based skill retrieval plugin for [Hermes Agent](https://hermes-agent.nousresearch.com). Replaces the full skill list in the system prompt with a names-only compact view (~2K tokens) and injects top-K relevant skill descriptions per turn (~300 tokens), saving ~9K tokens/turn while keeping skills discoverable by name. Those figures were measured on a Hermes install with ~300 skills; the saving scales with your own skill count.

This is a **Hermes Agent** plugin. It is not a Claude Code plugin and will not load in Claude Code — that runtime has no `pre_llm_call` event, no Python `register()` entry point, and reads `.claude-plugin/plugin.json` rather than `plugin.yaml`. Requires Hermes Agent >=0.21.1.

See [SKILL.md](SKILL.md) for full architecture, token measurements, how it works, performance, limitations, and how to verify a healthy install.

## Installation

This repository IS the plugin directory (repo root = plugin root):

```bash
# From a checkout of this repo
ln -s "$(pwd)" ~/.hermes/plugins/skill-retrieval

# Dependencies (Hermes Python env)
# pyyaml is the only dependency — usually already present in a Hermes env
pip install pyyaml
```

User plugins are opt-in — Hermes discovers the directory but does not load it until you enable it:

```bash
hermes plugins enable skill-retrieval
```

Then restart the Hermes session so the plugin's `register()` runs.

## Configuration

All settings are environment variables (read once at plugin import; restart
the session to apply):

| Setting | Default | Override |
|---------|---------|----------|
| Top-K results | `6` | `SKILL_RETRIEVAL_TOP_K` env var |
| System prompt compaction | enabled | `SKILL_RETRIEVAL_COMPACT=0` disables compaction but keeps retrieval injection |
| BM25 k1 | `1.5` | `SKILL_RETRIEVAL_K1` env var |
| BM25 b | `0.75` | `SKILL_RETRIEVAL_B` env var |
| Relevance floor | `0.25` | `SKILL_RETRIEVAL_MIN_SCORE_RATIO` env var (in `[0, 1)`) |

```bash
export SKILL_RETRIEVAL_TOP_K=8
export SKILL_RETRIEVAL_COMPACT=0  # retrieval-only mode; the skills prompt is left unchanged (the builder is still wrapped to record tool capabilities)
```

## Development

The test suite runs without a Hermes install (pure stdlib + pytest + pyyaml):

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

CI runs the suite on every push and pull request
([.github/workflows/ci.yml](.github/workflows/ci.yml)).

## A note on history

The repository's first commit message says "v1.0.0" but the first public
release was tagged `v0.4.0` (the version in `plugin.yaml` at that commit).
The published history was left unrewritten; `v0.5.0` onward follows
`plugin.yaml` exactly.

## Uninstall

1. Disable the plugin (`hermes plugins disable skill-retrieval`), then remove it from the Hermes plugins folder:
   ```bash
   # If symlinked:
   rm ~/.hermes/plugins/skill-retrieval
   # If copied:
   rm -rf ~/.hermes/plugins/skill-retrieval
   ```
2. Restart the Hermes session
3. The system prompt reverts on restart (the patch is in-process only, not persistent)

## License

MIT. See [LICENSE](LICENSE) for the upstream SR-Agents notice and this project's notice.
