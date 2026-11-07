@AGENTS.md

## Claude Code

`.claude/settings.json` enables the `pyright-lsp` plugin from the official
Claude plugins marketplace, so Claude Code gets Pyright diagnostics and code
navigation for the Python sources. It complements, and does not replace, the
mypy check in `make lint`.
