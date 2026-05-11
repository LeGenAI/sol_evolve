# Security

SolEvolve can execute generated Python and SAT solver commands in research workflows. Treat untrusted prompts, generated scripts, CNF files, and solver binaries as untrusted code.

- Keep API keys in `.env` or environment variables only.
- Do not upload raw secrets, full solver models, or large generated files to LangSmith.
- Review any command before running human-in-the-loop demos with solver/tool execution enabled.
- Prefer `--dry-run` for public CI and initial environment checks.
