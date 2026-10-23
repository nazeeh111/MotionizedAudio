# Contributing

Thanks for your interest in contributing!

## How to contribute

1. **Open an issue first** — describe the bug or feature before writing code. No silent PRs.
2. **Fork and branch** — create a feature branch from `main`.
3. **Keep PRs small** — one logical change per PR.
4. **Keep Python readable** — run the supplied Ruff checks. CI checks import, selected syntax/style errors and unused names; it does not enforce all of PEP 8.
5. **Test before opening a PR** — run `./test.sh` (and `./test.sh gpu` if touching GPU code).

## Running tests

The container workflow runs lint and tests without installing project dependencies into your local Python:

```bash
./test.sh          # CPU: lint + unit tests
./test.sh gpu      # GPU: lint + unit tests (requires nvidia-container-toolkit)
./test.sh --build  # Force rebuild image before testing
```
