# `misanthropic` project

The `misanthropic` project provides a simple, ergonomic, client for the Anthropic Messages API along with some example applications. It is designed to be easy to use and integrate into your own projects.

## Contents:

- [the `misanthropic` crate itself](misanthropic/README.md)
- [the `chat` demo](chat/README.md)

Additional (cli) examples can be found in the [`misanthropic/examples`](/misanthropic/examples/) directory.

## Developing

This project uses [`just`](https://github.com/casey/just) as a task runner.
After cloning, enable the pre-commit gate once:

```sh
just install-hooks   # runs `just test` before every commit
```

- `just test` — the offline gate: format, lint, and the all-features +
  no-default-features test matrix (mirrors CI; costs nothing per commit).
- `just test-ignored` — the `#[ignore]`d tests that hit the live API (needs an
  API key in `misanthropic/api.key`).

Please do not bypass the gate with --no-verify.

## FAQ

See [here](misanthropic/README.md) for the main FAQ and crate README.md.