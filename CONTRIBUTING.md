# Contributing to pr-council-mcp

Thanks for helping improve pr-council-mcp. This is a Salesforce-sponsored open source project. Maintainers make the
final decisions about project direction and which contributions are accepted.

## Before you start

Search the [existing issues](https://github.com/salesforce-misc/pr-council-mcp/issues) before reporting a bug or
proposing a feature. For substantial changes, open an issue first so the approach can be discussed before you invest
significant time.

Bug reports should include clear reproduction steps, expected behavior, actual behavior, and relevant platform and
Python version information. Never include credentials, private repository content, or other sensitive data.

## Development setup

The project currently supports macOS and Python 3.12. Clone
[localmcplib](https://github.com/salesforce-misc/localmcplib) beside this repository, then install the locked
development environment:

```bash
uv sync --frozen
```

Run the complete validation suite before submitting a pull request:

```bash
make ci
uv build
```

Changes should be focused, include tests for observable behavior, and update documentation when user-facing behavior
changes. Keep commits atomic and descriptive, and link the relevant issue in the pull request.

## Contributor License Agreement

Contributors must sign the [Salesforce Contributor License Agreement](https://cla.salesforce.com/sign-cla). You only
need to do this once for Salesforce open source projects.

## Code of conduct and license

All contributors must follow the project [Code of Conduct](CODE_OF_CONDUCT.md). By contributing, you agree to license
your contribution under the terms of the project's [Apache-2.0 license](LICENSE.txt).
