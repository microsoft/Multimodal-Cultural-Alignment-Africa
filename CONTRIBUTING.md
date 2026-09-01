# Contributing

This project welcomes contributions and suggestions.

Most contributions require you to agree to a Contributor License Agreement
(CLA) declaring that you have the right to, and actually do, grant Microsoft
the rights to use your contribution. For details, see
[Microsoft Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will determine whether you need to
provide a CLA and will add the appropriate status check or comment. You
typically need to complete this process only once across Microsoft-managed
repositories.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest -q
```

Before submitting a change:

1. Add or update tests for changed behavior.
2. Keep raw annotation exports, credentials, personal data, and generated
   research results out of the repository.
3. Use synthetic records for tests and documentation.
4. Confirm that `pytest -q` passes.
5. Explain user-visible behavior and reproducibility implications in the pull
   request.

By participating, you agree to follow the
[Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md).
