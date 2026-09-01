# Annotation Platform

This directory is reserved for the public annotation interface used to collect
human judgments of cultural alignment.

The implementation is not included yet. Before adding it, ensure that:

- exported annotations exclude names, email addresses, and direct identifiers;
- project configuration and annotation schemas are documented;
- local development uses synthetic demonstration data;
- authentication secrets remain outside version control; and
- tests cover export compatibility with the evaluator at the repository root.
