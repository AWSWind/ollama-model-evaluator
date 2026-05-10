"""Reusable fakes for integration and unit tests.

Each fake exposes an ASGI app (for HTTP fakes) or an in-memory object
(for other collaborators) so tests can drive the Backend without real
network or filesystem dependencies.

Design reference: ``.kiro/specs/ollama-model-evaluator/design.md``
§Test data hygiene: "every test uses the in-process FakeOllamaServer or
FakeOllamaClient".
"""
