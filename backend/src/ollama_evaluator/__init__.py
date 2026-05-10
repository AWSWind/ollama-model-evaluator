"""Ollama Model Evaluator backend package.

This package is the home of the Python backend that orchestrates evaluations
against an Ollama server, scores responses, and exposes a CLI and HTTP API.

The concrete implementation lives in later tasks; at scaffold time the package
only exposes its version string.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
