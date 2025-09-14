# cli_args.py
from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass
from typing import Dict

# Import model catalog
def _load_models() -> Dict[str, str]:
    try:
        from src.models.registry import MODELS 
    except Exception as e:
        raise RuntimeError(
            "Unable to import src.models.config.MODELS. "
            "Ensure your PYTHONPATH includes the project root and that src is a package."
        ) from e
    if not isinstance(MODELS, dict) or not MODELS:
        raise RuntimeError("src.models.config.MODELS must be a non-empty dict[str, str].")
    # Normalize keys to strings and values to strings
    return {str(k): str(v) for k, v in MODELS.items()}

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")

@dataclass(frozen=True)
class CLIConfig:
    model: str
    ticker: str

def _validate_ticker(value: str) -> str:
    v = value.strip().upper()
    if not _TICKER_RE.match(v):
        raise argparse.ArgumentTypeError(
            f"Invalid ticker '{value}'. Use 1–10 chars A–Z, 0–9, '.', or '-'."
        )
    return v

def _validate_model(value: str, available: Dict[str, str]) -> str:
    v = value.strip()
    if v in available:
        return v
    # Suggest closest matches
    suggestions = difflib.get_close_matches(v, list(available.keys()), n=3, cutoff=0.5)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise argparse.ArgumentTypeError(f"Unknown model '{v}'.{hint}")

def _models_as_text(available: Dict[str, str]) -> str:
    # Pretty list: left-align model key, wrap descriptions
    try:
        import textwrap
        lines = ["Available models:\n"]
        max_key = max((len(k) for k in available), default=0)
        width = 88
        for name in sorted(available):
            desc = available[name].strip()
            prefix = f"  {name.ljust(max_key)}  "
            wrapped = textwrap.fill(desc, width=width, subsequent_indent=" " * len(prefix))
            lines.append(prefix + wrapped)
        return "\n".join(lines)
    except Exception:
        # Fallback if textwrap import or something silly fails
        parts = ["Available models:\n"]
        for k in sorted(available):
            parts.append(f"  {k}: {available[k]}")
        return "\n".join(parts)

def parse_cli_args(argv: list[str] | None = None) -> CLIConfig:
    """
    Parse command-line args for model and ticker.
    Also supports --list-models to print available models and exit.
    """
    available = _load_models()

    parser = argparse.ArgumentParser(
        prog="stock-predictor",
        description="CLI for choosing model and ticker."
    )

    parser.add_argument(
        "--model",
        required=False,
        help="Model identifier string (see --list-models).",
    )
    parser.add_argument(
        "--ticker",
        required=False,
        type=_validate_ticker,
        help="Ticker symbol, e.g. AAPL, MSFT, BRK.B.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models from src.models.registry and exit."
    )

    # If you prefer a subcommand instead of a flag, you can add one later. This is simpler.

    ns = parser.parse_args(argv)

    if ns.list_models:
        print(_models_as_text(available))
        sys.exit(0)

    # Enforce required when not listing
    if not ns.model or not ns.ticker:
        parser.error("the following arguments are required: --model, --ticker (or use --list-models)")

    model_name = _validate_model(ns.model, available)
    return CLIConfig(model=model_name, ticker=ns.ticker)
