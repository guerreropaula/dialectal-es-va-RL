"""
dialectal_score.py
==================
Standalone script to compute the Dialectal Valencian Score for a set of
hypotheses. Measures what proportion of CA↔VA contrastive slots the model
fills with Valencian-specific forms.

Usage
-----
    python dialectal_score.py --input hypotheses.txt [--output report.json]

    # Or import as a module:
    from dialectal_score import dialectal_score, CA_VA_FEATURES

Input format
------------
One hypothesis per line (plain UTF-8 text).  Lines matching [SKIPPED] or
[EMPTY] are ignored.
"""

import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# CA → VA contrastive feature map
# (data-driven from gplsi/CA-VA_alignment_test + manual validation)
# ---------------------------------------------------------------------------
CA_VA_FEATURES: Dict[str, str] = {
    # Determiners / proximal pronouns
    "aquesta":        "esta",
    "aquest":         "este",
    "aquestes":       "estes",
    "aquests":        "estos",
    # Possessives
    "seva":           "seua",
    "seves":          "seues",
    # Temporal / ordinal adjectives
    "darrer":         "últim",
    "darrers":        "últims",
    "darrera":        "última",
    # Verbs — infinitives
    "tenir":          "tindre",
    "obtenir":        "obtindre",
    "veure":          "vore",
    # Verbs — 3rd person singular present (inchoative)
    "segueix":        "seguix",
    "segueixen":      "seguixen",
    "requereix":      "requerix",
    "divideix":       "dividix",
    "constitueixen":  "constituïxen",
    "absorbeixen":    "absorbixen",
    # Lexical substitutions
    "nens":           "xiquets",
    "nen":            "xiquet",
    "nena":           "xiqueta",
    "nenes":          "xiquetes",
    "feina":          "faena",
    "ara":            "ara",       # same — kept for completeness
    "però":           "però",      # same
    "molt":           "molt",      # same
}

# Remove identity mappings
CA_VA_FEATURES = {k: v for k, v in CA_VA_FEATURES.items() if k != v}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def dialectal_score(
    hypotheses: List[str],
    label: str = "model",
) -> Tuple[float, Dict[str, Dict]]:
    """
    Compute the overall dialectal Valencian score and per-feature breakdown.

    Parameters
    ----------
    hypotheses : list of str
        Model output sentences (one per source sentence).
    label : str
        Name used in printed output.

    Returns
    -------
    score : float
        Overall VA rate = VA_hits / (VA_hits + CA_hits) across all features.
    per_feature : dict
        Detailed counts per CA↔VA feature pair.
    """
    valid = [h.lower() for h in hypotheses
             if h not in ("[SKIPPED]", "[EMPTY]", None, "")]
    corpus = " ".join(valid)

    per_feature: Dict[str, Dict] = {}
    total_va, total_ca = 0, 0

    for ca_form, va_form in CA_VA_FEATURES.items():
        va_hits = len(re.findall(r"\b" + re.escape(va_form) + r"\b", corpus))
        ca_hits = len(re.findall(r"\b" + re.escape(ca_form) + r"\b", corpus))
        total   = va_hits + ca_hits
        va_rate: Optional[float] = va_hits / total if total > 0 else None

        per_feature[ca_form] = {
            "va_form": va_form,
            "va_hits": va_hits,
            "ca_hits": ca_hits,
            "total":   total,
            "va_rate": round(va_rate, 4) if va_rate is not None else None,
        }
        total_va += va_hits
        total_ca += ca_hits

    overall = total_va / (total_va + total_ca) if (total_va + total_ca) > 0 else 0.0
    return round(overall, 4), per_feature


def print_report(label: str, score: float, per_feature: Dict[str, Dict]) -> None:
    """Pretty-print the dialectal score report."""
    print(f"\n{'='*60}")
    print(f"  Dialectal VA Score — {label}")
    print(f"{'='*60}")
    print(f"  Overall VA rate: {score:.1%}  "
          f"({sum(v['va_hits'] for v in per_feature.values())} VA hits / "
          f"{sum(v['total'] for v in per_feature.values())} total slots)\n")

    # Table header
    print(f"  {'CA form':<18} {'VA form':<18} {'VA hits':>8} {'CA hits':>8} {'VA rate':>9}")
    print(f"  {'-'*18} {'-'*18} {'-'*8} {'-'*8} {'-'*9}")

    for ca, info in sorted(per_feature.items(), key=lambda x: -(x[1]["total"])):
        if info["total"] == 0:
            continue
        rate_str = f"{info['va_rate']:.1%}" if info["va_rate"] is not None else "  —   "
        print(f"  {ca:<18} {info['va_form']:<18} {info['va_hits']:>8} "
              f"{info['ca_hits']:>8} {rate_str:>9}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the Dialectal Valencian Score for model hypotheses."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to a plain-text file with one hypothesis per line."
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Optional path to save a JSON report."
    )
    parser.add_argument(
        "--label", "-l", default="model",
        help="Label shown in the report (default: 'model')."
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    hypotheses = in_path.read_text(encoding="utf-8").splitlines()
    print(f"Loaded {len(hypotheses)} hypotheses from {in_path.name}")

    score, per_feature = dialectal_score(hypotheses, label=args.label)
    print_report(args.label, score, per_feature)

    if args.output:
        report = {
            "label":       args.label,
            "n_hypotheses": len(hypotheses),
            "overall_va_rate": score,
            "per_feature":  per_feature,
        }
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
