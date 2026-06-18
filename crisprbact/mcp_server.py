"""MCP server exposing CRISPRbact prediction and off-target tools."""

from __future__ import annotations

import os
import tempfile
from typing import Optional

from mcp.server.fastmcp import FastMCP
from Bio import SeqIO

from crisprbact.predict import on_target_predict, GUIDE_LEN
from crisprbact.off_target import compute_off_target_df, extract_records, extract_features

mcp = FastMCP(
    "crisprbact",
    instructions=(
        "CRISPRbact provides tools for designing CRISPRi guide RNAs in bacteria. "
        "Use predict_guides to find and score guide RNAs for a target DNA sequence. "
        "Higher scores indicate stronger predicted CRISPRi knockdown activity. "
        "Use find_off_targets to check a specific guide against a genome."
    ),
)


def _compact_off_targets(off_targets_per_seed: list[dict]) -> list[dict]:
    """Collapse full off-target lists into counts per seed size."""
    return [
        {"seed_size": entry["seed_size"], "count": len(entry["off_targets"])}
        for entry in off_targets_per_seed
    ]


@mcp.tool()
def predict_guides(
    sequence: str,
    genome_path: Optional[str] = None,
    seed_sizes: Optional[list[int]] = None,
    top_n: int = 10,
    detailed_off_targets: bool = False,
) -> list[dict]:
    """Predict CRISPRi guide RNAs for a target DNA sequence.

    Finds all NGG PAM sites on the coding strand of the input sequence,
    predicts on-target activity using a linear model, and optionally
    computes off-targets against a genome.

    Higher scores indicate stronger predicted CRISPRi knockdown activity.

    By default off-targets are returned as counts per seed size to keep the
    response compact.  For full positional detail on a specific guide, use
    the find_off_targets tool instead.

    Args:
        sequence: DNA sequence to target (ATGC only). Guides will silence
            genes encoded on the coding strand of this sequence.
        genome_path: Optional path to a GenBank file for off-target analysis.
        seed_sizes: Seed sizes for off-target matching.
            Defaults to [8, 9, 10, 11, 12, 20].
        top_n: Number of top guides to return, sorted by score (best first).
            Defaults to 10. Set to 0 to return all guides.
        detailed_off_targets: If False (default), off-targets are summarized
            as counts per seed size. If True, full positional detail is
            included (warning: can be very large for whole-genome searches).

    Returns:
        List of guide RNA dicts sorted by score (highest first) with keys:
        target_id, guide, guide_start, guide_end, pam_pos, score,
        off_targets_per_seed.
    """
    genome = None
    if genome_path:
        genome = list(SeqIO.parse(genome_path, "genbank"))
    results = on_target_predict(sequence, genome=genome, seed_sizes=seed_sizes)
    results.sort(key=lambda g: g["score"], reverse=True)
    if top_n > 0:
        results = results[:top_n]
    if not detailed_off_targets:
        for guide in results:
            guide["off_targets_per_seed"] = _compact_off_targets(
                guide["off_targets_per_seed"]
            )
    return results


@mcp.tool()
def find_off_targets(
    guide: str,
    genome_path: str,
    seed_size: int = 12,
) -> list[dict]:
    """Find off-target sites for a specific guide RNA in a genome.

    Searches for seed matches of the guide in the genome and annotates
    hits with overlapping genomic features (CDS, ncRNA, rRNA, tRNA).

    Args:
        guide: 20-nt guide RNA sequence (ATGC only, 5'→3').
        genome_path: Path to a GenBank genome file.
        seed_size: Length of the 3' seed to match (default 12).

    Returns:
        List of off-target hit dicts with position, strand, record ID,
        longest perfect match, PAM sequence, and overlapping features.
    """
    if len(guide) != GUIDE_LEN:
        raise ValueError(f"Guide must be {GUIDE_LEN} nt, got {len(guide)}")
    guide = guide.upper()

    records = list(SeqIO.parse(genome_path, "genbank"))
    if not records:
        raise ValueError(f"No records found in {genome_path}")
    feature_df = extract_features(records)

    off_df = compute_off_target_df(guide, seed_size, records, feature_df)
    if off_df is None or off_df.empty:
        return []

    results = []
    for _, row in off_df.iterrows():
        hit = {
            "start": int(row["start"]),
            "end": int(row["end"]),
            "pampos": int(row["pampos"]),
            "strand": row["strand"],
            "recid": row["recid"],
            "longest_perfect_match": row["longest_perfect_match"],
            "pam_seq": row["pam_seq"],
        }
        feats = row["features"]
        if len(feats) > 0:
            f = feats[0]
            hit["feature_type"] = f.type
            hit["feature_start"] = int(f.location.start)
            hit["feature_end"] = int(f.location.end)
            hit["feature_strand"] = f.location.strand
            for k, vals in f.qualifiers.items():
                if k != "translation":
                    hit[f"feature_{k}"] = "::".join(vals)
        results.append(hit)
    return results


def main():
    mcp.run()


if __name__ == "__main__":
    main()
