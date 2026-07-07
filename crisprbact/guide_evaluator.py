import pandas as pd
from Bio import SeqIO

from crisprbact.library import (
    find_all_targets,
    add_annotations,
    add_on_target_predictions,
    add_score_quartile,
    add_off_targets,
    add_badseeds,
)


def find_selected_targets(ref_file, guide_list):
    """
    Find only the user-provided guides in a genome.

    Parameters
    ----------
    ref_file : str
        Path to GenBank file.

    guide_list : list[str]
        List of guide sequences.

    Returns
    -------
    DataFrame
        Columns:
        guide
        seq
        recid
        strand
        pos
    """

    # Convert guides to uppercase for matching
    guide_list = [
        g.strip().upper()
        for g in guide_list
        if g.strip()
    ]

    # Read genome
    records = list(SeqIO.parse(ref_file, "genbank"))

    selected = []

    # Search both strands
    for strand in ["+", "-"]:

        targets = find_all_targets(records, strand)

        for guide, seq, recid, strand, pos in targets:

            if guide.upper() in guide_list:

                selected.append([
                    guide,
                    seq,
                    recid,
                    strand,
                    pos
                ])

    return pd.DataFrame(
        selected,
        columns=[
            "guide",
            "seq",
            "recid",
            "strand",
            "pos"
        ]
    )


def evaluate_guides(ref_file, guide_list):
    """
    Evaluate user-provided guide RNAs using the CRISPRbact pipeline.
    """

    # Preserve original user input
    original_guides = [
        g.strip().upper()
        for g in guide_list
        if g.strip()
    ]

    # Find guides in genome
    targets = find_selected_targets(ref_file, original_guides)

    # Read genome
    records = list(SeqIO.parse(ref_file, "genbank"))

    if not targets.empty:
        targets = add_annotations(targets, records)
        targets = add_on_target_predictions(targets)
        targets = add_score_quartile(targets)
        targets = add_off_targets(targets, records)
        targets = add_badseeds(targets)

        targets["Status"] = "Found"

    else:
        targets = pd.DataFrame()

    # -----------------------------------
    # Add guides that were not found
    # -----------------------------------

    found_guides = set(targets["guide"]) if not targets.empty else set()

    missing = []

    for guide in original_guides:

        if guide not in found_guides:

            missing.append({
                "guide": guide,
                "Status": "Not Found"
            })

    if missing:

        missing_df = pd.DataFrame(missing)

        targets = pd.concat(
            [targets, missing_df],
            ignore_index=True,
            sort=False
        )

    # Preserve original order

    targets["guide"] = pd.Categorical(
        targets["guide"],
        categories=original_guides,
        ordered=True
    )

    targets = targets.sort_values("guide").reset_index(drop=True)
    targets = add_recommendation(targets)

    return targets

def add_recommendation(targets):
    """
    Recommend the best guide using the same priorities
    as CRISPRbact's ranking algorithm.
    """

    if targets.empty:
        return targets

    # Initialize
    targets["Recommendation"] = "✅ Good"

    # Ignore guides not found
    mask = targets["Status"] == "Found"

    found = targets[mask].copy()

    if found.empty:
        return targets

    # Same priorities as CRISPRbact
    found = found.sort_values(
        [
            "ntargets",
            "noff_12",
            "noff_11_gene",
            "noff_9_prom",
            "inbadseeds",
            "score_quartile",
            "score",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True,
            True,
            False,
        ],
    )

    # Best guide
    best = found.index[0]

    targets.loc[best, "Recommendation"] = "⭐ Recommended"

    # Bad guides

    bad = (
        (targets["inbadseeds"] == True)
        | (targets["score_quartile"] == 4)
    )

    targets.loc[bad, "Recommendation"] = "❌ Not Recommended"

    # Missing guides

    targets.loc[
        targets["Status"] == "Not Found",
        "Recommendation"
    ] = "❌ Not Found"

    return targets