"""Genome-wide CRISPRi library design.

This module provides functions to design a CRISPRi guide RNA library
targeting all genes in a bacterial genome. It finds all PAM sites,
annotates them with gene information, predicts on-target activity,
evaluates off-targets, and ranks guides to select the best N per gene.
"""

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

from crisprbact.predict import predict, GUIDE_LEN, SCORE_BOUNDARIES, remove_GG_of_PAM, reshape_targets
from crisprbact.off_target_dict import get_off_dics, count_off_target_col, count_off_targets


# Bad seed sequences for E. coli MG1655: 5-nt PAM-proximal motifs that cause
# toxicity via off-target binding to essential gene promoters.
# Source: Cui et al. Nat Commun 2018; Rostain et al. NAR 2023 (doi:10.1093/nar/gkad170).
# These are strain- and species-specific — only applicable to E. coli MG1655
# and close derivatives.
DEFAULT_BAD_SEEDS = ["TATAG", "AAAGG", "GGTTA", "ACCCA", "AGGGG", "GATAT"]


def _auto_detect_badseeds(records):
    """Auto-detect bad seeds from genome organism annotation.

    Returns DEFAULT_BAD_SEEDS for E. coli, empty list for all other organisms.
    """
    organism = records[0].annotations.get("organism", "")
    if "Escherichia coli" in organism:
        print(
            f"Organism: {organism} — using default E. coli bad seeds: "
            f"{', '.join(DEFAULT_BAD_SEEDS)}"
        )
        return DEFAULT_BAD_SEEDS
    else:
        print(
            f"Organism: {organism or 'unknown'} — no known bad seeds for this "
            f"organism. Skipping bad seed filtering. "
            f"Pass badseeds=[] explicitly to silence this message, "
            f"or provide custom bad seeds for your organism."
        )
        return []


def find_all_targets(records, strand):
    """Find all NGG PAM targets in genome records.

    Parameters
    ----------
    records : list of SeqRecord
        Parsed genome records.
    strand : str
        "+" or "-" strand to search.

    Returns
    -------
    list of list
        Each element is [guide, target, recid, strand, pos].
    """
    targets = []
    for rec in records:
        L = len(rec.seq)
        if strand == "+":
            seq = rec.seq

            def pam_pos(p, _L=L):
                return (p - 1) % _L

        elif strand == "-":
            seq = rec.seq.reverse_complement()

            def pam_pos(p, _L=L):
                return (-p) % _L

        i = 21

        while True:
            p = seq.find("GG", start=i)

            if p == -1 or p + 18 > L:
                break

            target = str(seq[p - 7 : p + 18])
            guide = str(seq[p - 21 : p - 1])
            pos = pam_pos(p)
            targets.append([guide, target, rec.id, strand, pos])
            i = p + 1

    return targets


def annotate_targets(targets, records, keep_intergenic=False):
    """Annotate target DataFrame with gene information.

    For each gene feature in the genome records, finds targets whose PAM
    position falls within the gene and annotates them with gene name,
    locus_tag, orientation, and position within the gene.

    Parameters
    ----------
    targets : DataFrame
        Must have columns: recid, pos, strand.
    records : list of SeqRecord
        Parsed genome records.
    keep_intergenic : bool
        If False (default), return only targets within annotated genes and
        add a ``second_half_gene`` column.
        If True, return all rows with NaN gene columns for intergenic targets.

    Returns
    -------
    DataFrame
        Annotated with locus_tag, gene, gene_ori, targets_coding_strand.
        When keep_intergenic=False, also adds second_half_gene.
    """
    n = len(targets)
    locus_tags = np.full(n, pd.NA, dtype=object)
    genes = np.full(n, pd.NA, dtype=object)
    gene_oris = np.full(n, pd.NA, dtype=object)
    targets_coding = np.full(n, pd.NA, dtype=object)
    # For second_half_gene (only needed when dropping intergenic)
    gene_starts_fill = np.full(n, np.nan)
    gene_ends_fill = np.full(n, np.nan)

    recids = targets["recid"].to_numpy(dtype=object)
    positions = targets["pos"].to_numpy(dtype=float)
    strands = targets["strand"].to_numpy(dtype=object)

    for rec in tqdm(records, desc="Annotating genes"):
        features = [
            f for f in rec.features
            if f.type == "gene" and "locus_tag" in f.qualifiers
        ]
        if not features:
            continue

        rec_mask = recids == rec.id
        if not rec_mask.any():
            continue

        rec_idx = np.where(rec_mask)[0]
        rec_pos = positions[rec_idx]
        sort_order = np.argsort(rec_pos)
        sorted_idx = rec_idx[sort_order]
        sorted_pos = rec_pos[sort_order]

        f_starts = np.array([int(f.location.start) for f in features], dtype=float)
        f_ends = np.array([int(f.location.end) for f in features], dtype=float)
        f_gene_strands = np.array([f.location.strand for f in features])
        f_locus_tags = [f.qualifiers["locus_tag"][0] for f in features]
        f_gene_names = [f.qualifiers.get("gene", [None])[0] for f in features]

        los = np.searchsorted(sorted_pos, f_starts + 1, side="left")
        his = np.searchsorted(sorted_pos, f_ends - GUIDE_LEN, side="left")
        counts = his - los
        counts[counts < 0] = 0

        rec_strands = strands[sorted_idx]

        for g_idx in np.where(counts > 0)[0]:
            lo, hi = los[g_idx], his[g_idx]
            orig = sorted_idx[lo:hi]

            locus_tags[orig] = f_locus_tags[g_idx]
            genes[orig] = f_gene_names[g_idx]
            gene_oris[orig] = f_gene_strands[g_idx]
            gene_starts_fill[orig] = f_starts[g_idx]
            gene_ends_fill[orig] = f_ends[g_idx]

            g_strand = f_gene_strands[g_idx]
            row_strands = rec_strands[lo:hi]
            if g_strand == 1:
                targets_coding[orig[row_strands == "+"]] = False
                targets_coding[orig[row_strands == "-"]] = True
            elif g_strand == -1:
                targets_coding[orig[row_strands == "+"]] = True
                targets_coding[orig[row_strands == "-"]] = False

    result = targets.copy()
    result["locus_tag"] = locus_tags
    result["gene"] = genes
    result["gene_ori"] = gene_oris
    result["targets_coding_strand"] = targets_coding

    if not keep_intergenic:
        matched = np.array([v is not pd.NA for v in locus_tags])
        result = result[matched].reset_index(drop=True)

        # Compute second_half_gene for matched rows
        pos = result["pos"].values.astype(float)
        g_strand = result["gene_ori"].values
        g_start = gene_starts_fill[matched]
        g_end = gene_ends_fill[matched]
        g_half = (g_end - g_start) // 2
        tgt_strand = result["strand"].values

        plus_gene = g_strand == 1
        minus_gene = g_strand == -1

        second_half = np.zeros(len(result), dtype=bool)
        second_half |= plus_gene & (pos > g_start + g_half) & (pos < g_end - GUIDE_LEN)
        second_half |= minus_gene & (pos > g_start) & (pos < g_end - g_half)
        result["second_half_gene"] = second_half

    return result


def add_annotations(targets, records):
    """Annotate target DataFrame with gene information (intergenic targets dropped).

    Thin wrapper around :func:`annotate_targets` with ``keep_intergenic=False``.
    """
    return annotate_targets(targets, records, keep_intergenic=False)


def add_on_target_predictions(targets):
    """Add on-target activity predictions to target DataFrame.

    Uses the same linear model as the main crisprbact predict module.
    Expects a 'seq' column with the 25bp target sequence window.

    Parameters
    ----------
    targets : DataFrame
        Must have a 'seq' column containing 25bp target sequences.

    Returns
    -------
    DataFrame
        With added 'score' column and 'seq' column dropped.
    """
    X = reshape_targets(remove_GG_of_PAM(iter(targets["seq"])))
    targets["score"] = predict(X)
    return targets.drop(columns=["seq"])


def add_score_quartile(targets):
    """Classify targets into score quartiles using fixed boundaries.

    Uses the fixed boundaries from ``predict.SCORE_BOUNDARIES`` (computed
    from 1000 random sequences) so quartile labels are consistent across all
    pipelines (generate_library, map_library, etc.).

    Parameters
    ----------
    targets : DataFrame
        Must have a 'score' column.

    Returns
    -------
    DataFrame
        With added 'score_quartile' column (1=best, 4=worst).
    """
    q1_cut, q2_cut, q3_cut = SCORE_BOUNDARIES
    scores = targets["score"].values
    # searchsorted on a descending boundary array:
    # score >= q1_cut → quartile 1, q2_cut <= score < q1_cut → 2, etc.
    score_quartile = np.ones(len(scores), dtype=int)
    score_quartile[scores < q1_cut] = 2
    score_quartile[scores < q2_cut] = 3
    score_quartile[scores < q3_cut] = 4
    targets["score_quartile"] = score_quartile
    return targets


def add_off_targets(
    targets, records=None, ref_name="default", cache_dir="off_dics",
    drop_intermediate=True, off_dics=None,
):
    """Add off-target counts to target DataFrame.

    Parameters
    ----------
    targets : DataFrame
        Must have columns: guide, recid, pos.
    records : list of SeqRecord, optional
        Parsed genome records. Required when ``off_dics`` is None.
    ref_name : str
        Name for caching off-target dictionaries.
    cache_dir : str
        Directory for cached off-target dictionaries.
    drop_intermediate : bool
        If True (default), drop intermediate ±strand columns after summing.
    off_dics : dict, optional
        Pre-computed off-target dictionaries (from :func:`~crisprbact.off_target_dict.get_off_dics`).
        When provided, ``records``, ``ref_name``, and ``cache_dir`` are ignored.

    Returns
    -------
    DataFrame
        With added off-target count columns.
    """
    if off_dics is None:
        off_dics = get_off_dics(records, ref_name=ref_name, cache_dir=cache_dir)

    targets["ntargets_plus"] = count_off_target_col(targets, off_dics["off_plus"], 20)
    targets["ntargets_minus"] = count_off_target_col(targets, off_dics["off_minus"], 20)
    targets["ntargets"] = targets["ntargets_plus"] + targets["ntargets_minus"] + 1

    targets["noff_12_plus"] = count_off_target_col(targets, off_dics["off_plus"], 12)
    targets["noff_12_minus"] = count_off_target_col(targets, off_dics["off_minus"], 12)
    targets["noff_11_gene"] = count_off_target_col(targets, off_dics["off_11_gene"], 11)
    targets["noff_9_prom_plus"] = count_off_target_col(targets, off_dics["off_9_prom_plus"], 9)
    targets["noff_9_prom_minus"] = count_off_target_col(targets, off_dics["off_9_prom_minus"], 9)

    targets["noff_12"] = targets["noff_12_plus"] + targets["noff_12_minus"]
    targets["noff_9_prom"] = targets["noff_9_prom_plus"] + targets["noff_9_prom_minus"]

    if drop_intermediate:
        targets = targets.drop(columns=[
            "ntargets_plus", "ntargets_minus",
            "noff_12_plus", "noff_12_minus",
            "noff_9_prom_plus", "noff_9_prom_minus",
        ])
    return targets


def add_badseeds(targets, badseeds=None):
    """Flag targets with known bad seed sequences.

    Parameters
    ----------
    targets : DataFrame
        Must have a 'guide' column.
    badseeds : list of str, optional
        Seed sequences to flag. Defaults to DEFAULT_BAD_SEEDS.

    Returns
    -------
    DataFrame
        With added 'inbadseeds' boolean column.
    """
    if badseeds is None:
        badseeds = DEFAULT_BAD_SEEDS
    targets["inbadseeds"] = targets["guide"].apply(lambda g: str(g[-5:]) in badseeds)
    return targets


def rank_targets(targets):
    """Rank guides per gene by off-target count, score, and position.

    Guides are sorted by multiple criteria (fewer off-targets preferred,
    higher score preferred, first half of gene preferred) and assigned
    a per-gene rank.

    Parameters
    ----------
    targets : DataFrame
        Must have columns: locus_tag, ntargets, noff_12, noff_11_gene,
        noff_9_prom, inbadseeds, score_quartile, second_half_gene, score.

    Returns
    -------
    DataFrame
        With added 'gene_rank' column.
    """
    targets = targets.sort_values(
        [
            "locus_tag",
            "ntargets",
            "noff_12",
            "noff_11_gene",
            "noff_9_prom",
            "inbadseeds",
            "score_quartile",
            "second_half_gene",
            "score",
        ],
        ascending=[
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
        ],
    )

    targets["gene_rank"] = targets.groupby("locus_tag").cumcount() + 1
    return targets


def generate_library(
    ref_file,
    n=3,
    ref_name="default",
    cache_dir="off_dics",
    badseeds=None,
    output_csv=None,
    output_report=None,
):
    """Generate a CRISPRi guide RNA library for a bacterial genome.

    This is the main orchestrator function. It finds all PAM targets,
    annotates them with gene info, predicts on-target activity, evaluates
    off-targets, ranks guides, and returns the top N per gene.

    Parameters
    ----------
    ref_file : str or path-like
        Path to a GenBank format genome file.
    n : int
        Number of guides to select per gene.
    ref_name : str
        Name for the genome (used for caching off-target dicts).
        If "default", uses the first record's ID.
    cache_dir : str
        Directory for cached off-target dictionaries.
    badseeds : list of str, optional
        Bad seed sequences to flag.
    output_csv : str or path-like, optional
        Path to write the library as a CSV file. If None, no file is written.
    output_report : str or path-like, optional
        Path to write an HTML report. If None, no report is generated.

    Returns
    -------
    DataFrame
        Library of top N guides per gene with all annotations and scores.
    """
    records = list(SeqIO.parse(ref_file, "genbank"))
    if ref_name == "default":
        ref_name = records[0].id

    if badseeds is None:
        badseeds = _auto_detect_badseeds(records)

    print("Finding all targets...")
    targets_plus = find_all_targets(records, "+")
    targets_minus = find_all_targets(records, "-")

    targets = pd.DataFrame(
        targets_plus + targets_minus,
        columns=["guide", "seq", "recid", "strand", "pos"],
    )

    print("Adding annotations...")
    targets = add_annotations(targets, records)

    print("Keeping targets on the coding strand of genes...")
    targets = targets[targets.targets_coding_strand].copy()

    print("Adding on-target activity predictions...")
    targets = add_on_target_predictions(targets)
    targets = add_score_quartile(targets)

    print("Adding off-targets...")
    targets = add_off_targets(targets, records, ref_name=ref_name, cache_dir=cache_dir)

    print("Adding bad seeds...")
    targets = add_badseeds(targets, badseeds=badseeds)

    print("Ranking targets...")
    targets = rank_targets(targets)

    library = targets[targets.gene_rank <= n].copy()

    if output_csv is not None:
        library.to_csv(output_csv, index=False)
        print(f"Library written to {output_csv}")

    if output_report is not None:
        from crisprbact.visualize import generate_report

        generate_report(library, records, n, output_report, badseeds=badseeds)
        print(f"Report written to {output_report}")

    return library
