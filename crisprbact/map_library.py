"""Map an existing guide library against a new genome.

Locates each guide in the input CSV in the target genome, predicts on-target
activity, counts off-targets, and flags bad seeds. Useful for evaluating
whether a library designed for one strain will work on another.
"""

import os
import pickle
import re

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

from crisprbact.predict import GUIDE_LEN, SCORE_BOUNDARIES, UPSTREAM_CTX, DOWNSTREAM_CTX
from crisprbact.library import add_on_target_predictions, add_badseeds, annotate_targets, _auto_detect_badseeds
from crisprbact.off_target_dict import get_off_dics, count_off_targets


# IUPAC nucleotide codes → regex character classes
_IUPAC = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "R": "[AG]",
    "Y": "[CT]",
    "S": "[GC]",
    "W": "[AT]",
    "K": "[GT]",
    "M": "[AC]",
    "B": "[CGT]",
    "D": "[AGT]",
    "H": "[ACT]",
    "V": "[ACG]",
    "N": "[ACGT]",
}

def _pam_to_regex(pam):
    """Convert an IUPAC PAM string to a regex pattern.

    Parameters
    ----------
    pam : str
        PAM sequence using IUPAC codes (e.g. "NGG", "NGA").

    Returns
    -------
    str
        Regex pattern string (e.g. "[ACGT]GG").
    """
    return "".join(_IUPAC.get(c.upper(), c) for c in pam)


def build_pam_index(records, pam="NGG", guide_len=GUIDE_LEN):
    """Build a reverse lookup dict mapping guide sequences to PAM site info.

    Scans both strands of all records for PAM sites and returns a dictionary
    mapping each 20-nt guide to all its genomic occurrences. The sequence
    window is a 25bp target (guide[-6:] + PAM + 16bp downstream) compatible
    with ``add_on_target_predictions``.

    Position convention (matches ``find_all_targets`` and off-target dicts):
    - Forward strand: ``pam_pos = m.start()`` (position of PAM N in genome).
    - Reverse strand: ``pam_pos = L - m.start() - 1`` (revcomp → genome coords).

    Parameters
    ----------
    records : list of SeqRecord
        Parsed genome records.
    pam : str
        PAM sequence (IUPAC), default "NGG".
    guide_len : int
        Guide length in nucleotides (default 20).

    Returns
    -------
    dict
        ``{guide_seq: [(recid, strand, pam_pos, seq_window), ...]}``
    """
    pam_len = len(pam)
    pam_re = re.compile("(?=" + _pam_to_regex(pam) + ")")
    index = {}

    for rec in records:
        L = len(rec.seq)
        fwd_seq = str(rec.seq).upper()
        rev_seq = str(rec.seq.reverse_complement()).upper()

        for strand, seq in [("+", fwd_seq), ("-", rev_seq)]:
            for m in pam_re.finditer(seq):
                # m.start() = position of first PAM character (N in NGG)
                pam_start = m.start()
                guide_start = pam_start - guide_len
                left = pam_start - UPSTREAM_CTX           # upstream context
                right = pam_start + pam_len + DOWNSTREAM_CTX  # PAM + downstream

                if guide_start < 0 or right > L:
                    continue

                guide = seq[guide_start:pam_start]
                seq_window = seq[left:right]

                if strand == "+":
                    pam_pos = pam_start
                else:
                    # Convert revcomp position to genome (forward) coordinates.
                    # N at pam_start in revcomp → position L - pam_start - 1 in genome.
                    pam_pos = L - pam_start - 1

                entry = (rec.id, strand, pam_pos, seq_window)
                if guide in index:
                    index[guide].append(entry)
                else:
                    index[guide] = [entry]

    return index


def _annotate_with_genes(df, records):
    """Add gene annotation columns, keeping intergenic rows as NaN."""
    return annotate_targets(df, records, keep_intergenic=True)


def _apply_score_quartile(scores):
    """Classify on-target scores into quartiles using fixed boundaries.

    Boundaries: Q1 >= 0.4 > Q2 >= -0.08 > Q3 >= -0.59 > Q4.
    NaN scores (guides with no genomic match) → pd.NA.

    Parameters
    ----------
    scores : iterable
        Predicted on-target scores (float or NaN).

    Returns
    -------
    pandas.array
        Nullable integer array of quartile labels (1–4) or pd.NA.
    """
    q1, q2, q3 = SCORE_BOUNDARIES
    result = []
    for s in scores:
        if pd.isna(s):
            result.append(pd.NA)
        elif s >= q1:
            result.append(1)
        elif s >= q2:
            result.append(2)
        elif s >= q3:
            result.append(3)
        else:
            result.append(4)
    return pd.array(result, dtype=pd.Int64Dtype())


def map_library(
    guides,
    ref_file,
    pam="NGG",
    cache_dir="off_dics",
    ref_name=None,
    badseeds=None,
    pam_index_cache=None,
    output_csv=None,
    output_report=None,
):
    """Map a guide library against a genome and evaluate guide properties.

    Locates each guide from ``guides`` in ``ref_file``, predicts on-target
    activity, counts off-targets using dictionary-based seed matching, and flags
    bad seeds. Returns one row per genomic match; guides with no match appear
    once with NaN genomic columns.

    Note: off-target analysis always uses NGG-based dictionaries regardless of
    the ``pam`` parameter (documented limitation).

    Parameters
    ----------
    guides : str, path-like, DataFrame, or list
        Input guides. Can be a CSV file path with a ``guide`` column,
        a DataFrame with a ``guide`` column, or a list/tuple of guide
        sequences.
    ref_file : str or path-like
        Path to GenBank genome file.
    pam : str
        PAM sequence (IUPAC), default "NGG".
    cache_dir : str
        Directory for cached off-target dictionaries.
    ref_name : str or None
        Genome name for off-target dict caching. If None, uses first record ID.
    badseeds : list of str or None
        Bad seed sequences. If None, auto-detects: E. coli → DEFAULT_BAD_SEEDS,
        other organisms → empty list.
    pam_index_cache : str or path-like or None
        Path to a pickle file for caching the PAM index. If the file exists it
        is loaded directly (skipping genome scanning). If the file does not
        exist the index is built and saved. Pass None to disable caching.
    output_csv : str or path-like or None
        Path to write output CSV. If None, not written.
    output_report : str or path-like or None
        Path to write an HTML report. If None, no report is generated.

    Returns
    -------
    DataFrame
        Columns: guide, n_matches, recid, strand, pos, locus_tag, gene,
        gene_ori, targets_coding_strand, on_target_score, score_quartile,
        ntargets, noff_12, noff_11_gene, noff_9_prom, inbadseeds.
    """
    # --- Load inputs ---
    if isinstance(guides, pd.DataFrame):
        guides_df = guides
    elif isinstance(guides, (list, tuple)):
        guides_df = pd.DataFrame({"guide": guides})
    else:
        guides_df = pd.read_csv(guides)
    if "guide" not in guides_df.columns:
        raise ValueError("Input CSV must have a 'guide' column.")
    guides = guides_df["guide"].str.upper().tolist()

    records = list(SeqIO.parse(ref_file, "genbank"))
    if not records:
        raise ValueError(f"No records found in {ref_file}")

    if ref_name is None:
        ref_name = records[0].id

    if badseeds is None:
        badseeds = _auto_detect_badseeds(records)

    # --- Build or load PAM index ---
    if pam_index_cache is not None:
        cache_file = str(pam_index_cache)
        if os.path.exists(cache_file):
            print(f"Loading PAM index from cache: {cache_file}")
            with open(cache_file, "rb") as fh:
                pam_index = pickle.load(fh)
        else:
            print("Building PAM index...")
            pam_index = build_pam_index(records, pam=pam)
            with open(cache_file, "wb") as fh:
                pickle.dump(pam_index, fh, pickle.HIGHEST_PROTOCOL)
            print(f"PAM index cached to: {cache_file}")
    else:
        print("Building PAM index...")
        pam_index = build_pam_index(records, pam=pam)

    # --- Expand guides to rows (one per match, or one NaN row if no match) ---
    rows = []
    for guide in guides:
        matches = pam_index.get(guide, [])
        n_matches = len(matches)
        if matches:
            for recid, strand, pos, seq_window in matches:
                rows.append(
                    {
                        "guide": guide,
                        "n_matches": n_matches,
                        "recid": recid,
                        "strand": strand,
                        "pos": pos,
                        "seq": seq_window,
                    }
                )
        else:
            rows.append(
                {
                    "guide": guide,
                    "n_matches": 0,
                    "recid": pd.NA,
                    "strand": pd.NA,
                    "pos": pd.NA,
                    "seq": pd.NA,
                }
            )

    df = pd.DataFrame(rows).reset_index(drop=True)

    # --- Annotate matched rows with gene information ---
    print("Annotating with gene information...")
    matched_mask = df["recid"].notna()
    df["locus_tag"] = pd.NA
    df["gene"] = pd.NA
    df["gene_ori"] = pd.NA
    df["targets_coding_strand"] = pd.NA

    if matched_mask.any():
        matched_df = _annotate_with_genes(
            df[matched_mask].reset_index(drop=True), records
        )
        for col in ["locus_tag", "gene", "gene_ori", "targets_coding_strand"]:
            df.loc[matched_mask, col] = matched_df[col].values

    # --- On-target prediction (only for rows with a genomic match) ---
    print("Predicting on-target activity...")
    df["on_target_score"] = np.nan

    if matched_mask.any():
        pred_df = df[matched_mask].copy().reset_index(drop=True)
        pred_df = add_on_target_predictions(pred_df)  # adds 'score' column
        df.loc[matched_mask, "on_target_score"] = pred_df["score"].values

    # --- Score quartile using fixed boundaries ---
    df["score_quartile"] = _apply_score_quartile(df["on_target_score"])

    # --- Off-target counts (computed once per unique guide) ---
    print("Computing off-targets...")
    off_dics = get_off_dics(records, ref_name=ref_name, cache_dir=cache_dir)

    df["ntargets"] = df["n_matches"]
    df["noff_12"] = 0
    df["noff_11_gene"] = 0
    df["noff_9_prom"] = 0

    # Precompute first-match position per guide for self-exclusion (O(n) once)
    first_match = (
        df[df["recid"].notna()]
        .drop_duplicates("guide")
        .set_index("guide")[["pos", "recid"]]
    )

    unique_guides = df["guide"].unique()
    guide_noff12 = {}
    guide_noff11 = {}
    guide_noff9 = {}
    for guide in tqdm(unique_guides, desc="Off-target analysis"):
        if guide in first_match.index:
            pos = first_match.at[guide, "pos"]
            recid = first_match.at[guide, "recid"]
        else:
            pos = None
            recid = None

        guide_noff12[guide] = count_off_targets(
            guide, pos, recid, off_dics["off_plus"], 12
        ) + count_off_targets(
            guide, pos, recid, off_dics["off_minus"], 12
        )
        guide_noff11[guide] = count_off_targets(
            guide, pos, recid, off_dics["off_11_gene"], 11
        )
        guide_noff9[guide] = count_off_targets(
            guide, pos, recid, off_dics["off_9_prom_plus"], 9
        ) + count_off_targets(
            guide, pos, recid, off_dics["off_9_prom_minus"], 9
        )

    # Vectorized assignment: one map() per column instead of a per-guide loop
    df["noff_12"] = df["guide"].map(guide_noff12).fillna(0).astype(int)
    df["noff_11_gene"] = df["guide"].map(guide_noff11).fillna(0).astype(int)
    df["noff_9_prom"] = df["guide"].map(guide_noff9).fillna(0).astype(int)

    # --- Flag bad seeds ---
    print("Flagging bad seeds...")
    df = add_badseeds(df, badseeds=badseeds)

    # --- Select and order output columns ---
    output_cols = [
        "guide",
        "n_matches",
        "recid",
        "strand",
        "pos",
        "locus_tag",
        "gene",
        "gene_ori",
        "targets_coding_strand",
        "on_target_score",
        "score_quartile",
        "ntargets",
        "noff_12",
        "noff_11_gene",
        "noff_9_prom",
        "inbadseeds",
    ]
    df = df[output_cols]

    if output_csv is not None:
        df.to_csv(output_csv, index=False)
        print(f"Library map written to {output_csv}")

    if output_report is not None:
        from crisprbact.visualize import generate_map_report

        generate_map_report(df, records, output_report)
        print(f"Report written to {output_report}")

    return df
