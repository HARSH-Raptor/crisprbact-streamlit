"""Multi-genome CRISPRi core library design.

This module orchestrates the design of a CRISPRi guide RNA library targeting
genes conserved across multiple bacterial strains (core genome). Based on the
EcoCG strategy described in Rousset et al. 2021 (Nat Microbiol).
"""

import glob
import os
import shutil
import tempfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

from crisprbact.library import (
    add_badseeds,
    add_on_target_predictions,
    add_off_targets,
    _auto_detect_badseeds,
)
from crisprbact.off_target_dict import get_off_dics, count_off_targets
from crisprbact.pangenome import (
    extract_proteins_to_fasta,
    run_mmseqs2_easy_cluster,
    parse_cluster_tsv,
    find_core_families,
)
from crisprbact.predict import GUIDE_LEN


def _find_coding_guides_in_gene(rec, feature, fwd_seq=None, rev_seq=None):
    """Find NGG PAM guides targeting the coding strand of a single gene feature.

    Returns a list of [guide, target, recid, strand, pos] rows with
    positions in full genome coordinates (compatible with off-target dicts).

    Parameters
    ----------
    rec : SeqRecord
        The full genome record containing the gene.
    feature : SeqFeature
        The gene feature (CDS, gene, etc.).
    fwd_seq : str or None
        Precomputed ``str(rec.seq)``. Computed on demand if None.
    rev_seq : str or None
        Precomputed ``str(rec.seq.reverse_complement())``. Computed on demand if None.

    Returns
    -------
    list of list
        Each element is [guide, target, recid, strand_searched, pos].
    """
    gene_start = int(feature.location.start)
    gene_end = int(feature.location.end)
    gene_strand = feature.location.strand
    L = len(rec.seq)
    results = []

    # CRISPRi targets the coding strand: guides are designed from the template.
    # For + genes (strand=1): search revcomp for NGG (== search + strand for CCN)
    # For - genes (strand=-1): search + strand for NGG
    if gene_strand == 1:
        search_strand = "-"
        seq = rev_seq if rev_seq is not None else str(rec.seq.reverse_complement())
        # pos = (-p) % L = L - p must fall in [gene_start, gene_end - GUIDE_LEN]
        # => p in [L - gene_end + GUIDE_LEN, L - gene_start]
        p_min = max(21, L - gene_end + GUIDE_LEN)
        p_max = min(L - 18, L - gene_start)

        def pam_pos(p):
            return (-p) % L

    elif gene_strand == -1:
        search_strand = "+"
        seq = fwd_seq if fwd_seq is not None else str(rec.seq)
        # pos = (p - 1) % L must fall in [gene_start, gene_end - GUIDE_LEN]
        # => p in [gene_start + 1, gene_end - GUIDE_LEN + 1]
        p_min = max(21, gene_start + 1)
        p_max = min(L - 18, gene_end - GUIDE_LEN + 1)

        def pam_pos(p):
            return (p - 1) % L

    else:
        return results

    i = p_min
    while i <= p_max:
        p = seq.find("GG", i, p_max + 2)
        if p == -1 or p > p_max:
            break
        target = seq[p - 7 : p + 18]
        guide = seq[p - 21 : p - 1]
        pos = pam_pos(p)
        results.append([guide, target, rec.id, search_strand, pos])
        i = p + 1

    return results


def find_guides_for_family(family_members, all_records,
                           lt_to_feat_per_genome=None, seq_cache=None):
    """Find NGG PAM guides targeting all members of a core gene family.

    For each (genome_id, locus_tag) pair, locates the corresponding gene
    feature and finds guides targeting its coding strand.

    Parameters
    ----------
    family_members : dict
        ``{genome_id: [locus_tags]}`` as returned by :func:`find_core_families`.
    all_records : dict
        ``{genome_id: list of SeqRecord}`` — parsed genome records per genome.
    lt_to_feat_per_genome : dict or None
        Precomputed ``{genome_id: {locus_tag: (rec, feature)}}`` lookup.
        Built on demand (per genome) if None.
    seq_cache : dict or None
        Precomputed ``{rec_id: {"fwd": str, "rev": str}}`` sequence strings.
        Sequences are computed on demand if None.

    Returns
    -------
    DataFrame
        Columns: ``guide``, ``recid``, ``strand``, ``pos``,
        ``genome_id``, ``locus_tag``, ``score``.
        Empty DataFrame if no guides found.
    """
    rows = []
    for genome_id, locus_tags in family_members.items():
        records = all_records[genome_id]

        # Use precomputed locus_tag lookup if available, else build it now
        if lt_to_feat_per_genome is not None:
            lt_to_feat = lt_to_feat_per_genome[genome_id]
        else:
            lt_to_feat = {}
            for rec in records:
                for feat in rec.features:
                    if feat.type in ("CDS", "gene", "ncRNA", "rRNA", "tRNA"):
                        lt = feat.qualifiers.get("locus_tag", [None])[0]
                        if not lt:
                            continue
                        if lt not in lt_to_feat:
                            lt_to_feat[lt] = (rec, feat)
                        elif feat.type != "gene" and lt_to_feat[lt][1].type == "gene":
                            lt_to_feat[lt] = (rec, feat)

        for locus_tag in locus_tags:
            if locus_tag not in lt_to_feat:
                continue
            rec, feat = lt_to_feat[locus_tag]

            # Use precomputed sequence strings if available
            if seq_cache is not None and rec.id in seq_cache:
                fwd_seq = seq_cache[rec.id]["fwd"]
                rev_seq = seq_cache[rec.id]["rev"]
            else:
                fwd_seq = rev_seq = None

            targets = _find_coding_guides_in_gene(
                rec, feat,
                fwd_seq=fwd_seq, rev_seq=rev_seq,
            )
            for guide, seq_win, recid, strand, pos in targets:
                rows.append({
                    "guide": guide,
                    "seq": seq_win,
                    "recid": recid,
                    "strand": strand,
                    "pos": pos,
                    "genome_id": genome_id,
                    "locus_tag": locus_tag,
                })

    if not rows:
        return pd.DataFrame(
            columns=["guide", "recid", "strand", "pos",
                     "genome_id", "locus_tag", "score"]
        )

    df = pd.DataFrame(rows)
    df = add_on_target_predictions(df)
    return df


def compute_guide_coverage(candidates_df, n_genomes):
    """Add coverage columns to candidates DataFrame.

    For each unique guide, counts how many distinct genomes have a match and
    computes the coverage fraction.

    Parameters
    ----------
    candidates_df : DataFrame
        Must have ``guide`` and ``genome_id`` columns.
    n_genomes : int
        Total number of genomes in the study.

    Returns
    -------
    DataFrame
        With added columns ``n_covered`` and ``coverage``.
    """
    cov = (
        candidates_df.groupby("guide")["genome_id"]
        .nunique()
        .reset_index()
        .rename(columns={"genome_id": "n_covered"})
    )
    cov["coverage"] = cov["n_covered"] / n_genomes
    return candidates_df.merge(cov, on="guide", how="left")


def preselect_by_coverage(candidates_df, n_genomes, min_guides=5):
    """Filter candidates to those with near-maximal genome coverage.

    Iteratively lowers the coverage threshold from the maximum observed value
    until at least ``min_guides`` unique guides remain.

    Parameters
    ----------
    candidates_df : DataFrame
        Must have ``coverage`` column (added by :func:`compute_guide_coverage`).
    n_genomes : int
        Total number of genomes.
    min_guides : int
        Minimum number of distinct guide sequences to retain.

    Returns
    -------
    DataFrame
        Filtered to guides meeting the coverage threshold.
    """
    unique_guides = candidates_df.drop_duplicates("guide")[["guide", "coverage"]]
    max_coverage = unique_guides["coverage"].max()

    threshold = max_coverage
    # Step size: one genome at a time
    step = 1.0 / n_genomes

    min_threshold = max(step, max_coverage / 2)

    while threshold > min_threshold:
        kept = unique_guides[unique_guides["coverage"] >= threshold]
        if len(kept) >= min_guides:
            break
        threshold -= step

    kept_guides = unique_guides[unique_guides["coverage"] >= threshold]["guide"]
    result = candidates_df[candidates_df["guide"].isin(kept_guides)]

    # Rescue: if still too few, include everything covering >= 1 genome
    if result["guide"].nunique() < min_guides:
        result = candidates_df

    return result.copy()


def compute_multiGenome_off_targets(candidates_df, off_dics_per_genome, guide_positions):
    """Compute cross-genome off-target scores for candidate guides.

    For each guide, computes the fraction of genomes that have at least one
    off-target hit (11-nt seed in gene coding strands, or 9-nt seed in
    promoter regions).

    Parameters
    ----------
    candidates_df : DataFrame
        Must have ``guide`` and ``genome_id`` columns.
    off_dics_per_genome : dict
        ``{genome_id: off_dics_dict}`` where ``off_dics_dict`` is the result
        of :func:`~crisprbact.off_target_dict.get_off_dics`.
    guide_positions : dict
        ``{guide: {genome_id: (recid, pos)}}`` — known match positions per genome.

    Returns
    -------
    DataFrame
        Original DataFrame with added columns ``off_11_gene_score`` and
        ``off_9_prom_score`` (fraction of genomes with ≥1 hit).
    """
    genome_ids = list(off_dics_per_genome.keys())
    n_genomes = len(genome_ids)

    unique_guides = candidates_df["guide"].unique()
    off_11_scores = {}
    off_9_scores = {}

    for guide in unique_guides:
        guide_pos = guide_positions.get(guide, {})
        n_11_hit = 0
        n_9_hit = 0
        for genome_id in genome_ids:
            off_dics = off_dics_per_genome[genome_id]
            pos_info = guide_pos.get(genome_id)
            if pos_info is not None:
                recid, pos = pos_info
            else:
                recid, pos = None, None

            n11 = count_off_targets(
                guide, pos, recid, off_dics["off_11_gene"], 11
            )
            if n11 > 0:
                n_11_hit += 1

            n9p = count_off_targets(
                guide, pos, recid, off_dics["off_9_prom_plus"], 9
            )
            n9m = count_off_targets(
                guide, pos, recid, off_dics["off_9_prom_minus"], 9
            )
            if n9p + n9m > 0:
                n_9_hit += 1

        off_11_scores[guide] = n_11_hit / n_genomes
        off_9_scores[guide] = n_9_hit / n_genomes

    candidates_df = candidates_df.copy()
    candidates_df["off_11_gene_score"] = candidates_df["guide"].map(off_11_scores)
    candidates_df["off_9_prom_score"] = candidates_df["guide"].map(off_9_scores)
    return candidates_df


def compute_global_penalty_score(candidates_df):
    """Add Global Penalty Score (GPS) column to candidates DataFrame.

    GPS = off_score_norm + eff_score_norm + cov_score, where:
    - off_score = off_11_gene_score + off_9_prom_score (normalized 0–1)
    - eff_score = 1 - score (normalized 0–1)
    - cov_score = 1 - coverage (already in 0–1)

    All normalization is min-max within the current set of candidates.
    Lower GPS = better guide.

    Parameters
    ----------
    candidates_df : DataFrame
        Must have columns ``off_11_gene_score``, ``off_9_prom_score``,
        ``score``, ``coverage``.

    Returns
    -------
    DataFrame
        With added ``gps`` column.
    """
    df = candidates_df.copy()

    def minmax(series):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - mn) / (mx - mn)

    # One row per guide for normalization
    per_guide = df.drop_duplicates("guide").set_index("guide")
    off_raw = per_guide["off_11_gene_score"] + per_guide["off_9_prom_score"]
    off_norm = minmax(off_raw)
    eff_raw = 1 - per_guide["score"]
    eff_norm = minmax(eff_raw)
    cov_score = 1 - per_guide["coverage"]

    gps_per_guide = (off_norm + eff_norm + cov_score).rename("gps")
    df = df.merge(gps_per_guide.reset_index(), on="guide", how="left")
    return df


def select_top_guides(candidates_df, n=3, coverage_threshold=0.8, n_genomes=None):
    """Select the top N guides by GPS, with optional 4th-guide rescue.

    Deduplicates to one row per unique guide (taking the row with best score
    as representative), sorts by GPS ascending, and selects the top ``n``.

    If the collective coverage of the top ``n`` guides is below
    ``coverage_threshold``, an additional (n+1-th) guide is added.

    Parameters
    ----------
    candidates_df : DataFrame
        Must have columns ``guide``, ``gps``, ``score``, ``coverage``,
        ``genome_id``.
    n : int
        Number of guides to select.
    coverage_threshold : float
        Minimum collective coverage for the rescue heuristic.
    n_genomes : int or None
        Total genomes; used to compute collective coverage if provided.

    Returns
    -------
    DataFrame
        Selected guides (deduplicated, one row per guide).
    """
    # Deduplicate: one row per guide, best score wins
    per_guide = (
        candidates_df.sort_values("score", ascending=False)
        .drop_duplicates("guide")
        .sort_values("gps", ascending=True)
        .reset_index(drop=True)
    )

    selected = per_guide.head(n).copy()

    # 4th-guide rescue: compute collective coverage
    if n_genomes is not None and coverage_threshold is not None and len(per_guide) > n:
        # Covered genomes: union of genome_ids with a match for each selected guide
        selected_guides = set(selected["guide"])
        covered_genomes = set(
            candidates_df[candidates_df["guide"].isin(selected_guides)]["genome_id"]
        )
        collective_coverage = len(covered_genomes) / n_genomes
        if collective_coverage < coverage_threshold:
            extra = per_guide.iloc[[n]]
            selected = pd.concat([selected, extra], ignore_index=True)

    return selected


def _build_guide_positions(candidates_df):
    """Build guide → {genome_id → (recid, pos)} mapping.

    Takes the first occurrence of each (guide, genome_id) pair.

    Parameters
    ----------
    candidates_df : DataFrame
        Must have columns ``guide``, ``genome_id``, ``recid``, ``pos``.

    Returns
    -------
    dict
        ``{guide: {genome_id: (recid, pos)}}``
    """
    dedup = candidates_df.drop_duplicates(["guide", "genome_id"])
    guide_positions = {}
    for g, gid, rid, pos in zip(
        dedup["guide"].to_numpy(),
        dedup["genome_id"].to_numpy(),
        dedup["recid"].to_numpy(),
        dedup["pos"].to_numpy(),
    ):
        if g not in guide_positions:
            guide_positions[g] = {}
        guide_positions[g][gid] = (rid, pos)
    return guide_positions


def _family_gene_names(core_families, all_records,
                       feature_types=("CDS", "gene")):
    """Return the most common gene name for each family across all member locus_tags.

    Parameters
    ----------
    core_families : dict
        ``{family_id: {genome_id: [locus_tags]}}``
    all_records : dict
        ``{genome_id: list of SeqRecord}``
    feature_types : tuple of str
        Feature types to scan for gene names.

    Returns
    -------
    dict
        ``{family_id: gene_name_or_None}``
    """
    # Build (genome_id, locus_tag) → gene_name lookup
    locus_gene = {}
    for genome_id, records in all_records.items():
        for rec in records:
            for feat in rec.features:
                if feat.type in feature_types:
                    lt = feat.qualifiers.get("locus_tag", [None])[0]
                    gene = feat.qualifiers.get("gene", [None])[0]
                    if lt and gene:
                        locus_gene[(genome_id, lt)] = gene

    result = {}
    for fam_id, members in core_families.items():
        names = [
            locus_gene[(gid, lt)]
            for gid, lts in members.items()
            for lt in lts
            if (gid, lt) in locus_gene
        ]
        result[fam_id] = Counter(names).most_common(1)[0][0] if names else None
    return result


RNA_FEATURE_TYPES = ("ncRNA", "rRNA", "tRNA")


def compute_guide_coverage_by_match(candidates_df, off_dics_per_genome, n_genomes):
    """Compute coverage by checking for perfect 20nt+PAM matches across genomes.

    For each unique guide, checks off_plus[20] and off_minus[20] dicts in each
    genome. If the guide has at least one match (either strand), that genome is
    counted as covered.

    Parameters
    ----------
    candidates_df : DataFrame
        Must have a ``guide`` column.
    off_dics_per_genome : dict
        ``{genome_id: off_dics_dict}``
    n_genomes : int
        Total number of genomes.

    Returns
    -------
    DataFrame
        With added columns ``n_covered`` and ``coverage``.
    """
    genome_ids = list(off_dics_per_genome.keys())
    unique_guides = candidates_df["guide"].unique()

    guide_n_covered = {}
    for guide in unique_guides:
        seed = guide[-20:]
        count = 0
        for gid in genome_ids:
            od = off_dics_per_genome[gid]
            n_plus = len(od["off_plus"][20].get(seed, []))
            n_minus = len(od["off_minus"][20].get(seed, []))
            if n_plus + n_minus > 0:
                count += 1
        guide_n_covered[guide] = count

    df = candidates_df.copy()
    df["n_covered"] = df["guide"].map(guide_n_covered)
    df["coverage"] = df["n_covered"] / n_genomes
    return df


def find_rna_families(all_records, genome_ids, lt_to_feat_per_genome, seq_cache,
                      off_dics_per_genome, min_presence=0.9):
    """Find RNA families via guide-centric clustering.

    RNA features cannot be clustered by protein sequence, so this function
    uses shared guides to group RNA features into families (connected
    components in a bipartite guide-to-feature graph).

    Parameters
    ----------
    all_records : dict
        ``{genome_id: list of SeqRecord}``
    genome_ids : list of str
        Ordered genome IDs.
    lt_to_feat_per_genome : dict
        ``{genome_id: {locus_tag: (rec, feature)}}``
    seq_cache : dict
        ``{rec_id: {"fwd": str, "rev": str}}``
    off_dics_per_genome : dict
        ``{genome_id: off_dics_dict}``
    min_presence : float
        Minimum fraction of genomes where the family must be present
        (via perfect 20nt match) to be considered core.

    Returns
    -------
    tuple of (dict, dict)
        - rna_families: ``{family_id: {genome_id: [locus_tags]}}`` with
          R-prefixed IDs
        - rna_candidates: ``{family_id: DataFrame}`` of candidate guides
    """
    import math

    n_genomes = len(genome_ids)
    min_genomes = math.ceil(min_presence * n_genomes)

    # Phase A: Find all RNA guides from annotated features
    rows = []
    guide_to_features = defaultdict(set)  # guide -> set of (genome_id, locus_tag)

    for genome_id in genome_ids:
        lt_map = lt_to_feat_per_genome[genome_id]
        for lt, (rec, feat) in lt_map.items():
            if feat.type not in RNA_FEATURE_TYPES:
                continue
            fwd_seq = seq_cache.get(rec.id, {}).get("fwd")
            rev_seq = seq_cache.get(rec.id, {}).get("rev")
            targets = _find_coding_guides_in_gene(
                rec, feat, fwd_seq=fwd_seq, rev_seq=rev_seq
            )
            for guide, seq_win, recid, strand, pos in targets:
                rows.append({
                    "guide": guide,
                    "seq": seq_win,
                    "recid": recid,
                    "strand": strand,
                    "pos": pos,
                    "genome_id": genome_id,
                    "locus_tag": lt,
                })
                guide_to_features[guide].add((genome_id, lt))

    if not rows:
        return {}, {}

    # Phase C: Cluster into families via connected components (union-find)
    # Two features are in the same family if they share at least one guide.
    # Build adjacency from guides: each guide links all its features together.
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    all_feat_keys = set()
    for guide, feats in guide_to_features.items():
        feats_list = list(feats)
        for fk in feats_list:
            if fk not in parent:
                parent[fk] = fk
            all_feat_keys.add(fk)
        for i in range(1, len(feats_list)):
            union(feats_list[0], feats_list[i])

    # Group features by their root
    components = defaultdict(set)
    for fk in all_feat_keys:
        components[find(fk)].add(fk)

    # Also map each feature to its guides
    feat_to_guides = defaultdict(set)
    for guide, feats in guide_to_features.items():
        for fk in feats:
            feat_to_guides[fk].add(guide)

    rna_families = {}
    rna_candidates = {}
    counter = 0

    all_df = pd.DataFrame(rows)
    all_df = add_on_target_predictions(all_df)

    for component in components.values():
        feat_nodes = component
        guide_nodes = set()
        for fk in feat_nodes:
            guide_nodes.update(feat_to_guides[fk])

        if not feat_nodes or not guide_nodes:
            continue

        # Build family members: {genome_id: [locus_tags]}
        members = defaultdict(list)
        for gid, lt in feat_nodes:
            members[gid].append(lt)
        members = dict(members)

        # Check coverage via perfect match across all genomes
        n_covered = 0
        for gid in genome_ids:
            od = off_dics_per_genome[gid]
            for guide in guide_nodes:
                seed = guide[-20:]
                n_plus = len(od["off_plus"][20].get(seed, []))
                n_minus = len(od["off_minus"][20].get(seed, []))
                if n_plus + n_minus > 0:
                    n_covered += 1
                    break

        if n_covered < min_genomes:
            continue

        counter += 1
        family_id = f"R{counter:05d}"
        rna_families[family_id] = members

        # Extract candidate guides for this family
        candidates = all_df[all_df["guide"].isin(guide_nodes)].copy()
        rna_candidates[family_id] = candidates

    return rna_families, rna_candidates


def _family_feature_type(family_members, lt_to_feat_per_genome):
    """Determine the dominant feature type for a family.

    Returns the most common feature type among annotated members.
    """
    type_counts = Counter()
    for gid, lts in family_members.items():
        lt_map = lt_to_feat_per_genome.get(gid, {})
        for lt in lts:
            if lt in lt_map:
                _, feat = lt_map[lt]
                type_counts[feat.type] += 1
    if not type_counts:
        return "ncRNA"
    return type_counts.most_common(1)[0][0]


def _add_strain_off_targets(strain_df, off_dics):
    """Add per-genome off-target count columns to a strain DataFrame."""
    return add_off_targets(strain_df.copy(), off_dics=off_dics, drop_intermediate=False)


def generate_core_library(
    ref_files,
    n=3,
    min_presence=0.9,
    mmseqs_min_identity=0.5,
    mmseqs_coverage=0.8,
    cache_dir="off_dics",
    tmp_dir=None,
    coverage_threshold=0.8,
    badseeds=None,
    cluster_tsv_cache=None,
    output_csv=None,
    output_report=None,
    output_strains_dir=None,
):
    """Generate a core-genome CRISPRi library targeting conserved gene families.

    Identifies core gene families shared across all input genomes using
    MMseqs2 protein clustering, then designs guides targeting each family,
    ranked by a Global Penalty Score (GPS) that balances on-target efficiency,
    off-target risk, and cross-strain coverage.

    Parameters
    ----------
    ref_files : list of str or str
        Paths to GenBank genome files (one per strain), or a path to a
        directory containing GenBank files (``*.gb`` / ``*.gbk``).
    n : int
        Number of guides to select per core gene family.
    min_presence : float
        Minimum fraction of genomes a family must appear in to be considered
        "core" (default 0.9 = 90%).
    mmseqs_min_identity : float
        MMseqs2 ``--min-seq-id`` parameter.
    mmseqs_coverage : float
        MMseqs2 ``-c`` coverage parameter.
    cache_dir : str
        Directory for caching off-target dictionaries.
    tmp_dir : str or None
        Temporary directory for MMseqs2. Uses system temp if None.
    coverage_threshold : float
        Minimum collective coverage fraction for the 4th-guide rescue.
    badseeds : list of str or None
        Bad seed sequences to flag. If None, auto-detects from organism.
    cluster_tsv_cache : str or None
        Path to cache the MMseqs2 cluster TSV. If the file already exists,
        MMseqs2 is skipped and the cached TSV is loaded directly. If None,
        MMseqs2 always runs and results are not persisted.
    output_csv : str or None
        Path to write the core library CSV (one row per selected guide,
        cross-genome metrics only). If None, no file is written.
    output_report : str or None
        Path to write an HTML quality report. If None, no report is written.
    output_strains_dir : str or None
        Directory in which to write one CSV per strain, containing per-genome
        guide positions and off-target counts (format similar to single-genome
        library output). If None, no per-strain files are written.

    Returns
    -------
    DataFrame
        Core library with one row per selected guide. Key columns:
        ``family_id``, ``gene``, ``guide``, ``n_covered``, ``coverage``,
        ``gps``, ``mean_score`` (average on-target score across all strain
        matches), ``inbadseeds``.
    """
    # Expand a directory path to a sorted list of .gb/.gbk files
    if isinstance(ref_files, (str, bytes)):
        if os.path.isdir(ref_files):
            dir_path = ref_files
            patterns = [
                os.path.join(dir_path, "*.gb"),
                os.path.join(dir_path, "*.gbk"),
                os.path.join(dir_path, "*.genbank"),
            ]
            ref_files = sorted(
                p for pat in patterns for p in glob.glob(pat)
            )
            if not ref_files:
                raise ValueError(
                    f"No GenBank files (*.gb, *.gbk, *.genbank) found in {dir_path!r}"
                )
        else:
            ref_files = [ref_files]

    # --- Step 1: Parse all genomes ---
    print(f"Parsing {len(ref_files)} genome files...")
    all_records = {}
    genome_ids = []
    for ref_file in ref_files:
        records = list(SeqIO.parse(ref_file, "genbank"))
        genome_id = records[0].id
        all_records[genome_id] = records
        genome_ids.append(genome_id)
        print(f"  {genome_id}: {len(records)} record(s)")

    n_genomes = len(genome_ids)

    # Auto-detect bad seeds
    if badseeds is None:
        first_records = list(all_records.values())[0]
        badseeds = _auto_detect_badseeds(first_records)

    # --- Step 2: Extract proteins and cluster ---
    if cluster_tsv_cache and os.path.exists(cluster_tsv_cache):
        print(f"Loading cached cluster TSV from {cluster_tsv_cache}...")
        cluster_df = parse_cluster_tsv(cluster_tsv_cache)
    else:
        print("Extracting protein sequences...")
        with tempfile.TemporaryDirectory() as workdir:
            fasta_path = os.path.join(workdir, "proteins.fasta")
            extract_proteins_to_fasta(
                [all_records[gid] for gid in genome_ids], genome_ids, fasta_path
            )

            print("Running MMseqs2 clustering...")
            cluster_tsv = run_mmseqs2_easy_cluster(
                fasta_path,
                output_dir=workdir,
                min_identity=mmseqs_min_identity,
                coverage=mmseqs_coverage,
                tmp_dir=tmp_dir,
            )

            if cluster_tsv_cache:
                os.makedirs(os.path.dirname(os.path.abspath(cluster_tsv_cache)),
                            exist_ok=True)
                shutil.copy(cluster_tsv, cluster_tsv_cache)
                print(f"Cluster TSV cached to {cluster_tsv_cache}")

            cluster_df = parse_cluster_tsv(cluster_tsv)

    # --- Step 3: Identify core families ---
    print("Identifying core gene families...")
    core_families = find_core_families(
        cluster_df, genome_ids,
        min_presence=min_presence,
    )
    print(f"Found {len(core_families)} core gene families.")

    # --- Step 3b: Precompute sequence strings and locus_tag→feature lookups ---
    # Build these once per genome so the family loop can reuse them instead of
    # recomputing reverse-complement and feature iteration ~3000× per genome.
    print("Precomputing sequence strings and locus_tag→feature lookups...")
    lt_to_feat_per_genome = {}  # {genome_id: {locus_tag: (rec, feature)}}
    seq_cache = {}              # {rec_id: {"fwd": str, "rev": str}}
    for genome_id, records in all_records.items():
        lt_map = {}
        for rec in records:
            seq_cache[rec.id] = {
                "fwd": str(rec.seq),
                "rev": str(rec.seq.reverse_complement()),
            }
            for feat in rec.features:
                if feat.type in ("CDS", "gene", "ncRNA", "rRNA", "tRNA"):
                    lt = feat.qualifiers.get("locus_tag", [None])[0]
                    if not lt:
                        continue
                    if lt not in lt_map:
                        lt_map[lt] = (rec, feat)
                    elif feat.type != "gene" and lt_map[lt][1].type == "gene":
                        # Prefer specific type (CDS/rRNA/tRNA/ncRNA) over generic "gene"
                        lt_map[lt] = (rec, feat)
        lt_to_feat_per_genome[genome_id] = lt_map

    # --- Step 4: Build off-target dicts per genome ---
    print("Computing/loading off-target dictionaries...")
    off_dics_per_genome = {}
    for genome_id, records in all_records.items():
        off_dics_per_genome[genome_id] = get_off_dics(
            records, ref_name=genome_id, cache_dir=cache_dir
        )

    # --- Step 5: Family loop — find guides + coverage + off-targets + GPS + selection ---
    # Guide finding uses precomputed seq_cache and lt_to_feat_per_genome for speed.
    # Off-target computation uses precomputed off_dics per genome.
    gene_name_map = _family_gene_names(core_families, all_records)
    per_genome_covered = defaultdict(set)
    per_strain_rows = []
    family_results = []

    for family_id, members in tqdm(core_families.items(), desc="Processing families"):
        # a. Find coding-strand candidates using precomputed lookups.
        candidates_df = find_guides_for_family(
            members, all_records,
            lt_to_feat_per_genome=lt_to_feat_per_genome,
            seq_cache=seq_cache,
        )
        if candidates_df.empty:
            continue

        # b. Compute coverage and pre-select by coverage.
        candidates_df = compute_guide_coverage(candidates_df, n_genomes)
        candidates_df = preselect_by_coverage(candidates_df, n_genomes)
        if candidates_df.empty:
            continue

        # c. Compute off-targets (per-family, using precomputed dicts).
        guide_positions = _build_guide_positions(candidates_df)
        candidates_df = compute_multiGenome_off_targets(
            candidates_df, off_dics_per_genome, guide_positions
        )

        # d. Compute GPS and mean on-target score.
        candidates_df = compute_global_penalty_score(candidates_df)
        mean_scores = (
            candidates_df.groupby("guide")["score"]
            .mean()
            .rename("mean_score")
            .reset_index()
        )
        candidates_df = candidates_df.merge(mean_scores, on="guide", how="left")

        # e. Flag bad seeds and select top guides.
        candidates_df = add_badseeds(candidates_df, badseeds=badseeds)
        selected = select_top_guides(
            candidates_df, n=n,
            coverage_threshold=coverage_threshold,
            n_genomes=n_genomes,
        )
        selected["family_id"] = family_id
        family_results.append(selected)

        selected_guide_set = set(selected["guide"])
        covered_here = set(
            candidates_df[candidates_df["guide"].isin(selected_guide_set)]["genome_id"]
        )
        for gid in covered_here:
            per_genome_covered[gid].add(family_id)

        strain_rows = (
            candidates_df[candidates_df["guide"].isin(selected_guide_set)]
            .drop_duplicates(["guide", "genome_id", "locus_tag"])
            [["guide", "genome_id", "locus_tag", "recid",
              "strand", "pos", "score", "n_covered", "coverage",
              "gps", "inbadseeds"]]
            .copy()
        )
        strain_rows["family_id"] = family_id
        strain_rows["gene"] = gene_name_map.get(family_id)
        per_strain_rows.append(strain_rows)

    # --- Step 5b: RNA families ---
    print("Finding RNA families...")
    rna_families, rna_candidates = find_rna_families(
        all_records, genome_ids, lt_to_feat_per_genome, seq_cache,
        off_dics_per_genome, min_presence=min_presence,
    )
    rna_gene_name_map = _family_gene_names(
        rna_families, all_records,
        feature_types=RNA_FEATURE_TYPES,
    )
    print(f"Found {len(rna_families)} core RNA families.")

    # Build family_feature_types map for all families
    family_feature_types = {fid: "CDS" for fid in core_families}

    for family_id, candidates_df in tqdm(rna_candidates.items(),
                                         desc="Processing RNA families"):
        members = rna_families[family_id]

        # Coverage computed by perfect match
        candidates_df = compute_guide_coverage_by_match(
            candidates_df, off_dics_per_genome, n_genomes
        )
        candidates_df = preselect_by_coverage(candidates_df, n_genomes)
        if candidates_df.empty:
            continue

        # Same off-target, GPS, badseed, selection as CDS
        guide_positions = _build_guide_positions(candidates_df)
        candidates_df = compute_multiGenome_off_targets(
            candidates_df, off_dics_per_genome, guide_positions
        )
        candidates_df = compute_global_penalty_score(candidates_df)
        mean_scores = (
            candidates_df.groupby("guide")["score"]
            .mean().rename("mean_score").reset_index()
        )
        candidates_df = candidates_df.merge(mean_scores, on="guide", how="left")
        candidates_df = add_badseeds(candidates_df, badseeds=badseeds)
        selected = select_top_guides(
            candidates_df, n=n,
            coverage_threshold=coverage_threshold,
            n_genomes=n_genomes,
        )
        selected["family_id"] = family_id
        family_results.append(selected)

        # Track per-genome coverage via perfect match
        selected_guide_set = set(selected["guide"])
        for gid in genome_ids:
            od = off_dics_per_genome[gid]
            for guide in selected_guide_set:
                seed = guide[-20:]
                if (len(od["off_plus"][20].get(seed, []))
                        + len(od["off_minus"][20].get(seed, []))) > 0:
                    per_genome_covered[gid].add(family_id)
                    break

        strain_rows = (
            candidates_df[candidates_df["guide"].isin(selected_guide_set)]
            .drop_duplicates(["guide", "genome_id", "locus_tag"])
            [["guide", "genome_id", "locus_tag", "recid",
              "strand", "pos", "score", "n_covered", "coverage",
              "gps", "inbadseeds"]]
            .copy()
        )
        strain_rows["family_id"] = family_id
        strain_rows["gene"] = rna_gene_name_map.get(family_id)
        per_strain_rows.append(strain_rows)

    # Determine feature type for each RNA family
    for family_id, members in rna_families.items():
        family_feature_types[family_id] = _family_feature_type(
            members, lt_to_feat_per_genome
        )

    if not family_results:
        print("No guides found for any core family.")
        return pd.DataFrame()

    # --- Step 6: Assemble core library (one row per selected guide) ---
    all_gene_name_map = {**gene_name_map, **rna_gene_name_map}
    _core_cols = ["family_id", "gene", "guide", "n_covered", "coverage", "gps",
                  "off_11_gene_score", "off_9_prom_score", "mean_score",
                  "inbadseeds", "feature_type"]
    raw = pd.concat(family_results, ignore_index=True)
    raw["gene"] = raw["family_id"].map(all_gene_name_map)
    raw["feature_type"] = raw["family_id"].map(family_feature_types)
    library = (
        raw.drop_duplicates(["family_id", "guide"])
        [_core_cols]
        .reset_index(drop=True)
    )

    if output_csv is not None:
        library.to_csv(output_csv, index=False)
        print(f"Library written to {output_csv}")

    # --- Step 7: Write per-strain CSVs ---
    if output_strains_dir is not None and per_strain_rows:
        os.makedirs(output_strains_dir, exist_ok=True)
        all_strain_df = pd.concat(per_strain_rows, ignore_index=True)
        _strain_out_cols = [
            "guide", "family_id", "gene", "genome_id", "locus_tag",
            "recid", "strand", "pos", "score",
            "ntargets_plus", "ntargets_minus", "ntargets",
            "noff_12_plus", "noff_12_minus", "noff_12",
            "noff_11_gene",
            "noff_9_prom_plus", "noff_9_prom_minus", "noff_9_prom",
            "inbadseeds",
        ]
        print("Computing per-strain off-target counts...")
        for genome_id in genome_ids:
            strain_df = all_strain_df[all_strain_df["genome_id"] == genome_id].copy()
            if strain_df.empty:
                continue
            strain_df = _add_strain_off_targets(strain_df, off_dics_per_genome[genome_id])
            strain_df = strain_df[_strain_out_cols].reset_index(drop=True)
            # sanitise genome_id for filename
            safe_id = genome_id.replace("/", "_").replace(" ", "_")
            out_path = os.path.join(output_strains_dir, f"{safe_id}.csv")
            strain_df.to_csv(out_path, index=False)
            print(f"  Strain library written to {out_path}")

    if output_report is not None:
        from crisprbact.visualize import generate_core_report
        all_families = {**core_families, **rna_families}
        generate_core_report(
            library, all_records, genome_ids, all_families,
            per_genome_covered, n, output_report,
            gene_name_map=all_gene_name_map,
            family_feature_types=family_feature_types,
        )
        print(f"Report written to {output_report}")

    return library
