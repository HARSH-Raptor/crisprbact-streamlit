"""Dictionary-based off-target computation for genome-wide library design.

This module provides a fast approach to off-target analysis by building
seed→positions lookup dictionaries. It is optimized for genome-wide screens
where many guides need to be evaluated against the same genome.
"""

import os
import pickle
from collections import defaultdict

import numpy as np
from Bio.SeqRecord import SeqRecord

from crisprbact.off_target import FEATURE_TYPES
from crisprbact.utils import rev_comp


def save_obj(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _dict_grouping(pairs):
    d = defaultdict(list)
    for k, v in pairs:
        d[k].append(v)
    return dict(d)


def compute_off_target_dic(recs, seed_sizes=None, search_ori=1, verbose_id=True):
    """Build seed→positions lookup dictionaries from sequence records.

    Parameters
    ----------
    recs : iterable of SeqRecord
        Sequence records to scan. Records may encode positional info in their
        id field as "recid,start-end,strand" (as produced by write_gene_fasta
        / write_promoter_fasta), or be plain genome records.
    seed_sizes : list of int, optional
        Seed sizes to index. Default [8, 9, 10, 11].
    search_ori : {1, -1}
        1 for non-template strand, -1 for template strand.
    verbose_id : bool
        If True, position keys are "recid_pos" strings (needed for
        multi-contig genomes). If False, positions are plain integers.

    Returns
    -------
    dict
        {seed_size: {seed_sequence: [positions...]}}
    """
    if seed_sizes is None:
        seed_sizes = [8, 9, 10, 11]

    tempdic = {s: [] for s in seed_sizes}
    for rec in recs:
        try:
            recid, pos, strand = rec.id.split(",")
            strand = int(strand)
            left, right = pos.split("-")
            left, right = int(left), int(right)
        except (ValueError, AttributeError):
            recid = rec.id
            left = 0
            right = len(rec.seq)
            strand = 1

        L = len(rec.seq)
        if search_ori == 1:
            seq = rec.seq

            def pam_pos(p, strand=strand, left=left, right=right):
                relpos = p + 2
                if strand == 1:
                    return left + relpos
                else:
                    return right - relpos - 1

        else:
            seq = rec.seq.reverse_complement()

            def pam_pos(p, strand=strand, left=left, right=right, L=L):
                relpos = L - p - 3
                if strand == 1:
                    return left + relpos
                else:
                    return right - relpos - 1

        i = 0
        while True:
            p = seq.find("CC", start=i)
            if p == -1 or p > L - 23:
                break
            for s in seed_sizes:
                seed = rev_comp(str(seq[p + 3:p + 3 + s]))
                if verbose_id:
                    tempdic[s].append((seed, "{}_{}".format(recid, pam_pos(p))))
                else:
                    tempdic[s].append((seed, pam_pos(p)))
            i = p + 1

    return {s: _dict_grouping(tempdic[s]) for s in seed_sizes}


def extract_gene_records(records):
    """Extract gene sequences from genome records as SeqRecords.

    Parameters
    ----------
    records : list of SeqRecord
        Parsed genome records (GenBank format).

    Returns
    -------
    list of SeqRecord
        Each record's id encodes "recid,start-end,strand".
    """
    seqs = []
    for rec in records:
        for g in rec.features:
            if g.type in FEATURE_TYPES:
                gene_rec = SeqRecord(
                    g.extract(rec.seq),
                    id="{},{}-{},{}".format(
                        rec.id,
                        int(g.location.start),
                        int(g.location.end),
                        g.location.strand,
                    ),
                    description="",
                )
                seqs.append(gene_rec)
    return seqs


def extract_promoter_records(records, upstream=100, downstream=20):
    """Extract promoter sequences from genome records as SeqRecords.

    Promoters are defined as sequences *upstream* bases upstream to
    *downstream* bases downstream of each annotated gene start.

    Parameters
    ----------
    records : list of SeqRecord
        Parsed genome records (GenBank format).
    upstream : int
        Bases upstream of gene start.
    downstream : int
        Bases downstream of gene start.

    Returns
    -------
    list of SeqRecord
    """
    seqs = []
    for rec in records:
        for g in rec.features:
            if g.type in FEATURE_TYPES:
                if g.location.strand == 1:
                    left = int(g.location.start) - upstream
                    right = int(g.location.start) + downstream
                else:
                    left = int(g.location.end) - downstream
                    right = int(g.location.end) + upstream
                left = max(0, left)
                right = min(len(rec.seq), right)
                prom_rec = SeqRecord(
                    rec.seq[left:right],
                    id="{},{}-{},{}".format(rec.id, left, right, 1),
                    description="",
                )
                seqs.append(prom_rec)
    return seqs


def extract_genome_records(records):
    """Convert genome records to plain SeqRecords (strip annotations).

    Parameters
    ----------
    records : list of SeqRecord
        Parsed genome records.

    Returns
    -------
    list of SeqRecord
    """
    return [SeqRecord(rec.seq, id=rec.id, description="") for rec in records]


def _compute_and_cache_dic(recs, seed_sizes, search_ori, cache_path, name):
    """Compute off-target dict from records and cache to disk."""
    resdic = compute_off_target_dic(recs, seed_sizes=seed_sizes, search_ori=search_ori)
    save_obj(resdic, os.path.join(cache_path, name + ".pkl"))
    return resdic


def get_off_dics(records, ref_name="default", cache_dir="off_dics"):
    """Compute or load cached off-target dictionaries for a genome.

    Parameters
    ----------
    records : list of SeqRecord
        Parsed genome records (GenBank format).
    ref_name : str
        Name used to key the cache directory.
    cache_dir : str
        Root directory for cached dictionaries.

    Returns
    -------
    dict
        Keys are dict names ("off_plus", "off_minus", etc.),
        values are the seed→positions dictionaries.
    """
    dic_path = os.path.join(cache_dir, ref_name)
    os.makedirs(dic_path, exist_ok=True)

    required_dics = [
        "off_minus",
        "off_plus",
        "off_11_gene",
        "off_9_prom_minus",
        "off_9_prom_plus",
    ]
    all_cached = all(
        os.path.exists(os.path.join(dic_path, d + ".pkl")) for d in required_dics
    )

    if all_cached:
        return {d: load_obj(os.path.join(dic_path, d + ".pkl")) for d in required_dics}

    genome_recs = extract_genome_records(records)
    gene_recs = extract_gene_records(records)
    prom_recs = extract_promoter_records(records)

    return {
        "off_plus": _compute_and_cache_dic(genome_recs, [20, 12], 1, dic_path, "off_plus"),
        "off_minus": _compute_and_cache_dic(genome_recs, [20, 12], -1, dic_path, "off_minus"),
        "off_11_gene": _compute_and_cache_dic(gene_recs, [11], 1, dic_path, "off_11_gene"),
        "off_9_prom_plus": _compute_and_cache_dic(prom_recs, [9], 1, dic_path, "off_9_prom_plus"),
        "off_9_prom_minus": _compute_and_cache_dic(prom_recs, [9], -1, dic_path, "off_9_prom_minus"),
    }


def count_off_target_col(targets, dic, seed_size):
    """Vectorized off-target count for a single seed dictionary and size.

    Replaces a row-wise ``.apply()`` call by:
    1. Looking up the total count for each guide's seed in one pass.
    2. Subtracting 1 for self-exclusion where the guide's own position
       appears in the seed's position list.

    Parameters
    ----------
    targets : DataFrame
        Must have columns: guide, pos, recid.
    dic : dict
        Seed-size keyed dictionary {seed_size: {seed: [position_strings]}}.
    seed_size : int
        Seed length to use.

    Returns
    -------
    numpy.ndarray of int
        Off-target counts aligned with targets.index.
    """
    seed_dic = dic[seed_size]  # {seed_str: [position_strings]}

    # Extract PAM-proximal seeds for every guide
    seeds = targets["guide"].str[-seed_size:].str.upper()

    # Total count per seed (no self-exclusion yet)
    seed_total = {s: len(p) for s, p in seed_dic.items()}
    total = seeds.map(seed_total).fillna(0).to_numpy(dtype=int)

    # Build position sets per unique seed for O(1) self-exclusion lookup
    unique_seeds = seeds.unique()
    seed_sets = {s: set(seed_dic.get(s, [])) for s in unique_seeds}

    # Self-exclusion: subtract 1 if this guide's own position is in the list
    self_keys = targets["recid"].astype(str) + "_" + targets["pos"].astype(str)
    exclude = np.fromiter(
        (seed_sets.get(s, frozenset()).__contains__(sk)
         for s, sk in zip(seeds, self_keys)),
        dtype=int,
        count=len(targets),
    )

    return np.clip(total - exclude, 0, None)


def count_off_targets(guide, pos, recid, dic, seed_size):
    """Count off-targets for a single guide, excluding its own position.

    ``pos`` and ``recid`` may be None (e.g. when the guide has no perfect
    match in the genome); in that case no self-exclusion is applied.

    Parameters
    ----------
    guide : str
        20-nt guide sequence.
    pos : int, str, or None
        Position of the guide's PAM in the genome, or None to skip exclusion.
    recid : str or None
        Record ID where the guide is located, or None to skip exclusion.
    dic : dict
        Seed→positions dictionary for a given seed size.
    seed_size : int
        Length of the seed to look up.

    Returns
    -------
    int
        Number of off-target positions.
    """
    if pos is None or recid is None:
        seed = guide[-seed_size:]
        return len(dic[seed_size].get(seed, []))
    seed = guide[-seed_size:]
    positions = dic[seed_size].get(seed, [])
    n = len(positions)
    actual_target = "{}_{}".format(recid, pos)
    if actual_target in positions:
        n -= 1
    return n
