"""Pangenome construction via MMseqs2 for core-genome CRISPRi library design.

This module identifies core gene families conserved across multiple bacterial
genomes using protein clustering with MMseqs2.
"""

import math
import os
import shutil
import subprocess
import tempfile

import pandas as pd


def extract_proteins_to_fasta(records_list, genome_ids, output_fasta):
    """Extract CDS protein sequences from multiple GenBank records to FASTA.

    Parameters
    ----------
    records_list : list of list of SeqRecord
        One list of records per genome.
    genome_ids : list of str
        Genome identifiers, one per entry in records_list.
    output_fasta : str or path-like
        Path to write the combined protein FASTA file.

    Returns
    -------
    str
        Path to the written FASTA file.
    """
    with open(output_fasta, "w") as fh:
        for genome_id, records in zip(genome_ids, records_list):
            for rec in records:
                for feature in rec.features:
                    if feature.type != "CDS":
                        continue
                    locus_tags = feature.qualifiers.get("locus_tag", [])
                    if not locus_tags:
                        continue
                    locus_tag = locus_tags[0]
                    translation = feature.qualifiers.get("translation", [None])[0]
                    if translation is None:
                        try:
                            nuc = feature.location.extract(rec.seq)
                            translation = str(nuc.translate(to_stop=True))
                            if len(translation) < 10:
                                continue
                        except Exception:
                            continue
                    fh.write(f">{genome_id}|{locus_tag}\n{translation}\n")
    return output_fasta


def run_mmseqs2_easy_cluster(
    input_fasta, output_dir, min_identity=0.5, coverage=0.8, tmp_dir=None
):
    """Cluster proteins using MMseqs2 easy-cluster.

    Parameters
    ----------
    input_fasta : str
        Path to the input protein FASTA file.
    output_dir : str
        Directory where MMseqs2 output will be written.
    min_identity : float
        Minimum sequence identity for clustering (--min-seq-id).
    coverage : float
        Minimum coverage fraction (-c).
    tmp_dir : str, optional
        Temporary directory for MMseqs2. If None, a system temp dir is used.

    Returns
    -------
    str
        Path to the cluster TSV file (``{output_dir}/clusters_cluster.tsv``).

    Raises
    ------
    RuntimeError
        If mmseqs2 is not found in PATH.
    """
    if shutil.which("mmseqs") is None:
        raise RuntimeError(
            "MMseqs2 not found. Please install it:\n"
            "  conda install -c conda-forge -c bioconda mmseqs2\n"
            "  or visit https://github.com/soedinglab/MMseqs2"
        )

    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, "clusters")

    cleanup_tmp = False
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp()
        cleanup_tmp = True

    try:
        cmd = [
            "mmseqs",
            "easy-cluster",
            input_fasta,
            prefix,
            tmp_dir,
            "--min-seq-id",
            str(min_identity),
            "-c",
            str(coverage),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        if cleanup_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return prefix + "_cluster.tsv"


def parse_cluster_tsv(cluster_tsv):
    """Parse MMseqs2 cluster TSV output.

    The TSV has two columns: representative and member. Member IDs are
    expected to have the format ``{genome_id}|{locus_tag}``.

    Parameters
    ----------
    cluster_tsv : str
        Path to the MMseqs2 cluster TSV file.

    Returns
    -------
    DataFrame
        Columns: ``family_id`` (representative), ``genome_id``, ``locus_tag``.
    """
    df = pd.read_csv(
        cluster_tsv,
        sep="\t",
        header=None,
        names=["family_id", "member"],
    )
    # Strip genome_id prefix from representative ID to get a clean family_id
    df["family_id"] = df["family_id"].str.split("|", n=1).str[1]
    # Split member on '|' to get genome_id and locus_tag
    split = df["member"].str.split("|", n=1, expand=True)
    df["genome_id"] = split[0]
    df["locus_tag"] = split[1]
    df = df.drop(columns=["member"])
    return df


def find_core_families(cluster_df, genome_ids, min_presence=0.9):
    """Identify core gene families present in most genomes.

    Parameters
    ----------
    cluster_df : DataFrame
        Output of :func:`parse_cluster_tsv` with columns
        ``family_id``, ``genome_id``, ``locus_tag``.
    genome_ids : list of str
        All genome identifiers.
    min_presence : float
        Minimum fraction of genomes a family must appear in.

    Returns
    -------
    dict
        ``{family_id: {genome_id: [locus_tags]}}`` for core families.
    """
    min_genomes = math.ceil(min_presence * len(genome_ids))

    core = {}
    counter = 0
    for _rep_id, group in cluster_df.groupby("family_id"):
        n_genomes = group["genome_id"].nunique()
        if n_genomes < min_genomes:
            continue
        # Build genome_id -> [locus_tags] mapping
        members = {}
        for genome_id, sub in group.groupby("genome_id"):
            members[genome_id] = sub["locus_tag"].tolist()
        counter += 1
        core[f"F{counter:05d}"] = members

    return core
