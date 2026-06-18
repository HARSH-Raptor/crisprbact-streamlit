"""Add-on library design for strain-specific CRISPRi guide supplementation.

Generates guides for genes in a target strain that are not adequately covered
by an existing library (e.g. a core-genome library). When combined, the
existing library and add-on library provide comprehensive gene coverage.

For each gene the pipeline brings the total number of coding-strand guides up
to ``n``: if a gene already has ``k`` coding-strand matches in the existing
library, the add-on contributes up to ``n − k`` new guides (skipping any guide
sequence already present in the existing library for that gene).
"""

import pandas as pd
from Bio import SeqIO

from crisprbact.library import _auto_detect_badseeds, generate_library
from crisprbact.map_library import map_library


def generate_addon_library(
    ref_file,
    existing_library,
    n=3,
    cache_dir="off_dics",
    ref_name=None,
    badseeds=None,
    output_csv=None,
    output_report=None,
):
    """Generate an add-on library to supplement an existing guide RNA library.

    For each gene in the target genome, counts how many coding-strand guides
    ``k`` the existing library already provides. If ``k < n``, the add-on
    contributes up to ``n − k`` additional guides (excluding any guide sequence
    already present in the existing library for that gene). Genes with ``k ≥ n``
    are skipped entirely.

    Parameters
    ----------
    ref_file : str or path-like
        Path to GenBank genome file for the target strain.
    existing_library : str, path-like, or DataFrame
        Existing library CSV (or DataFrame with 'guide' column) to supplement.
    n : int
        Target total number of coding-strand guides per gene across both
        libraries combined.
    cache_dir : str
        Directory for cached off-target dictionaries.
    ref_name : str or None
        Genome name for off-target dict caching. If None, uses first record ID.
    badseeds : list of str or None
        Bad seed sequences to flag. If None, auto-detects based on organism.
    output_csv : str or path-like or None
        Path to write the add-on library CSV. If None, not written.
    output_report : str or path-like or None
        Path to write an HTML report. If None, no report is generated.

    Returns
    -------
    DataFrame
        Add-on library guides (same columns as generate_library output).
        Empty DataFrame if all genes already have ≥ n coding-strand guides.
    """
    records = list(SeqIO.parse(ref_file, "genbank"))
    if not records:
        raise ValueError(f"No records found in {ref_file}")

    if ref_name is None:
        ref_name = records[0].id

    # Auto-detect badseeds once, consistently for both map and generate steps
    if badseeds is None:
        badseeds = _auto_detect_badseeds(records)

    # Normalise existing_library to a DataFrame to check for emptiness
    if isinstance(existing_library, pd.DataFrame):
        existing_df = existing_library
    else:
        existing_df = pd.read_csv(existing_library)

    if "guide" not in existing_df.columns:
        raise ValueError("Existing library must have a 'guide' column.")

    _MAP_COLS = [
        "guide", "n_matches", "recid", "strand", "pos",
        "locus_tag", "gene", "gene_ori", "targets_coding_strand",
        "on_target_score", "score_quartile",
        "ntargets", "noff_12", "noff_11_gene", "noff_9_prom", "inbadseeds",
    ]

    guides = existing_df["guide"].dropna().tolist()
    if not guides:
        # No guides in the existing library: every gene needs a full add-on
        print("Existing library is empty — all genes will receive add-on guides.")
        map_result = pd.DataFrame(columns=_MAP_COLS)
        existing_guides_per_gene = {}
    else:
        print("Mapping existing library to target genome...")
        map_result = map_library(
            existing_library,
            ref_file,
            cache_dir=cache_dir,
            ref_name=ref_name,
            badseeds=badseeds,
        )

        # Per-gene set of coding-strand guide sequences already covered
        coding = map_result[map_result["targets_coding_strand"] == True]  # noqa
        existing_guides_per_gene = {
            lt: set(grp["guide"].str.upper())
            for lt, grp in coding.groupby("locus_tag")
        }

    # existing_coverage[lt] = number of unique coding-strand guides already present
    existing_coverage = {lt: len(gs) for lt, gs in existing_guides_per_gene.items()}

    # Collect all gene locus_tags in the target genome
    all_genes = {
        lt
        for rec in records
        for feat in rec.features
        if feat.type == "gene" and "locus_tag" in feat.qualifiers
        for lt in feat.qualifiers["locus_tag"]
    }

    needs_addon = {lt for lt in all_genes if existing_coverage.get(lt, 0) < n}

    print(
        f"Genes needing add-on (< {n} existing coding-strand guides): "
        f"{len(needs_addon)} / {len(all_genes)}"
    )

    if not needs_addon:
        print("All genes already have ≥ n coding-strand guides. Returning empty DataFrame.")
        addon_lib = pd.DataFrame()
    else:
        print("Generating full library for target genome...")
        full_lib = generate_library(
            ref_file,
            n=n,
            ref_name=ref_name,
            cache_dir=cache_dir,
            badseeds=badseeds,
        )

        # For each gene needing add-on: exclude existing guides, take top (n − k)
        addon_rows = []
        for lt in needs_addon:
            k = existing_coverage.get(lt, 0)
            needed = n - k
            gene_df = full_lib[full_lib["locus_tag"] == lt]
            if len(gene_df) == 0:
                continue
            # Exclude guide sequences already in the existing library for this gene
            already = existing_guides_per_gene.get(lt, set())
            gene_df = gene_df[~gene_df["guide"].str.upper().isin(already)]
            # Take the top `needed` guides (lowest gene_rank = best)
            gene_df = gene_df.nsmallest(needed, "gene_rank")
            if len(gene_df) > 0:
                addon_rows.append(gene_df)

        addon_lib = (
            pd.concat(addon_rows, ignore_index=True) if addon_rows else pd.DataFrame()
        )

    if output_csv is not None:
        addon_lib.to_csv(output_csv, index=False)
        print(f"Add-on library written to {output_csv}")

    if output_report is not None:
        from crisprbact.visualize import generate_addon_report

        generate_addon_report(
            addon_lib,
            map_result,
            records,
            existing_coverage,
            n,
            output_report,
        )
        print(f"Report written to {output_report}")

    return addon_lib
