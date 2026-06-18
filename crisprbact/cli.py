from Bio import SeqIO
import click
from crisprbact.predict import on_target_predict


class Config(object):
    def __init__(self):
        self.verbose = False


pass_config = click.make_pass_decorator(Config, ensure=True)

OFF_TARGET_DETAILS = [
    "off_target_recid",
    "off_target_start",
    "off_target_end",
    "off_target_pampos",
    "off_target_strand",
    "off_target_longest_perfect_match",
    "off_target_good_orientation",
    "off_target_feat_type",
    "off_target_feat_start",
    "off_target_feat_end",
    "off_target_feat_strand",
    "off_target_locus_tag",
    "off_target_gene",
    "off_target_note",
    "off_target_product",
    "off_target_protein_id",
]
HEADER = [
    "target_id",
    "guide",
    "guide_start",
    "guide_end",
    "pam_pos",
    "score",
    "target_seq_id",
    "seed_size",
] + OFF_TARGET_DETAILS
GENOME_FORMAT = "genbank"


def _parse_badseeds(badseeds):
    """Parse --badseeds CLI option to a list or None (auto-detect)."""
    if badseeds is None:
        return None
    return [] if badseeds.lower() == "none" else badseeds.split(",")


@click.group()
@click.option("-v", "--verbose", is_flag=True)
@pass_config
def main(config, verbose):
    config.verbose = verbose


@main.group()
@pass_config
def predict(config):
    pass


@predict.command()
@click.option(
    "-t",
    "--target",
    type=str,
    required=True,
    help="Sequence file to target",
)
@click.option(
    "-s",
    "--off-target-sequence",
    type=click.File("r"),
    help="Sequence in which you want to find off-targets",
)
@click.option(
    "-w",
    "--off-target-sequence-format",
    type=click.Choice(["fasta", "gb", "genbank"]),
    default=GENOME_FORMAT,
    show_default=True,
    help="Sequence in which you want to find off-targets format",
)
@click.argument("output-file", type=click.File("w"), default="-")
@pass_config
def from_str(
    config, target, off_target_sequence, off_target_sequence_format, output_file
):
    """
    Outputs candidate guide RNAs for the S. pyogenes dCas9 with predicted on-target
    activity from a target gene.

    [OUTPUT_FILE] file where the candidate guide RNAs are saved. Default = "stdout"

    """
    if config.verbose:
        print_parameters(target)
    if off_target_sequence:
        genome_fh = SeqIO.parse(off_target_sequence, off_target_sequence_format)
    else:
        genome_fh = None
    guide_rnas = on_target_predict(target, genome_fh)

    click.echo("\t".join(HEADER), file=output_file)
    write_guide_rnas(guide_rnas, output_file, len(HEADER))


@predict.command()
@click.option(
    "-t",
    "--target",
    type=click.File("r"),
    required=True,
    help="Sequence file to target",
)
@click.option(
    "-f",
    "--seq-format",
    type=click.Choice(["fasta", "gb", "genbank"]),
    help="Sequence file to target format",
    default="fasta",
    show_default=True,
)
@click.option(
    "-s",
    "--off-target-sequence",
    type=click.File("r"),
    help="Sequence in which you want to find off-targets",
)
@click.option(
    "-w",
    "--off-target-sequence-format",
    type=click.Choice(["fasta", "gb", "genbank"]),
    default=GENOME_FORMAT,
    show_default=True,
    help="Sequence in which you want to find off-targets format",
)
@click.argument("output-file", type=click.File("w"), default="-")
@pass_config
def from_seq(
    config,
    target,
    seq_format,
    off_target_sequence,
    off_target_sequence_format,
    output_file,
):
    """
    Outputs candidate guide RNAs for the S. pyogenes dCas9 with predicted on-target
    activity from a target gene.

    [OUTPUT_FILE] file where the candidate guide RNAs are saved. Default = "stdout"

    """
    fg = "blue"
    if config.verbose:
        print_parameters(target.name, fg)

    # Parse genome once before the record loop so the file handle is not
    # re-used (and exhausted) on every iteration when there are multiple records.
    if off_target_sequence:
        genome_records = list(SeqIO.parse(off_target_sequence, off_target_sequence_format))
    else:
        genome_records = None

    click.echo("\t".join(HEADER), file=output_file)
    for record in SeqIO.parse(target, seq_format):
        if config.verbose:
            click.secho(" - search guide RNAs for %s " % record.id, fg=fg)
        guide_rnas = on_target_predict(str(record.seq), genome_records)
        write_guide_rnas(guide_rnas, output_file, len(HEADER), record.id)


def print_parameters(target, fg="blue"):
    click.secho("[Verbose mode]", fg=fg)
    click.secho("Target sequence : %s" % target, fg=fg)


def write_guide_rnas(
    guide_rnas,
    output_file,
    header_size,
    seq_id="N/A",
):
    for guide_rna in guide_rnas:
        row = [
            str(guide_rna["target_id"]),
            guide_rna["guide"],
            str(guide_rna["guide_start"]),
            str(guide_rna["guide_end"]),
            str(guide_rna["pam_pos"]),
            str(guide_rna["score"]),
            seq_id,
        ]
        # seed_size = ""
        if len(guide_rna["off_targets_per_seed"]) > 0:
            for off_target_per_seed in guide_rna["off_targets_per_seed"]:
                for off_target in off_target_per_seed["off_targets"]:
                    seed_size = off_target_per_seed["seed_size"]

                    def extract_off_target_detail(key):
                        if key in off_target:
                            return str(off_target[key])
                        else:
                            return ""

                    details = map(extract_off_target_detail, OFF_TARGET_DETAILS)
                    line_to_print = row + [str(seed_size)] + list(details)
                    if len(line_to_print) != header_size:
                        raise ValueError(
                            f"Row length {len(line_to_print)} != header {header_size}"
                        )
                    click.echo(
                        "\t".join(line_to_print),
                        file=output_file,
                    )
        else:
            line_to_print = row + [""] + list(map(lambda x: "", OFF_TARGET_DETAILS))
            if len(line_to_print) != header_size:
                raise ValueError(
                    f"Row length {len(line_to_print)} != header {header_size}"
                )
            click.echo(
                "\t".join(line_to_print),
                file=output_file,
            )


@main.group()
@pass_config
def library(config):
    """Commands for genome-wide CRISPRi library design."""
    pass


@library.command()
@click.option(
    "--ref",
    type=click.Path(exists=True),
    required=True,
    help="Path to GenBank file of the genome.",
)
@click.option(
    "-n",
    "--n-guides",
    default=3,
    type=int,
    show_default=True,
    help="Number of guides to select per gene.",
)
@click.option(
    "--ref-name",
    default="default",
    show_default=True,
    help="Genome name for caching off-target dicts. Default: first record ID.",
)
@click.option(
    "--cache-dir",
    default="off_dics",
    show_default=True,
    help="Directory to cache off-target dictionaries.",
)
@click.option(
    "-o",
    "--output",
    default=None,
    help="Output CSV file path. If not set, prints to stdout.",
)
@click.option(
    "--report",
    default=None,
    type=click.Path(),
    help="Path to write an HTML quality report.",
)
@click.option(
    "--badseeds",
    default=None,
    help=(
        "Comma-separated bad seed sequences to flag, or 'none' to disable. "
        "If not provided, auto-detects: uses E. coli defaults for E. coli "
        "genomes, or empty list for other organisms."
    ),
)
@pass_config
def generate(config, ref, n_guides, ref_name, cache_dir, output, report, badseeds):
    """Generate a genome-wide CRISPRi guide RNA library.

    Finds all NGG PAM targets in the genome, annotates them with gene
    information, predicts on-target activity, evaluates off-targets, and
    selects the top N guides per gene.

    \b
    Examples:
      crisprbact library generate --ref genome.gb -o library.csv
      crisprbact library generate --ref genome.gb -o library.csv --report report.html
      crisprbact library generate --ref genome.gb -n 10 --badseeds TATAG,AAAGG
      crisprbact library generate --ref genome.gb --badseeds none
    """
    from crisprbact.library import generate_library

    seeds = _parse_badseeds(badseeds)

    lib = generate_library(
        ref,
        n=n_guides,
        ref_name=ref_name,
        cache_dir=cache_dir,
        badseeds=seeds,
        output_csv=output,
        output_report=report,
    )
    if not output:
        click.echo(lib.to_csv(index=False))


@library.command("generate-core")
@click.option(
    "--genomes-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=None,
    help="Directory containing GenBank genome files (*.gb / *.gbk).",
)
@click.option(
    "--ref",
    type=click.Path(exists=True),
    multiple=True,
    help="Path to a GenBank genome file. Repeat for each strain. "
    "Use --genomes-dir to pass a folder instead.",
)
@click.option(
    "-n",
    "--n-guides",
    default=3,
    type=int,
    show_default=True,
    help="Number of guides to select per core gene family.",
)
@click.option(
    "--min-presence",
    default=0.9,
    type=float,
    show_default=True,
    help="Minimum fraction of genomes a family must appear in.",
)
@click.option(
    "--mmseqs-identity",
    default=0.5,
    type=float,
    show_default=True,
    help="MMseqs2 --min-seq-id clustering identity threshold.",
)
@click.option(
    "--mmseqs-coverage",
    default=0.8,
    type=float,
    show_default=True,
    help="MMseqs2 -c coverage threshold.",
)
@click.option(
    "--coverage-threshold",
    default=0.8,
    type=float,
    show_default=True,
    help="Minimum collective coverage for 4th-guide rescue.",
)
@click.option(
    "--cache-dir",
    default="off_dics",
    show_default=True,
    help="Directory to cache off-target dictionaries.",
)
@click.option(
    "--tmp-dir",
    default=None,
    type=click.Path(),
    help="Temporary directory for MMseqs2 (default: system temp).",
)
@click.option(
    "--badseeds",
    default=None,
    help=(
        "Comma-separated bad seed sequences to flag, or 'none' to disable. "
        "Auto-detects E. coli defaults if not provided."
    ),
)
@click.option(
    "-o",
    "--output",
    default=None,
    help="Output CSV file path. If not set, prints to stdout.",
)
@click.option(
    "--strains-dir",
    default=None,
    type=click.Path(),
    help="Directory to write per-strain guide CSV files.",
)
@click.option(
    "--report",
    default=None,
    type=click.Path(),
    help="Path to write an HTML quality report.",
)
@pass_config
def generate_core(
    config,
    genomes_dir,
    ref,
    n_guides,
    min_presence,
    mmseqs_identity,
    mmseqs_coverage,
    coverage_threshold,
    cache_dir,
    tmp_dir,
    badseeds,
    output,
    strains_dir,
    report,
):
    """Generate a core-genome CRISPRi library across multiple strains.

    Clusters proteins from all input genomes with MMseqs2 to identify core
    gene families, then designs guides targeting each family, ranked by a
    Global Penalty Score balancing efficiency, off-targets, and coverage.

    Pass genomes as a directory (--genomes-dir) or as individual files
    (--ref, repeatable). Exactly one of the two must be provided.

    \b
    Examples:
      crisprbact library generate-core --genomes-dir genomes/ -o core_lib.csv
      crisprbact library generate-core --genomes-dir genomes/ -n 3 \\
          --min-presence 0.9 --cache-dir off_dics -o core_lib.csv \\
          --strains-dir strains/ --report report.html
      crisprbact library generate-core --ref g1.gb --ref g2.gb -o core_lib.csv
    """
    from crisprbact.core_library import generate_core_library

    seeds = _parse_badseeds(badseeds)

    # Resolve genome file list
    if genomes_dir and ref:
        raise click.UsageError(
            "Specify either --genomes-dir or --ref, not both."
        )
    if not genomes_dir and not ref:
        raise click.UsageError(
            "Provide --genomes-dir <dir> or one or more --ref <file> options."
        )
    ref_files = genomes_dir if genomes_dir else list(ref)

    lib = generate_core_library(
        ref_files,
        n=n_guides,
        min_presence=min_presence,
        mmseqs_min_identity=mmseqs_identity,
        mmseqs_coverage=mmseqs_coverage,
        cache_dir=cache_dir,
        tmp_dir=tmp_dir,
        coverage_threshold=coverage_threshold,
        badseeds=seeds,
        output_csv=output,
        output_report=report,
        output_strains_dir=strains_dir,
    )

    if not output:
        click.echo(lib.to_csv(index=False))


@library.command("generate-addon")
@click.option(
    "--ref",
    type=click.Path(exists=True),
    required=True,
    help="Path to GenBank genome file for the target strain.",
)
@click.option(
    "-l",
    "--library",
    "library_csv",
    type=click.Path(exists=True),
    required=True,
    help="Existing library CSV with a 'guide' column to supplement.",
)
@click.option(
    "-n",
    "--n-guides",
    default=3,
    type=int,
    show_default=True,
    help=(
        "Target total coding-strand guides per gene across both libraries. "
        "Genes with fewer than n existing guides receive add-on guides to reach n."
    ),
)
@click.option(
    "--cache-dir",
    default="off_dics",
    show_default=True,
    help="Directory to cache off-target dictionaries.",
)
@click.option(
    "--ref-name",
    default=None,
    help="Genome name for caching off-target dicts. Default: first record ID.",
)
@click.option(
    "--badseeds",
    default=None,
    help=(
        "Comma-separated bad seed sequences to flag, or 'none' to disable. "
        "Auto-detects E. coli defaults if not provided."
    ),
)
@click.option(
    "--report",
    default=None,
    type=click.Path(),
    help="Path to write an HTML report.",
)
@click.option(
    "-o",
    "--output",
    default=None,
    help="Output CSV file path. If not set, prints to stdout.",
)
@pass_config
def generate_addon(
    config,
    ref,
    library_csv,
    n_guides,
    cache_dir,
    ref_name,
    badseeds,
    report,
    output,
):
    """Generate an add-on library to complement an existing guide library.

    Maps the existing library against the target genome and, for each gene with
    fewer than n coding-strand guides, generates new guides to bring the total
    up to n. Guides already present in the existing library are not repeated.

    \b
    Examples:
      crisprbact library generate-addon --ref strain.gb -l core_lib.csv -o addon.csv
      crisprbact library generate-addon --ref strain.gb -l core_lib.csv \\
          -n 5 --report addon_report.html -o addon.csv
    """
    from crisprbact.addon_library import generate_addon_library

    seeds = _parse_badseeds(badseeds)

    addon_lib = generate_addon_library(
        ref_file=ref,
        existing_library=library_csv,
        n=n_guides,
        cache_dir=cache_dir,
        ref_name=ref_name,
        badseeds=seeds,
        output_csv=output,
        output_report=report,
    )

    if not output:
        click.echo(addon_lib.to_csv(index=False))


@library.command("map")
@click.option(
    "--ref",
    type=click.Path(exists=True),
    required=True,
    help="Path to GenBank file of the reference genome.",
)
@click.option(
    "-l",
    "--library",
    "library_csv",
    type=click.Path(exists=True),
    required=True,
    help="Input CSV file with a 'guide' column.",
)
@click.option(
    "--pam",
    default="NGG",
    show_default=True,
    help="PAM sequence (IUPAC). Off-target analysis always uses NGG dicts.",
)
@click.option(
    "--cache-dir",
    default="off_dics",
    show_default=True,
    help="Directory to cache off-target dictionaries.",
)
@click.option(
    "--ref-name",
    default=None,
    help="Genome name for caching off-target dicts. Default: first record ID.",
)
@click.option(
    "--pam-index-cache",
    default=None,
    type=click.Path(),
    help="Path to cache the PAM index (pickle). Speeds up repeated runs.",
)
@click.option(
    "--badseeds",
    default=None,
    help=(
        "Comma-separated bad seed sequences to flag, or 'none' to disable. "
        "Auto-detects E. coli defaults if not provided."
    ),
)
@click.option(
    "--report",
    default=None,
    type=click.Path(),
    help="Path to write an HTML quality report.",
)
@click.option(
    "-o",
    "--output",
    default=None,
    help="Output CSV file path. If not set, prints to stdout.",
)
@pass_config
def map_library_cmd(
    config, ref, library_csv, pam, cache_dir, ref_name, pam_index_cache, badseeds,
    report, output,
):
    """Map an existing guide library against a genome.

    Finds each guide's PAM site in the genome, predicts on-target activity,
    counts off-targets, and flags bad seeds. Useful for cross-strain evaluation.

    \b
    Examples:
      crisprbact library map --ref genome.gb -l library.csv -o mapped.csv
      crisprbact library map --ref genome.gb -l library.csv --pam NGA -o mapped.csv
      crisprbact library map --ref genome.gb -l library.csv \\
          --pam-index-cache pam_index.pkl --report report.html -o mapped.csv
    """
    from crisprbact.map_library import map_library

    seeds = _parse_badseeds(badseeds)

    result = map_library(
        guides=library_csv,
        ref_file=ref,
        pam=pam,
        cache_dir=cache_dir,
        ref_name=ref_name,
        badseeds=seeds,
        pam_index_cache=pam_index_cache,
        output_csv=output,
        output_report=report,
    )

    if not output:
        click.echo(result.to_csv(index=False))


if __name__ == "__main__":
    main()
