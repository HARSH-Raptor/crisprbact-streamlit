"""Visualization tools for CRISPRi library output.

Provides genome-wide plots (matplotlib) and gene-level text inspection
functions for verifying guide RNA designs.
"""

import base64
import html as html_module
import io
import random

import numpy as np

from crisprbact.off_target import FEATURE_TYPES
from crisprbact.predict import SCORE_BOUNDARIES
from crisprbact.utils import rev_comp

from scripts._column_defs import (
    CORE_COLUMNS,
    LIBRARY_PRIMARY_COLUMNS,
    MAP_COLUMNS,
)


_HTML_BASE_CSS = """\
  body { font-family: sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; }
  h1 { color: #333; }
  h2 { color: #555; border-bottom: 1px solid #ccc; padding-bottom: 5px; }
  pre { background: #f5f5f5; padding: 15px; border-radius: 5px;
        overflow-x: auto; font-size: 0.85em; }
  img { display: block; margin: 10px 0; }
  .plot-row { display: flex; gap: 20px; align-items: flex-start; margin-bottom: 10px; }
  .plot-cell { flex: 1; min-width: 0; }
  details { margin: 10px 0; }
  details summary { cursor: pointer; font-weight: bold; color: #555; padding: 5px 0; }
  details summary:hover { color: #333; }
  details[open] summary { margin-bottom: 10px; }
  table.stats { border-collapse: collapse; margin: 10px 0; font-size: 0.9em; }
  table.stats th, table.stats td { padding: 6px 14px; border: 1px solid #ddd; text-align: left; }
  table.stats thead tr { background: #4a90d9; color: #fff; }
  table.stats tr:nth-child(even) td { background: #f5f8ff; }
  .warn { background: #fff8e1; border-left: 4px solid #f9a825; padding: 10px; }
  table.cov { border-collapse: collapse; margin-bottom: 20px; }
  table.cov th, table.cov td { padding: 8px 16px; border: 1px solid #ddd; text-align: left; }
  table.cov thead tr { background: #4a90d9; color: #fff; }
  table.cov th.combined { background: #2e7d32; }
  table.cov tr.alt td { background: #f5f8ff; }
  table.cov td.metric { font-weight: 500; }
  table.cov td.combined { background: #e8f5e9; font-weight: bold; }\
"""


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install crisprbact[viz]"
        )


# ---------------------------------------------------------------------------
# Genome-wide plot functions
# ---------------------------------------------------------------------------


def plot_score_distribution(library, score_col=None):
    """Histogram of on-target scores with random-guide quartile boundaries.

    Vertical lines show the quartile boundaries from ``predict.SCORE_BOUNDARIES``
    (computed from 1000 random sequences). A good library should be shifted to
    the right.

    Parameters
    ----------
    library : DataFrame
        Output of generate_library() or generate_core_library(). Must have a
        ``score`` column, or ``mean_score`` when ``score_col`` is not given.
    score_col : str or None
        Column to plot. If None, uses ``mean_score`` when present, else ``score``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if score_col is None:
        score_col = "mean_score" if "mean_score" in library.columns else "score"
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    scores = library[score_col].values

    ax.hist(scores, bins=50, edgecolor="black", alpha=0.7)

    q1, q2, q3 = SCORE_BOUNDARIES
    for val, label in [(q1, "Q1/Q2"), (q2, "Q2/Q3"), (q3, "Q3/Q4")]:
        ax.axvline(val, color="red", linestyle="--", alpha=0.7)
        ax.text(
            val, ax.get_ylim()[1] * 0.95,
            f" {label}", fontsize=8, color="red",
        )

    xlabel = "Mean on-target score (across strains)" if score_col == "mean_score" else "On-target score"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of guides")
    ax.set_title("Score distribution (dashed lines = random-guide quartile boundaries)")
    fig.tight_layout()
    return fig


def plot_guides_per_gene(library):
    """Histogram of number of guides per gene.

    Parameters
    ----------
    library : DataFrame
        Output of generate_library(), must have 'locus_tag' column.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))

    counts = library.groupby("locus_tag").size()
    vc = counts.value_counts().sort_index()
    ax.bar(vc.index, vc.values, width=0.8, edgecolor="black", alpha=0.7)

    ax.set_xlabel("Number of guides per gene")
    ax.set_ylabel("Number of genes")
    ax.set_title("Guides per gene distribution")
    fig.tight_layout()
    return fig


def plot_off_target_distribution(library):
    """Histogram of off-target counts per guide.

    Parameters
    ----------
    library : DataFrame
        Output of generate_library(), must have 'ntargets' column.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(library["ntargets"].values, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Total off-targets (ntargets)")
    axes[0].set_ylabel("Number of guides")
    axes[0].set_title("ntargets distribution")

    if "noff_12" in library.columns:
        axes[1].hist(library["noff_12"].values, bins=50, edgecolor="black", alpha=0.7)
        axes[1].set_xlabel("12-mer off-targets (noff_12)")
        axes[1].set_ylabel("Number of guides")
        axes[1].set_title("noff_12 distribution")

    fig.tight_layout()
    return fig


def plot_genome_coverage(library, records):
    """Scatter plot of guide positions across the genome.

    Parameters
    ----------
    library : DataFrame
        Output of generate_library(), must have 'pos' and 'score' columns.
    records : list of SeqRecord
        Parsed genome records.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    n_recs = len(records)
    fig, axes = plt.subplots(
        n_recs, 1, figsize=(14, 3 * n_recs), squeeze=False
    )

    for i, rec in enumerate(records):
        ax = axes[i, 0]
        rec_lib = library[library["recid"] == rec.id]
        if len(rec_lib) > 0:
            ax.scatter(
                rec_lib["pos"].values,
                rec_lib["score"].values,
                s=1,
                alpha=0.3,
            )
        ax.set_xlim(0, len(rec.seq))
        ax.set_xlabel("Genome position")
        ax.set_ylabel("Score")
        ax.set_title(f"Coverage: {rec.id}")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gene-level text inspection functions
# ---------------------------------------------------------------------------


def _find_gene_feature(records, query):
    """Find a gene feature by locus_tag or gene name."""
    for rec in records:
        for feat in rec.features:
            if feat.type != "gene":
                continue
            quals = feat.qualifiers
            locus_tag = quals.get("locus_tag", [None])[0]
            gene_name = quals.get("gene", [None])[0]
            if query in (locus_tag, gene_name):
                return rec, feat
    return None, None


def inspect_gene(library, records, query, context=10, score_col="score"):
    """Text-based view of guides targeting a gene with sequence context.

    Shows the coding strand (5'->3') with PAM (CCN) in lowercase and the
    guide RNA sequence. The guide RNA binds the coding strand (non-template)
    for CRISPRi. On the coding strand, the PAM appears as CCN (complement
    of NGG on the template strand), immediately upstream of the guide
    binding site.

    Parameters
    ----------
    library : DataFrame
        Output of generate_library().
    records : list of SeqRecord
        Parsed genome records.
    query : str
        Gene name or locus_tag to inspect.
    context : int
        Number of bases to show on each side of the guide+PAM.
    score_col : str
        Column name for the score (default ``"score"``).

    Returns
    -------
    str
        Formatted text showing gene info and guide alignments.
    """
    rec, feat = _find_gene_feature(records, query)
    if feat is None:
        return f"Gene '{query}' not found in records."

    quals = feat.qualifiers
    locus_tag = quals.get("locus_tag", [None])[0]
    gene_name = quals.get("gene", ["N/A"])[0]
    gene_start = int(feat.location.start)
    gene_end = int(feat.location.end)
    gene_strand = feat.location.strand
    gene_len = gene_end - gene_start

    # Find guides for this gene
    gene_guides = library[
        (library["locus_tag"] == locus_tag)
        | (library["gene"] == query)
    ]
    if "gene_rank" in gene_guides.columns:
        gene_guides = gene_guides.sort_values("gene_rank")
    elif score_col in gene_guides.columns:
        gene_guides = gene_guides.sort_values(score_col, ascending=False)

    if len(gene_guides) == 0:
        return f"No guides found for gene '{query}' (locus_tag={locus_tag})."

    lines = []
    strand_str = "+" if gene_strand == 1 else "-"
    lines.append(
        f"Gene: {gene_name} | locus_tag: {locus_tag} | "
        f"{gene_start}-{gene_end} ({strand_str}) | length: {gene_len} bp"
    )
    lines.append("=" * 80)

    genome_seq = str(rec.seq)
    genome_len = len(genome_seq)

    for i, (_, row) in enumerate(gene_guides.iterrows()):
        rank = int(row["gene_rank"]) if "gene_rank" in row.index else i + 1
        guide_seq = row["guide"]
        guide_strand = row["strand"]
        pam_pos = int(row["pos"])
        score = row[score_col]
        ntargets = int(row.get("ntargets", 0))
        noff_12 = int(row.get("noff_12", 0))

        lines.append("")
        lines.append(
            f"Guide #{rank} | score: {score:.3f} | strand: {guide_strand} | "
            f"pos: {pam_pos} | ntargets: {ntargets} | noff_12: {noff_12}"
        )
        lines.append(f"  guide: 5'-{guide_seq}-3'")
        lines.append("-" * 80)

        # Determine region of + strand to extract.
        # pam_pos is the genome (+) strand coordinate of the N in NGG.
        #
        # For guide_strand "+": on + strand, [guide 20nt][NGG]
        #   guide at [pam_pos-20, pam_pos), NGG at [pam_pos, pam_pos+3)
        #
        # For guide_strand "-": on + strand, [CCN'][guide_RC 20nt]
        #   CCN' at [pam_pos-2, pam_pos+1), guide_RC at [pam_pos+1, pam_pos+21)
        if guide_strand == "+":
            region_start = max(0, pam_pos - 20 - context)
            region_end = min(genome_len, pam_pos + 3 + context)
        else:
            region_start = max(0, pam_pos - 2 - context)
            region_end = min(genome_len, pam_pos + 21 + context)

        plus_region = genome_seq[region_start:region_end]

        # Get coding strand (5'->3')
        if gene_strand == 1:
            coding = plus_region
            coord_start = region_start - gene_start
        else:
            coding = rev_comp(plus_region)
            coord_start = gene_end - region_end

        # Find the guide's reverse complement on the coding strand.
        # On the coding strand, the target always appears as:
        #   5'- ...CCN [rev_comp(guide)]... -3'
        # because the PAM (NGG) is on the template strand.
        guide_rc = rev_comp(guide_seq)
        idx = coding.find(guide_rc)

        if idx >= 3:
            _format_coding_strand(
                lines, coding, idx, idx + 20, idx - 3, idx,
                coord_start, guide_seq,
            )
        else:
            lines.append(f"  coding 5'-{coding}-3'")
            lines.append("  (could not locate guide pattern)")

    lines.append("")
    return "\n".join(lines)


def _format_coding_strand(
    lines, coding_seq, guide_start, guide_end, pam_start, pam_end,
    start_coord, guide_seq,
):
    """Format the coding strand with PAM highlighted and guide RNA aligned.

    Shows the coding strand 5'->3', base-pairing indicators, and the guide
    RNA 3'->5' aligned underneath. The PAM (CCN) is shown in lowercase.

    Parameters
    ----------
    coding_seq : str
        Coding strand sequence, 5'->3'.
    guide_start, guide_end : int
        Offsets of the guide binding region (rev_comp of guide RNA).
    pam_start, pam_end : int
        Offsets of the CCN PAM on the coding strand.
    start_coord : int
        Gene-relative coordinate of the first base in the display.
    guide_seq : str
        The guide RNA sequence (5'->3').
    """
    # PAM in lowercase
    chars = list(coding_seq)
    for i in range(max(0, pam_start), min(len(chars), pam_end)):
        chars[i] = chars[i].lower()
    coding_display = "".join(chars)

    # Base-pairing line: pipes under the guide region
    pairing = list(" " * len(coding_seq))
    for i in range(max(0, guide_start), min(len(pairing), guide_end)):
        pairing[i] = "|"
    pairing_display = "".join(pairing)

    # Guide RNA displayed 3'->5' (reversed), aligned under guide region
    guide_reversed = guide_seq[::-1]  # 3'->5'
    guide_line = list(" " * len(coding_seq))
    for j, c in enumerate(guide_reversed):
        idx = guide_start + j
        if 0 <= idx < len(guide_line):
            guide_line[idx] = c
    guide_display = "".join(guide_line)

    end_coord = start_coord + len(coding_seq)
    pad = 12

    lines.append(f"{'':>{pad}}{start_coord:<30}{end_coord}")
    lines.append(f"  coding 5'-{coding_display}-3'")
    lines.append(f"{'':>{pad}}{pairing_display}")
    lines.append(f"  guide  3'-{guide_display}-5'")
    lines.append(f"{'':>{12}}{'':>{pam_start}}PAM")


def _fig_to_base64(fig):
    """Render a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt = _require_matplotlib()
    plt.close(fig)
    return base64.b64encode(buf.read()).decode("ascii")


def _img_tag(b64):
    """Return an <img> tag for a base64-encoded PNG."""
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;" />'


def _plot_row(pairs):
    """Return HTML for one or more plots arranged side by side.

    Parameters
    ----------
    pairs : list of (title, b64) tuples
        One element → full-width. Two elements → side-by-side flex row.
    """
    if len(pairs) == 1:
        title, b64 = pairs[0]
        return f"<h2>{html_module.escape(title)}</h2>\n{_img_tag(b64)}\n"
    cells = "".join(
        f'<div class="plot-cell"><h2>{html_module.escape(t)}</h2>{_img_tag(b)}</div>'
        for t, b in pairs
    )
    return f'<div class="plot-row">{cells}</div>\n'


def _all_gene_info(records):
    """Return a dict of locus_tag -> (gene_name, size_bp, function) for all genes.

    Function is extracted from the matching CDS feature if available.
    """
    # First pass: collect function from CDS features
    cds_functions = {}
    for rec in records:
        for feat in rec.features:
            if feat.type == "CDS":
                lt = feat.qualifiers.get("locus_tag", [None])[0]
                if lt is not None:
                    func = feat.qualifiers.get("function", [None])[0]
                    if func is None:
                        func = feat.qualifiers.get("product", [None])[0]
                    cds_functions[lt] = func

    # Second pass: collect gene features
    genes = {}
    for rec in records:
        for feat in rec.features:
            if feat.type != "gene":
                continue
            quals = feat.qualifiers
            lt = quals.get("locus_tag", [None])[0]
            if lt is not None:
                name = quals.get("gene", ["N/A"])[0]
                size = int(feat.location.end) - int(feat.location.start)
                func = cds_functions.get(lt)
                genes[lt] = (name, size, func)
    return genes


# ---------------------------------------------------------------------------
# Glossary — single source of truth for metric definitions
# ---------------------------------------------------------------------------

def _build_glossary():
    """Build glossary from column definitions + extra entries."""
    glossary = {}
    for col_list in (LIBRARY_PRIMARY_COLUMNS, MAP_COLUMNS, CORE_COLUMNS):
        for name, desc in col_list:
            glossary.setdefault(name, desc)
    # Core-specific extras
    glossary["gps"] = (
        "Global Penalty Score (GPS) — a composite ranking metric that "
        "balances three normalised penalties:\n"
        "  GPS = off_score_norm + eff_score_norm + cov_score\n"
        "Where off_score_norm penalises off-target hits, eff_score_norm "
        "penalises low-activity guides, and cov_score penalises incomplete "
        "strain coverage. Lower GPS = better guide."
    )
    glossary["mean_score"] = (
        "Mean predicted on-target activity across all strain matches."
    )
    return glossary


_GLOSSARY = _build_glossary()


# ---------------------------------------------------------------------------
# Shared HTML helper functions
# ---------------------------------------------------------------------------


def _unified_title_html(report_type, genome_name):
    """Return HTML for a unified report title."""
    title = f"<h1>CRISPRbact Report: {html_module.escape(report_type)}</h1>\n"
    if genome_name:
        title += (
            f"<p><strong>Genome:</strong> {html_module.escape(genome_name)}</p>\n"
        )
    return title


def _genome_summary_html(records):
    """Return an HTML <pre> block with genome summary stats."""
    total_bp = sum(len(rec.seq) for rec in records)
    all_genes = _all_gene_info(records)

    # Count features by type (CDS, tRNA, rRNA, ncRNA)
    _COUNTED_TYPES = ["CDS", "tRNA", "rRNA", "ncRNA"]
    type_counts = {t: 0 for t in _COUNTED_TYPES}
    for rec in records:
        for feat in rec.features:
            if feat.type in type_counts:
                type_counts[feat.type] += 1
    type_str = " · ".join(f"{t}: {c}" for t, c in type_counts.items() if c > 0)

    gene_line = f"Total genes:  {len(all_genes)}"
    if type_str:
        gene_line += f"  ({type_str})"

    lines = [
        f"Records:      {len(records)}",
        f"Genome size:  {total_bp:,} bp",
        gene_line,
    ]
    return "<pre>" + html_module.escape("\n".join(lines)) + "</pre>"


def _off_target_stats_html(df):
    """Return an HTML stats table showing off-target statistics.

    Checks which columns exist and adapts accordingly.
    """
    rows = []
    total = len(df)
    if total == 0:
        return "<p>No guides to analyse.</p>"

    checks = [
        ("ntargets", "Guides with ntargets > 1", lambda s: (s > 1).sum()),
        ("noff_12", "Guides with noff_12 > 0", lambda s: (s > 0).sum()),
        ("noff_11_gene", "Guides with noff_11_gene > 0", lambda s: (s > 0).sum()),
        ("noff_9_prom", "Guides with noff_9_prom > 0", lambda s: (s > 0).sum()),
    ]
    for col, label, counter in checks:
        if col in df.columns:
            count = int(counter(df[col]))
            pct = count / total * 100
            rows.append(f"<tr><td>{html_module.escape(label)}</td>"
                        f"<td>{count}</td><td>{pct:.1f}%</td></tr>")

    # Fraction-based columns (core)
    frac_checks = [
        ("off_11_gene_score", "Mean off_11_gene_score"),
        ("off_9_prom_score", "Mean off_9_prom_score"),
    ]
    for col, label in frac_checks:
        if col in df.columns:
            mean_val = df[col].mean()
            rows.append(f"<tr><td>{html_module.escape(label)}</td>"
                        f"<td colspan='2'>{mean_val:.3f}</td></tr>")

    if not rows:
        return ""

    return (
        '<table class="stats">\n'
        "<thead><tr><th>Metric</th><th>Count</th><th>%</th></tr></thead>\n"
        "<tbody>" + "\n".join(rows) + "</tbody>\n"
        "</table>"
    )


def _position_summary_html(df):
    """Return HTML paragraph with first/second half targeting fractions.

    Returns an empty string when ``second_half_gene`` is absent.
    """
    if "second_half_gene" not in df.columns:
        return ""
    total = len(df)
    if total == 0:
        return ""
    n_second = int(df["second_half_gene"].sum())
    n_first = total - n_second
    pct_first = n_first / total * 100
    pct_second = n_second / total * 100
    return (
        f"<p>Gene position: "
        f"first half <strong>{n_first}</strong> ({pct_first:.1f}%) · "
        f"second half <strong>{n_second}</strong> ({pct_second:.1f}%)</p>"
    )


def _badseed_summary_html(df):
    """Return HTML paragraph with bad seed count and percentage."""
    if "inbadseeds" not in df.columns:
        return "<p>Bad seed information not available.</p>"
    total = len(df)
    if total == 0:
        return "<p>No guides to analyse.</p>"
    count = int(df["inbadseeds"].sum())
    pct = count / total * 100
    return f"<p>Flagged guides: <strong>{count}</strong> ({pct:.1f}%)</p>"


def _gene_coverage_section(df, records, n=None, group_col="locus_tag"):
    """Return HTML showing gene coverage details.

    For each bucket (0, 1, 2 guides):
    - ≤20 genes: list always visible
    - 21–100 genes: list collapsed inside ``<details>``
    - >100 genes: count only, no list
    """
    all_genes = _all_gene_info(records)
    total_genes = len(all_genes)
    if total_genes == 0:
        return "<p>No genes found in genome records.</p>"

    guides_per_gene = df.groupby(group_col).size()
    n_covered = sum(1 for lt in all_genes if guides_per_gene.get(lt, 0) > 0)

    parts = [f"<p><strong>{n_covered} / {total_genes}</strong> genes covered"]
    if n is not None:
        parts[0] += f" (requested {n} guides/gene)"
    parts[0] += "</p>"

    # Group genes by guide count and sort
    buckets = {}
    for lt, (name, size, func) in all_genes.items():
        cnt = int(guides_per_gene.get(lt, 0))
        buckets.setdefault(cnt, []).append((lt, name, size, func))
    for cnt in buckets:
        buckets[cnt].sort(key=lambda x: x[0])

    def _gene_list_pre(genes, css_class=""):
        lines = [
            f"  {lt}  ({name}): {size} bp" + (f" - {func}" if func else "")
            for lt, name, size, func in genes
        ]
        cls = f' class="{css_class}"' if css_class else ""
        return f"<pre{cls}>" + html_module.escape("\n".join(lines)) + "</pre>"

    def _bucket_html(cnt, label):
        if cnt not in buckets:
            return ""
        genes = buckets[cnt]
        n_genes = len(genes)
        css = "warn" if cnt == 0 else ""
        header = f"Genes with {label} ({n_genes})"
        if n_genes > 100:
            # Count only
            return f"<p><strong>{header}</strong></p>"
        elif n_genes > 20:
            # Collapsible
            return (
                f"<details><summary>{header}</summary>\n"
                + _gene_list_pre(genes, css)
                + "\n</details>"
            )
        else:
            # Always visible
            return f"<p><strong>{header}</strong></p>\n" + _gene_list_pre(genes, css)

    for cnt, label in [(0, "0 guides"), (1, "1 guide"), (2, "2 guides")]:
        html = _bucket_html(cnt, label)
        if html:
            parts.append(html)

    return "\n".join(parts)


def _glossary_html(metrics=None):
    """Return a collapsible glossary section.

    Parameters
    ----------
    metrics : list of str or None
        If given, only show these metrics. Otherwise show all.
    """
    items = []
    for key, desc in _GLOSSARY.items():
        if metrics is not None and key not in metrics:
            continue
        items.append(
            f"<dt><strong>{html_module.escape(key)}</strong></dt>"
            f"<dd>{html_module.escape(desc)}</dd>"
        )
    if not items:
        return ""
    return (
        "<details>\n<summary>Glossary</summary>\n"
        "<dl>\n" + "\n".join(items) + "\n</dl>\n"
        "</details>"
    )


def _gene_inspection_html(df, records, score_col="score"):
    """Pick one random + and one - strand gene, return formatted HTML."""
    if "gene_ori" not in df.columns or df["gene_ori"].isna().all():
        return ""
    parts = []
    plus_genes = df[df["gene_ori"] == 1]["locus_tag"].unique()
    minus_genes = df[df["gene_ori"] == -1]["locus_tag"].unique()
    examples = []
    if len(plus_genes) > 0:
        examples.append(random.choice(plus_genes))
    if len(minus_genes) > 0:
        examples.append(random.choice(minus_genes))
    for lt in examples:
        text = inspect_gene(df, records, lt, score_col=score_col)
        parts.append(
            f"<h3>{html_module.escape(lt)}</h3>\n"
            f"<pre>{html_module.escape(text)}</pre>\n"
        )
    return "\n".join(parts)


def generate_report(library, records, n, output_path, badseeds=None):
    """Generate a self-contained HTML report for a CRISPRi library.

    Parameters
    ----------
    library : DataFrame
        Output of generate_library().
    records : list of SeqRecord
        Parsed genome records.
    n : int
        Requested number of guides per gene.
    output_path : str or path-like
        Path to write the HTML report.
    badseeds : list of str, optional
        Bad seed sequences that were used during library design. Shown in the
        Bad Seeds section of the report.
    """
    genome_name = records[0].description if records else "Unknown"
    all_genes = _all_gene_info(records)
    total_genes = len(all_genes)
    total_guides = len(library)
    n_covered = library["locus_tag"].nunique()

    # --- Generate plots ---
    score_b64 = _fig_to_base64(plot_score_distribution(library))
    gpg_b64 = _fig_to_base64(plot_guides_per_gene(library))
    cov_b64 = _fig_to_base64(plot_genome_coverage(library, records))

    plot_html = (
        _plot_row([("Score distribution", score_b64), ("Guides per gene", gpg_b64)])
        + _plot_row([("Genome coverage", cov_b64)])
    )

    # --- Library summary ---
    summary_html = (
        "<pre>"
        + html_module.escape(
            f"Total guides:   {total_guides}\n"
            f"Genes covered:  {n_covered} / {total_genes}\n"
            f"Requested N:    {n} guides per gene"
        )
        + "</pre>"
    )

    # --- Sections via shared helpers ---
    glossary_metrics = [
        "score", "score_quartile", "ntargets", "noff_12",
        "noff_11_gene", "noff_9_prom", "inbadseeds", "second_half_gene",
        "targets_coding_strand",
    ]

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>CRISPRbact Report: Library Design - {html_module.escape(genome_name)}</title>
<style>
{_HTML_BASE_CSS}
</style>
</head>
<body>
{_unified_title_html("Library Design", genome_name)}

<h2>Genome Summary</h2>
{_genome_summary_html(records)}

<h2>Library Summary</h2>
{summary_html}
{_position_summary_html(library)}

{plot_html}

<h2>Off-target Statistics</h2>
{_off_target_stats_html(library)}

<h2>Bad Seeds</h2>
{("<p>Seeds checked: <code>" + html_module.escape(", ".join(badseeds)) + "</code></p>") if badseeds else ""}
{_badseed_summary_html(library)}

<h2>Gene Coverage</h2>
{_gene_coverage_section(library, records, n=n)}

<h2>Gene Inspection Examples</h2>
<p>Random sample: one gene on each strand.</p>
{_gene_inspection_html(library, records)}

{_glossary_html(glossary_metrics)}

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Core-genome report helpers
# ---------------------------------------------------------------------------



def _plot_gps_distribution(library):
    """Histogram of GPS (Global Penalty Score) values."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(library["gps"].values, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("GPS (lower = better)")
    ax.set_ylabel("Number of guides")
    ax.set_title("GPS distribution")
    fig.tight_layout()
    return fig


def _plot_coverage_distribution(library):
    """Histogram of per-guide genome coverage fractions."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(library["coverage"].values, bins=20, edgecolor="black", alpha=0.7,
            range=(0, 1))
    ax.set_xlabel("Coverage (fraction of strains)")
    ax.set_ylabel("Number of guides")
    ax.set_title("Guide coverage distribution")
    fig.tight_layout()
    return fig


def _plot_per_genome_families(genome_ids, per_genome_covered, core_families):
    """Horizontal bar chart: number of core families covered per genome."""
    plt = _require_matplotlib()
    n_total = len(core_families)
    counts = [len(per_genome_covered.get(gid, set())) for gid in genome_ids]
    fig, ax = plt.subplots(figsize=(10, max(4, len(genome_ids) * 0.4 + 1)))
    y = range(len(genome_ids))
    ax.barh(list(y), counts, color="steelblue", alpha=0.8)
    ax.axvline(n_total, color="red", linestyle="--", alpha=0.7, label="Total core families")
    ax.set_yticks(list(y))
    ax.set_yticklabels(genome_ids, fontsize=8)
    ax.set_xlabel("Families covered by guides")
    ax.set_title("Per-genome families covered")
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_guides_per_family(library):
    """Histogram: number of guides selected per family."""
    plt = _require_matplotlib()
    counts = library.groupby("family_id").size()
    vc = counts.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(vc.index, vc.values, width=0.8, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Guides per family")
    ax.set_ylabel("Number of families")
    ax.set_title("Guides per family distribution")
    fig.tight_layout()
    return fig


def _plot_gene_type_breakdown(all_records, genome_ids):
    """Stacked horizontal bar chart of CDS / ncRNA / rRNA / tRNA per genome."""
    plt = _require_matplotlib()
    types = FEATURE_TYPES
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    data = {t: [] for t in types}
    for gid in genome_ids:
        counts = {t: 0 for t in types}
        for rec in all_records[gid]:
            for feat in rec.features:
                if feat.type in counts:
                    counts[feat.type] += 1
        for t in types:
            data[t].append(counts[t])

    fig, ax = plt.subplots(figsize=(10, max(4, len(genome_ids) * 0.4 + 1)))
    y = list(range(len(genome_ids)))
    lefts = [0] * len(genome_ids)
    for t, color in zip(types, colors):
        vals = data[t]
        ax.barh(y, vals, left=lefts, label=t, color=color, alpha=0.8)
        lefts = [l + v for l, v in zip(lefts, vals)]

    ax.set_yticks(y)
    ax.set_yticklabels(genome_ids, fontsize=8)
    ax.set_xlabel("Number of features")
    ax.set_title("Gene-type breakdown per genome")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def _plot_feature_type_coverage(all_records, genome_ids, core_families, per_genome_covered):
    """Per-genome coverage fraction by feature type.

    Scatter plot (with jitter) when ≤10 genomes; violin plot when >10.
    """
    plt = _require_matplotlib()
    rng = random.Random(0)  # deterministic jitter

    feature_types = FEATURE_TYPES
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    # Build fractions[ft] = list of per-genome coverage fractions
    fractions = {ft: [] for ft in feature_types}

    for gid in genome_ids:
        recs = all_records[gid]
        lt_to_type = {}
        for rec in recs:
            for feat in rec.features:
                if feat.type in feature_types:
                    lt = feat.qualifiers.get("locus_tag", [None])[0]
                    if lt is not None and lt not in lt_to_type:
                        lt_to_type[lt] = feat.type

        covered_lts = set()
        for fid in per_genome_covered.get(gid, set()):
            if fid in core_families and gid in core_families[fid]:
                covered_lts.update(core_families[fid][gid])

        for ft in feature_types:
            total = sum(1 for t in lt_to_type.values() if t == ft)
            covered = sum(1 for lt, t in lt_to_type.items() if t == ft and lt in covered_lts)
            fractions[ft].append(covered / total if total > 0 else float("nan"))

    fig, ax = plt.subplots(figsize=(5, 4))
    x_pos = list(range(len(feature_types)))

    if len(genome_ids) > 10:
        data = [[v for v in fractions[ft] if not (v != v)] for ft in feature_types]  # drop NaN
        parts = ax.violinplot(data, positions=x_pos, showmedians=True, showextrema=True)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        for key in ("cmedians", "cbars", "cmaxes", "cmins"):
            if key in parts:
                parts[key].set_color("black")
    else:
        jitter = 0.15
        for i, (ft, color) in enumerate(zip(feature_types, colors)):
            ys = fractions[ft]
            xs = [i + rng.uniform(-jitter, jitter) for _ in ys]
            ax.scatter(xs, ys, color=color, alpha=0.7, s=50, zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_types)
    ax.set_ylabel("Fraction of genes covered")
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Coverage by feature type")
    fig.tight_layout()
    return fig


def generate_core_report(
    library, all_records, genome_ids, core_families,
    per_genome_covered, n, output_path, gene_name_map=None,
    family_feature_types=None,
):
    """Generate a self-contained HTML report for a core-genome CRISPRi library.

    Parameters
    ----------
    library : DataFrame
        Output of generate_core_library().
    all_records : dict
        {genome_id: list of SeqRecord}.
    genome_ids : list of str
        Ordered list of genome IDs.
    core_families : dict
        {family_id: {genome_id: [locus_tags]}} as returned by find_core_families().
    per_genome_covered : dict
        {genome_id: set of family_ids covered by at least one selected guide}.
    n : int
        Requested number of guides per family.
    output_path : str or path-like
        Path to write the HTML report.
    gene_name_map : dict or None
        Pre-computed {family_id: gene_name} mapping. When None, recomputed
        from all_records (slower path for standalone calls).
    family_feature_types : dict or None
        {family_id: feature_type_str} mapping. When None, all families are
        assumed to be CDS.
    """
    n_genomes = len(genome_ids)
    n_core_families = len(core_families)
    families_in_library = set(library["family_id"].unique()) if len(library) else set()
    n_families_with_guides = len(families_in_library)
    n_families_no_guides = n_core_families - n_families_with_guides
    total_guides = len(library)

    # --- Summary ---
    summary_lines = [
        f"Genomes:                   {n_genomes}",
        f"Core families found:       {n_core_families}",
        f"Families with guides:      {n_families_with_guides}",
        f"Families without guides:   {n_families_no_guides}",
        f"Guides requested (n):      {n}",
        f"Total guides in library:   {total_guides}",
    ]
    if len(library):
        score_col = "mean_score" if "mean_score" in library.columns else "score"
        scores = library[score_col].values
        score_label = "Mean score" if score_col == "mean_score" else "Score"
        summary_lines.append(
            f"{score_label} min/max/mean/median: "
            f"{scores.min():.3f} / {scores.max():.3f} / "
            f"{scores.mean():.3f} / {np.median(scores):.3f}"
        )
        if "gps" in library.columns:
            gps = library["gps"].values
            summary_lines.append(
                f"GPS    min/max/mean/median: "
                f"{gps.min():.3f} / {gps.max():.3f} / "
                f"{gps.mean():.3f} / {np.median(gps):.3f}"
            )
        if "coverage" in library.columns:
            cov = library["coverage"].values
            summary_lines.append(
                f"Coverage min/max/mean/median: "
                f"{cov.min():.3f} / {cov.max():.3f} / "
                f"{cov.mean():.3f} / {np.median(cov):.3f}"
            )
            max_cov_count = int((library["coverage"] == 1.0).sum())
            max_cov_pct = max_cov_count / total_guides * 100
            summary_lines.append(
                f"Guides at max coverage:    {max_cov_count} ({max_cov_pct:.1f}%)"
            )
        if len(library):
            per_fam = library.groupby("family_id").size()
            rescue_count = int((per_fam > n).sum())
            summary_lines.append(
                f"Families with 4th-rescue:  {rescue_count}"
            )
    summary_text = "\n".join(summary_lines)

    # --- Per-strain table ---
    feature_types = FEATURE_TYPES

    # Default: all families are CDS if no map provided
    if family_feature_types is None:
        family_feature_types = {fid: "CDS" for fid in core_families}

    def _genome_size_mbp(records):
        return sum(len(rec.seq) for rec in records) / 1e6

    # Total families per feature type (same for all genomes — these are core)
    ft_totals = {}
    for ft in feature_types:
        ft_totals[ft] = sum(
            1 for fft in family_feature_types.values() if fft == ft
        )
    n_total_families = sum(ft_totals.values())

    table_rows = []
    for gid in genome_ids:
        recs = all_records[gid]
        size = _genome_size_mbp(recs)

        # Per-feature-type coverage: how many families have a selected guide
        # matching in this genome
        covered_fids = per_genome_covered.get(gid, set())
        ft_coverage = {}
        for ft in feature_types:
            covered = sum(
                1 for fid, fft in family_feature_types.items()
                if fft == ft and fid in covered_fids
            )
            ft_coverage[ft] = (covered, ft_totals[ft])

        table_rows.append((gid, size, n_total_families, ft_coverage))

    cov_headers = "".join(f"<th>{t}</th>" for t in feature_types)
    table_html = (
        '<table class="stats">\n'
        "<thead><tr>"
        f"<th>Genome ID</th><th>Size (Mbp)</th>"
        f"<th>Core families</th>"
        f"{cov_headers}"
        "</tr></thead>\n<tbody>\n"
    )
    for gid, size, n_fam, ft_cov in table_rows:
        cov_cells = ""
        for ft in feature_types:
            covered, total = ft_cov[ft]
            pct = (covered / total * 100) if total > 0 else 0
            cov_cells += f"<td>{pct:.1f}% ({covered}/{total})</td>"
        table_html += (
            f"<tr>"
            f"<td>{html_module.escape(gid)}</td>"
            f"<td>{size:.2f}</td>"
            f"<td>{n_fam}</td>"
            f"{cov_cells}"
            f"</tr>\n"
        )
    table_html += "</tbody></table>"

    # --- Families without guides ---
    if gene_name_map is None:
        from crisprbact.core_library import _family_gene_names
        gene_name_map = _family_gene_names(core_families, all_records)

    missed_families = [
        fid for fid in core_families if fid not in families_in_library
    ]
    if missed_families:
        missed_lines = []
        for fid in missed_families:
            members = core_families[fid]
            gids_in_fam = list(members.keys())
            gene_name = gene_name_map.get(fid)
            gene_str = f" [{gene_name}]" if gene_name else ""
            missed_lines.append(
                f"  {fid}{gene_str}  ({len(gids_in_fam)} genomes): "
                + ", ".join(html_module.escape(g) for g in gids_in_fam)
            )
        missed_html = (
            f'<pre class="warn">'
            f'{html_module.escape(chr(10).join(missed_lines))}</pre>'
        )
    else:
        missed_html = "<p>All core families have at least one guide.</p>"

    # --- Plots ---
    score_b64 = gps_b64 = ftcov_b64 = None
    if len(library):
        score_b64 = _fig_to_base64(plot_score_distribution(library))
        if "gps" in library.columns:
            gps_b64 = _fig_to_base64(_plot_gps_distribution(library))
    ftcov_b64 = _fig_to_base64(
        _plot_feature_type_coverage(all_records, genome_ids, core_families, per_genome_covered)
    )

    core_plots = []
    if score_b64:
        core_plots.append(("Score distribution", score_b64))
    if gps_b64:
        core_plots.append(("GPS distribution", gps_b64))
    core_plots.append(("Coverage by feature type", ftcov_b64))
    plot_html = _plot_row(core_plots)

    # --- Family coverage ---
    # Covered in at least one strain
    covered_any = set()
    for gid in genome_ids:
        covered_any |= per_genome_covered.get(gid, set())
    # Covered in all strains
    covered_all = set(covered_any)
    for gid in genome_ids:
        covered_all &= per_genome_covered.get(gid, set())

    # Breakdown by feature type
    def _cov_breakdown(covered_set):
        breakdown = {}
        for ft in feature_types:
            total = ft_totals.get(ft, 0)
            covered = sum(
                1 for fid in covered_set
                if family_feature_types.get(fid) == ft
            )
            breakdown[ft] = (covered, total)
        return breakdown

    any_breakdown = _cov_breakdown(covered_any)
    all_breakdown = _cov_breakdown(covered_all)

    family_cov_lines = [
        f"<p><strong>{len(covered_any)} / {n_core_families}</strong> "
        f"families covered in at least one strain "
        f"(requested {n} guides/family)</p>",
        f"<p><strong>{len(covered_all)} / {n_core_families}</strong> "
        f"families covered in all strains</p>",
    ]
    # Feature type breakdown table
    family_cov_lines.append(
        '<table class="stats"><thead><tr>'
        "<th>Feature type</th><th>Total</th>"
        "<th>Covered (≥1 strain)</th><th>Covered (all strains)</th>"
        "</tr></thead><tbody>"
    )
    for ft in feature_types:
        any_c, any_t = any_breakdown[ft]
        all_c, _ = all_breakdown[ft]
        if any_t == 0:
            continue
        any_pct = any_c / any_t * 100 if any_t else 0
        all_pct = all_c / any_t * 100 if any_t else 0
        family_cov_lines.append(
            f"<tr><td>{ft}</td><td>{any_t}</td>"
            f"<td>{any_pct:.1f}% ({any_c}/{any_t})</td>"
            f"<td>{all_pct:.1f}% ({all_c}/{any_t})</td></tr>"
        )
    family_cov_lines.append("</tbody></table>")
    family_cov_html = "\n".join(family_cov_lines)

    glossary_metrics = [
        "family_id", "guide", "n_covered", "coverage", "gps",
        "off_11_gene_score", "off_9_prom_score", "mean_score",
    ]

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>CRISPRbact Report: Core-Genome Library</title>
<style>
{_HTML_BASE_CSS}
  body {{ max-width: 1200px; }}
  table {{ margin: 10px 0; }}
</style>
</head>
<body>
{_unified_title_html("Core-Genome Library", "")}

<h2>Run Parameters &amp; Summary</h2>
<pre>{html_module.escape(summary_text)}</pre>
{_position_summary_html(library)}

{plot_html}

<h2>Family Coverage</h2>
{family_cov_html}

<h2>Families Without Guides</h2>
{missed_html}

<h2>Per-Strain Table</h2>
{table_html}

{_glossary_html(glossary_metrics)}

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def generate_map_report(result, records, output_path):
    """Generate a self-contained HTML report for a mapped guide library.

    Summarises how well an existing guide library maps to a new genome:
    match rate, on-target score distribution, off-target counts, and bad seeds.

    Parameters
    ----------
    result : DataFrame
        Output of :func:`crisprbact.map_library.map_library`.
    records : list of SeqRecord
        Parsed genome records for the reference genome.
    output_path : str or path-like
        Path to write the HTML report.
    """
    genome_name = records[0].description if records else "Unknown"

    # Per-guide stats (deduplicate: one row per guide)
    guide_df = result.drop_duplicates("guide")
    n_total = len(guide_df)
    n_matched = int((guide_df["n_matches"] > 0).sum())
    n_unmatched = n_total - n_matched
    match_pct = n_matched / n_total * 100 if n_total > 0 else 0.0

    # Matched guides (coding-strand rows for inspection)
    matched_df = result[result["n_matches"] > 0].drop_duplicates("guide")
    matched_coding = result[
        (result["n_matches"] > 0) & (result["targets_coding_strand"] == True)
    ]
    scores = matched_df["on_target_score"].dropna().values

    # --- Library summary ---
    summary_html = (
        "<pre>"
        + html_module.escape(
            f"Total guides in input:  {n_total}\n"
            f"Guides with a target:   {n_matched} ({match_pct:.1f}%)\n"
            f"Guides without target:  {n_unmatched}"
        )
        + "</pre>"
    )

    # --- Gene coverage distribution plot data ---
    all_locus_tags = {
        feat.qualifiers["locus_tag"][0]
        for rec in records
        for feat in rec.features
        if feat.type == "gene" and "locus_tag" in feat.qualifiers
    }
    gene_guide_counts = (
        result[result["locus_tag"].notna()]
        .groupby("locus_tag")["guide"]
        .nunique()
    )
    guides_per_gene = {lt: int(gene_guide_counts.get(lt, 0)) for lt in all_locus_tags}
    n_genes_total = len(all_locus_tags)
    n_genes_covered = sum(1 for c in guides_per_gene.values() if c > 0)

    # --- Wrong-orientation analysis ---
    wrong_ori = result[result["targets_coding_strand"] == False]
    n_wrong_guides = wrong_ori["guide"].nunique()
    n_wrong_genes = wrong_ori["locus_tag"].nunique()

    if len(wrong_ori) > 0:
        wrong_gene_summary = (
            wrong_ori.groupby(["locus_tag", "gene", "gene_ori", "strand"])
            .agg(n_guides=("guide", "nunique"))
            .reset_index()
            .sort_values("n_guides", ascending=False)
        )
        wrong_table_rows = ""
        for _, row in wrong_gene_summary.iterrows():
            gene_name = row["gene"] if str(row["gene"]) != "nan" else "\u2014"
            ori_str = "+" if row["gene_ori"] == 1 else "\u2212"
            wrong_table_rows += (
                f"<tr>"
                f"<td>{html_module.escape(str(row['locus_tag']))}</td>"
                f"<td>{html_module.escape(gene_name)}</td>"
                f"<td>{ori_str}</td>"
                f"<td>{html_module.escape(str(row['strand']))}</td>"
                f"<td>{int(row['n_guides'])}</td>"
                f"</tr>\n"
            )
        wrong_table_html = (
            '<table class="stats">\n'
            "<thead><tr>"
            "<th>Locus tag</th><th>Gene</th>"
            "<th>Gene strand</th><th>Guide strand</th><th>Guides</th>"
            "</tr></thead>\n"
            + wrong_table_rows
            + "</table>"
        )
    else:
        wrong_table_html = "<p>No wrong-orientation matches found.</p>"

    wrong_ori_html = (
        f"<details>\n"
        f"<summary>Wrong-orientation targets "
        f"({n_wrong_guides} guides, {n_wrong_genes} genes)</summary>\n"
        f"<p>These guides have a PAM site within an annotated gene but on "
        f"the <strong>non-coding (template) strand</strong>. CRISPRi blocks "
        f"RNA polymerase on the coding strand; these guides are not expected "
        f"to strongly silence the listed gene.</p>\n"
        f"{wrong_table_html}\n"
        f"</details>"
    )

    # --- Plots ---
    plt = _require_matplotlib()

    # Score distribution (matched guides)
    fig_score, ax = plt.subplots(figsize=(8, 5))
    if len(scores) > 0:
        ax.hist(scores, bins=50, edgecolor="black", alpha=0.7)
        _q1, _q2, _q3 = SCORE_BOUNDARIES
        for val, label in [(_q1, "Q1/Q2"), (_q2, "Q2/Q3"), (_q3, "Q3/Q4")]:
            ax.axvline(val, color="red", linestyle="--", alpha=0.7)
            ax.text(val, ax.get_ylim()[1] * 0.95, f" {label}", fontsize=8, color="red")
    ax.set_xlabel("On-target score")
    ax.set_ylabel("Number of guides")
    ax.set_title("Score distribution (matched guides)")
    fig_score.tight_layout()

    # Gene coverage distribution
    fig_gene_cov, ax3 = plt.subplots(figsize=(8, 5))
    counts_arr = np.array(list(guides_per_gene.values()))
    vals3, cnts3 = np.unique(counts_arr, return_counts=True)
    ax3.bar(vals3, cnts3, width=0.8, edgecolor="black", alpha=0.7)
    ax3.set_xlabel("Number of guides targeting the gene")
    ax3.set_ylabel("Number of genes")
    ax3.set_title(
        f"Gene coverage  ({n_genes_covered}/{n_genes_total} genes with \u22651 guide)"
    )
    fig_gene_cov.tight_layout()

    score_b64 = _fig_to_base64(fig_score)
    gene_cov_b64 = _fig_to_base64(fig_gene_cov)

    plot_html = _plot_row([
        ("Gene coverage", gene_cov_b64),
        ("Score distribution (guides with a target)", score_b64),
    ])

    # --- Sections via shared helpers ---
    glossary_metrics = [
        "n_matches", "on_target_score", "score_quartile", "ntargets",
        "noff_12", "noff_11_gene", "noff_9_prom", "inbadseeds",
        "targets_coding_strand",
    ]

    # Gene inspection on matched coding-strand rows
    inspect_html = _gene_inspection_html(
        matched_coding, records, score_col="on_target_score"
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>CRISPRbact Report: Library Mapping - {html_module.escape(genome_name)}</title>
<style>
{_HTML_BASE_CSS}
</style>
</head>
<body>
{_unified_title_html("Library Mapping", genome_name)}

<h2>Genome Summary</h2>
{_genome_summary_html(records)}

<h2>Library Summary</h2>
{summary_html}
{_position_summary_html(matched_df)}

{plot_html}

<h2>Off-target Statistics</h2>
{_off_target_stats_html(matched_df)}

<h2>Bad Seeds</h2>
{_badseed_summary_html(matched_df)}

<h2>Gene Coverage</h2>
{_gene_coverage_section(result[result["locus_tag"].notna()], records)}

{wrong_ori_html}

<h2>Gene Inspection Examples</h2>
<p>Random sample: one gene on each strand.</p>
{inspect_html}

{_glossary_html(glossary_metrics)}

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def _coverage_table_html(existing_coverage, addon_coverage, all_gene_lts, n):
    """Return an HTML table comparing existing / add-on / combined coverage.

    Parameters
    ----------
    existing_coverage : dict
        {locus_tag: n_guides} for coding-strand guides in the existing library.
    addon_coverage : dict
        {locus_tag: n_guides} for guides in the add-on library.
    all_gene_lts : iterable
        All locus tags in the target genome.
    n : int
        Full-coverage threshold (guides per gene).
    """
    all_lts = list(all_gene_lts)
    total = len(all_lts)

    def _count(cov, threshold):
        return sum(1 for lt in all_lts if cov.get(lt, 0) >= threshold)

    combined = {lt: existing_coverage.get(lt, 0) + addon_coverage.get(lt, 0) for lt in all_lts}

    def _pct(k):
        return f"{k}&thinsp;/&thinsp;{total}&ensp;({100 * k / total:.1f}%)" if total else str(k)

    ex_guides = sum(existing_coverage.get(lt, 0) for lt in all_lts)
    ao_guides = sum(addon_coverage.get(lt, 0) for lt in all_lts)

    rows = [
        ("Total coding-strand guide–gene pairs", str(ex_guides), str(ao_guides), str(ex_guides + ao_guides)),
        ("Genes with ≥1 guide", _pct(_count(existing_coverage, 1)), _pct(_count(addon_coverage, 1)), _pct(_count(combined, 1))),
        (f"Genes fully covered (≥{n} guides)", _pct(_count(existing_coverage, n)), _pct(_count(addon_coverage, n)), _pct(_count(combined, n))),
    ]

    body = ""
    for i, (metric, ex, ao, co) in enumerate(rows):
        bg = " class=\"alt\"" if i % 2 else ""
        body += (
            f"<tr{bg}>"
            f"<td class=\"metric\">{metric}</td>"
            f"<td>{ex}</td>"
            f"<td>{ao}</td>"
            f"<td class=\"combined\">{co}</td>"
            f"</tr>"
        )

    return (
        "<table class=\"cov\">"
        "<thead><tr>"
        "<th>Metric</th><th>Existing library</th><th>Add-on library</th>"
        "<th class=\"combined\">Combined</th>"
        "</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )


def plot_genome_coverage_overlay(existing_df, addon_df, records):
    """Scatter plot overlaying existing and add-on library positions.

    Parameters
    ----------
    existing_df : DataFrame
        Matched rows from map_library() (filtered to n_matches > 0).
        Uses ``on_target_score`` for y-axis.
    addon_df : DataFrame
        Add-on library output. Uses ``score`` for y-axis.
    records : list of SeqRecord
        Parsed genome records.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    n_recs = len(records)
    fig, axes = plt.subplots(
        n_recs, 1, figsize=(14, 3 * n_recs), squeeze=False
    )

    for i, rec in enumerate(records):
        ax = axes[i, 0]
        # Existing library (blue)
        if len(existing_df) > 0 and "recid" in existing_df.columns:
            ex_rec = existing_df[existing_df["recid"] == rec.id]
            if len(ex_rec) > 0 and "pos" in ex_rec.columns:
                score_col = "on_target_score" if "on_target_score" in ex_rec.columns else "score"
                ax.scatter(
                    ex_rec["pos"].values,
                    ex_rec[score_col].values,
                    s=1, alpha=0.3, color="steelblue", label="Existing library",
                )
        # Add-on library (orange)
        if len(addon_df) > 0 and "recid" in addon_df.columns:
            ao_rec = addon_df[addon_df["recid"] == rec.id]
            if len(ao_rec) > 0:
                ax.scatter(
                    ao_rec["pos"].values,
                    ao_rec["score"].values,
                    s=1, alpha=0.3, color="darkorange", label="Add-on library",
                )
        ax.set_xlim(0, len(rec.seq))
        ax.set_xlabel("Genome position")
        ax.set_ylabel("Score")
        ax.set_title(f"Coverage: {rec.id}")
        ax.legend(markerscale=8, loc="upper right")

    fig.tight_layout()
    return fig


def generate_addon_report(
    addon_lib, map_result, records, existing_coverage, n, output_path
):
    """Generate a self-contained HTML report for a CRISPRi add-on library.

    Shows coverage statistics before and after mixing the add-on library
    with the existing library.

    Parameters
    ----------
    addon_lib : DataFrame
        Output of generate_addon_library(). May be empty.
    map_result : DataFrame
        Output of map_library() for the existing library mapped to the target genome.
    records : list of SeqRecord
        Parsed genome records for the target genome.
    existing_coverage : dict
        {locus_tag: n_guides} for coding-strand guides in the existing library.
    n : int
        Target total guides per gene (used as the coverage threshold).
    output_path : str or path-like
        Path to write the HTML report.
    """
    genome_name = records[0].description if records else "Unknown"

    # All genes in genome
    all_gene_info = _all_gene_info(records)

    # Per-gene guide counts for the add-on library
    addon_coverage = {}
    if len(addon_lib) > 0 and "locus_tag" in addon_lib.columns:
        for lt, grp in addon_lib.groupby("locus_tag"):
            addon_coverage[lt] = len(grp)

    n_addon_guides = len(addon_lib)

    # Coverage comparison table
    cov_table = _coverage_table_html(
        existing_coverage, addon_coverage, all_gene_info.keys(), n
    )

    # Plots
    plot_html = ""
    if n_addon_guides > 0:
        score_b64 = _fig_to_base64(plot_score_distribution(addon_lib))
        gpg_b64 = _fig_to_base64(plot_guides_per_gene(addon_lib))
        plot_html = _plot_row(
            [
                ("Score distribution (add-on guides)", score_b64),
                ("Guides per gene (add-on)", gpg_b64),
            ]
        )

    # Genome coverage overlay plot (existing + addon)
    existing_matched = map_result[map_result["n_matches"] > 0] if len(map_result) else map_result
    overlay_html = ""
    if n_addon_guides > 0 or len(existing_matched) > 0:
        overlay_fig = plot_genome_coverage_overlay(
            existing_matched, addon_lib, records
        )
        overlay_b64 = _fig_to_base64(overlay_fig)
        overlay_html = _plot_row(
            [("Genome coverage (existing + add-on)", overlay_b64)]
        )

    # Build combined DataFrame with one row per guide for _gene_coverage_section
    import pandas as pd
    combined_rows = []
    for lt in all_gene_info:
        count = existing_coverage.get(lt, 0) + addon_coverage.get(lt, 0)
        for _ in range(count):
            combined_rows.append({"locus_tag": lt})
    combined_df = pd.DataFrame(combined_rows) if combined_rows else pd.DataFrame({"locus_tag": []})

    glossary_metrics = [
        "score", "score_quartile", "ntargets", "noff_12",
        "noff_11_gene", "noff_9_prom", "inbadseeds",
    ]

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>CRISPRbact Report: Add-on Library - {html_module.escape(genome_name)}</title>
<style>
{_HTML_BASE_CSS}
</style>
</head>
<body>
{_unified_title_html("Add-on Library", genome_name)}

<h2>Genome Summary</h2>
{_genome_summary_html(records)}

<h2>Coverage Summary</h2>
{cov_table}
{_position_summary_html(addon_lib)}

{plot_html}

{overlay_html}

<h2>Off-target Statistics</h2>
{_off_target_stats_html(addon_lib)}

<h2>Bad Seeds</h2>
{_badseed_summary_html(addon_lib)}

<h2>Gene Coverage (combined libraries)</h2>
{_gene_coverage_section(combined_df, records, n=n)}

<h2>Gene Inspection Examples</h2>
<p>Random sample: one gene on each strand (add-on library).</p>
{_gene_inspection_html(addon_lib, records) if n_addon_guides > 0 else "<p>No add-on guides.</p>"}

{_glossary_html(glossary_metrics)}

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def inspect_gene_summary(library, query, score_col="score"):
    """Compact table view of guides targeting a gene.

    Parameters
    ----------
    library : DataFrame
        Output of generate_library().
    query : str
        Gene name or locus_tag.
    score_col : str
        Column name for the score (default ``"score"``).

    Returns
    -------
    str
        Formatted table of guide details.
    """
    gene_guides = library[
        (library["locus_tag"] == query) | (library["gene"] == query)
    ]
    if "gene_rank" in gene_guides.columns:
        gene_guides = gene_guides.sort_values("gene_rank")
    elif score_col in gene_guides.columns:
        gene_guides = gene_guides.sort_values(score_col, ascending=False)

    if len(gene_guides) == 0:
        return f"No guides found for '{query}'."

    lines = []
    # Header
    cols = [
        ("rank", 5), ("guide", 22), ("strand", 7), ("pos", 10),
        ("score", 8), ("quartile", 9), ("ntargets", 9),
        ("noff_12", 8), ("noff_11g", 9), ("inbadseeds", 11),
        ("2nd_half", 9),
    ]
    header = "".join(f"{name:<{width}}" for name, width in cols)
    lines.append(header)
    lines.append("-" * len(header))

    for i, (_, row) in enumerate(gene_guides.iterrows()):
        rank = int(row["gene_rank"]) if "gene_rank" in row.index else i + 1
        vals = [
            (str(rank), 5),
            (row["guide"], 22),
            (row["strand"], 7),
            (str(int(row["pos"])), 10),
            (f"{row[score_col]:.3f}", 8),
            (str(int(row.get("score_quartile", 0))), 9),
            (str(int(row.get("ntargets", 0))), 9),
            (str(int(row.get("noff_12", 0))), 8),
            (str(int(row.get("noff_11_gene", 0))), 9),
            (str(row.get("inbadseeds", "")), 11),
            (str(row.get("second_half_gene", "")), 9),
        ]
        lines.append("".join(f"{v:<{w}}" for v, w in vals))

    return "\n".join(lines)
