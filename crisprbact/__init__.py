from crisprbact.predict import on_target_predict
from crisprbact.off_target import (
    compute_off_target_df,
    extract_features,
    extract_records,
)
from crisprbact.utils import NoRecordsException
from crisprbact.library import generate_library
from crisprbact.core_library import generate_core_library
from crisprbact.map_library import map_library, build_pam_index
from crisprbact.addon_library import generate_addon_library

__all__ = [
    "extract_records",
    "on_target_predict",
    "compute_off_target_df",
    "extract_features",
    "NoRecordsException",
    "generate_library",
    "generate_core_library",
    "map_library",
    "generate_addon_library",
]

try:
    from crisprbact.visualize import (
        generate_report,
        generate_map_report,
        generate_addon_report,
        inspect_gene,
        inspect_gene_summary,
        plot_genome_coverage,
        plot_guides_per_gene,
        plot_off_target_distribution,
        plot_score_distribution,
    )

    __all__ += [
        "generate_report",
        "generate_map_report",
        "generate_addon_report",
        "inspect_gene",
        "inspect_gene_summary",
        "plot_genome_coverage",
        "plot_guides_per_gene",
        "plot_off_target_distribution",
        "plot_score_distribution",
    ]
except ImportError:
    pass
