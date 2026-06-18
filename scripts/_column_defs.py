LIBRARY_PRIMARY_COLUMNS = [
    ("guide", "Guide RNA sequence"),
    ("locus_tag", "Gene locus tag"),
    ("gene", "Gene name"),
    ("score", "Predicted on-target score"),
    ("score_quartile", "Score quartile"),
    ("ntargets", "Number of off-targets"),
    ("noff_12", "12-mer off-target count"),
    ("noff_11_gene", "11-mer gene off-target count"),
    ("noff_9_prom", "9-mer promoter off-target count"),
    ("inbadseeds", "Bad seed flag"),
]

MAP_COLUMNS = [
    ("guide", "Guide RNA sequence"),
    ("n_matches", "Number of matches"),
    ("on_target_score", "Predicted on-target score"),
    ("score_quartile", "Score quartile"),
    ("ntargets", "Number of off-targets"),
    ("noff_12", "12-mer off-target count"),
    ("noff_11_gene", "11-mer gene off-target count"),
    ("noff_9_prom", "9-mer promoter off-target count"),
    ("inbadseeds", "Bad seed flag"),
]

CORE_COLUMNS = [
    ("family_id", "Core family identifier"),
    ("guide", "Guide RNA sequence"),
    ("coverage", "Genome coverage"),
    ("gps", "Global Penalty Score"),
    ("mean_score", "Mean guide score"),
]