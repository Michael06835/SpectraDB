Line B — B5 Final Report Archive
================================
Status: complete archive generated from frozen B4 assets.
Classification: exploratory / post-hoc; not a blind test; not confirmatory.

Archive contents:
- line_b_final_report.md : 16-section final report
- tables/                : 19 CSV tables covering 18 models, 3 seeds, 14/15 labels,
                           both F1 policies, paired differences + CIs, error
                           complementarity, OR/AND, Oracle, conditional
                           complementarity, structural subgroups, 88 cases.
- figures/               : 11 real-data PNG figures (Pillow).
- representative_molecules/ : 88 B4.4-supplement cases copied byte-identically
                           (RDKit PNG/SVG, spectra PNG, metadata JSON).
- asset_manifest.json    : input/output SHA256.
- verification.json      : independent verification.

Key verdicts:
H1 PARTIAL SUPPORT; H2 REJECTED; H3 PARTIAL SUPPORT; H4 DESCRIPTIVELY SUPPORTED.

WARN fixes:
1. B4.3 CIs are paired-difference CIs, not absolute-performance CIs.
2. H3 subgroup only_* enrichment is NOT directly tested and is not claimed.
3. AP/AUROC != chemical-information content; QM9S != experimental domain;
   representative molecules are qualitative, not XAI/statistical evidence.

No training, inference, threshold fitting, bootstrap or git operation was run.
