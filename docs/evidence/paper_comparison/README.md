# Paper comparison evidence

`paper_inventory.json` identifies all 12 research PDFs in the repository (213
physical pages total), with SHA256 and key method/result page references. The
assignment brief is not a competing method. Text and page render intermediates
remain in `outputs/paper_audit/`; licensed full text is not copied here.

`refinement_scene_bootstrap.json` reports the paired RAW/C_cuda analysis on the
frozen NYUv2 654-image outputs. It records four input CSV hashes and the script
hash, 215 scene clusters, seed 42 and 10,000 cluster bootstrap draws. Intervals
are exploratory, conditional on this dataset/protocol, and are not a comparison
against paper-reported scores from different protocols.

Reproduce from the project root:

```powershell
python scripts/audit_refinement_statistics.py
```

The full interpretation, per-paper numbers, PDF page/table references and
primary-source links are in [the review](../../paper_comparison.md).

Extended paired video results cover the same four self-recorded clips and two
GT-synchronized TUM excerpts. The 100-image processed iBims audit includes both
positive and negative RAW/C outcomes. Numerical evidence is kept separately in
`../extended_validation_cpu/` and `../extended_validation_gpu518/`; their metric
definitions and input precision must be retained when citing the results.
