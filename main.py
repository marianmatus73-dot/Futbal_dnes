
Football Report Explainability V15.1 integration prepared.

Purpose:
- replace generic "Reason: Raw edge..." output
- append structured explanation from football_explainability_v15
- show positive and negative signals in reports
- keep existing prediction pipeline unchanged

Integration point:
After candidate decision creation, call:

format_explanation_block(
    load_latest_explanation(
        settings.db_file,
        source_hash,
    )
)

and append result to report text.
