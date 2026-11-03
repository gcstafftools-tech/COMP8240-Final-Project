# COMP8240 project: Fisher replication, independent evaluation, and LLM audit

Research source: Fisher (1936) https://doi.org/10.1111/j.1469-1809.1936.tb02137.x

Official ACL LaTeX report `main.pdf` (10 total pages) and source `main.tex`.

## Original versus new data
- Iris: classic Fisher source; reconstruct published binary discriminant coefficients.
- Wine: separate UCI dataset, 178 older observations; LDA algorithm re-fit and evaluated.
- `constructed_wine.csv`: labelled simulated perturbation of original Wine samples; **not new physical samples**.
- `independent_wdbc_curated.csv`: a **newly curated file from an independently collected existing UCI dataset**, not previously used in the study (569 observations, 30 features, 2 classes); no new patients recruited. Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic . This is an educational statistical evaluation, **not a diagnostic model**.

## Recreate the evidence

```bash
python -m pip install -r requirements.txt
python historical_replication.py
python experiment.py
python construct_and_evaluate.py
python independent_data.py
python score_judge.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Results in `results.json`, `historical_replication_results.json`, `constructed_data_results.json`, `independent_wdbc_results.json`, `judge_scored_results.json`. The experimental software versions and CV seed are recorded by scripts.

## LLM-as-judge: what is and isn't independently repeatable

`judge_records.json` contains a previously archived **single conversational LLM evaluation** of ten factual claims. `JUDGE_PROMPT.txt` describes the judge instructions; `score_judge.py` reproduces the scoring for the archive. The archived session was not collected through an API and cannot be rerun exactly offline. `rerun_judge_api.py` implements an optional live model API rerun, writing new response records rather than overwriting the archive. **That script has not been run here** and requires the operator's authorised `OPENAI_API_KEY` and `JUDGE_MODEL`. Never commit credentials. Exact LLM outputs are not guaranteed to be deterministic, even with matching prompts.

The ten reference labels were curated during the study and are **not** a blinded independent human annotation sample. The 10/10 agreement in the archive is descriptive, not validated judge accuracy.

## Lecturer approval and submission

The researcher independently selected a historical source paper. **Lecturer approval is not established.** See `LECTURER_APPROVAL_EMAIL.txt` for the request to send through the university's usual channel. The teaching team alone can decide whether this is an acceptable novel project.

The report's original numerical replication does not rerun historical source code, which was not provided in the 1936 paper. This limitation is acknowledged in the report.

To update the public GitHub project, upload extracted files to https://github.com/gcstafftools-tech/COMP8240-Final-Project and commit. Check that all new CSV, JSON and Python scripts appear. Submit `main.pdf` only after checking the above academic requirements and any AI disclosure rules.
