# COMP8240 empirical research case study

**Scope:** Reproducible *modern reconstruction* of Fisher-style linear discriminant analysis using the Iris classification dataset and method-level evaluation on Wine. This is **not** a numerical replication of the exact 1936 publication, and no LLM-as-judge experiment has been executed. Do not describe it otherwise.

## Files

- `report.pdf` – report preview (10 pages; uses a **non-official** ACL-style compatibility package).
- `main.tex` – LaTeX report source.
- `acl.sty` – LOCAL PREVIEW-ONLY compatibility style, **not** official ACL.
- `experiment.py` – reproducible comparison, including LDA, logistic regression and majority baseline.
- `results.json` – actual saved metrics and software versions.
- `iris_confusion.pdf`, `wine_confusion.pdf` – confusion matrix figures.

## Reproduction

Create a Python environment and run:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python experiment.py
```

This overwrites results.json and the plots and prints metrics. Repeated stratified fivefold validation with 10 repeats, seed 42. Note that different library versions can introduce small numerical changes.

## Publication / report instructions

1. Create a **public** GitHub repository, e.g., `comp8240-research-project`.
2. Upload these files via `Add file > Upload files > Commit changes`.
3. Open the public repository in a browser while signed out to confirm it is reachable.
4. Insert the real repo URL into the final report as a footnote; DO NOT use an invented placeholder as though it is live.
5. Move `main.tex`, figures and references into the **official ACL template** from https://github.com/acl-org/acl-style-files ; the included `acl.sty` is not official. Compile and recheck main-body page limit.
6. Independently confirm all citations, experimental interpretations and your institution's academic-integrity requirements.

## Important limitations

The published Fisher (1936) paper provides no Python source to rerun. The present protocol evaluates a modern LDA implementation and does not reproduce a historically matched original-paper score. The Wine dataset is new **to this evaluation**, not newly collected. Label comparisons are algorithmic and an LLM judge was not executed. These are material gaps against the assignment rubric, not completed criteria.
