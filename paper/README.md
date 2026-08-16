# Paper

`main.tex` — arXiv-ready LaTeX source. Self-contained: standard `article` class,
inline bibliography, no external `.bib` or style files, so it compiles on arXiv's
TeX Live without extra uploads.

## Build

**Overleaf** (no install): upload `main.tex`, compile with pdfLaTeX.

**Local** (needs MacTeX / TeX Live):
```bash
pdflatex main.tex && pdflatex main.tex     # twice, for references
```

## Submitting to arXiv

Upload `main.tex` alone. arXiv compiles it. Suggested categories:
`cs.CR` (primary), cross-list `cs.LG`.

## Note on claims

Every number in the paper is measured and traceable to `../logs/training_runs.md`
or `../RESULTS.md`. The paper's contribution is diagnostic: the released
checkpoint is not state of the art and is not presented as such. Section 5.3
reports that the smallest model tested outperforms the largest, and Section 5.5
documents that the released checkpoint fails to separate attack from benign
traffic out of distribution.
