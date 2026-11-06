# A3 report

`a3.tex` is the whole report (same format as our A1). References are a numbered `thebibliography` list at the end. Add new entries there and cite them with `\cite{key}`.

Build (or upload the folder to Overleaf):

```
latexmk -pdf a3.tex
```

Figures live in `figures/`. Each diagram has an `.svg` source and the `.pdf` that LaTeX includes. After editing an SVG, regenerate its PDF, e.g. open it in Chrome/Edge and print to PDF with no margins, or use `inkscape file.svg --export-filename=file.pdf`.
