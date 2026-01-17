#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
TEX_DIR="$BASE_DIR/ICML2025_Template"
MAIN_TEX="$TEX_DIR/example_paper.tex"
OUT_DIR="$BASE_DIR/pdf"
mkdir -p "$OUT_DIR"
SRC_PDF="$TEX_DIR/example_paper.pdf"
if command -v latexmk >/dev/null 2>&1; then
  (cd "$TEX_DIR" && latexmk -pdf -interaction=nonstopmode -file-line-error "$(basename "$MAIN_TEX")")
elif command -v tectonic >/dev/null 2>&1; then
  (cd "$TEX_DIR" && tectonic -X compile "$(basename "$MAIN_TEX")")
elif command -v pdflatex >/dev/null 2>&1; then
  (cd "$TEX_DIR" && pdflatex -interaction=nonstopmode -file-line-error "$(basename "$MAIN_TEX")")
else
  if command -v brew >/dev/null 2>&1; then
    brew list tectonic >/dev/null 2>&1 || brew install tectonic
    (cd "$TEX_DIR" && tectonic -X compile "$(basename "$MAIN_TEX")")
  fi
fi
if [ -f "$SRC_PDF" ]; then
  cp "$SRC_PDF" "$OUT_DIR/"
  echo "$OUT_DIR/$(basename "$SRC_PDF")"
else
  echo "生成的 PDF 未找到，可能编译失败或未安装 LaTeX 编译器" >&2
  exit 1
fi
