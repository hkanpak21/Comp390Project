PAPER     := main
SVG_FIGS  := $(wildcard fig*.svg)
PDF_FIGS  := $(SVG_FIGS:.svg=.pdf)

# Detect SVG converter (rsvg-convert preferred; fall back to inkscape)
SVG2PDF := $(shell command -v rsvg-convert 2>/dev/null)
ifeq ($(SVG2PDF),)
  SVG2PDF := $(shell command -v inkscape 2>/dev/null)
  SVG2PDF_CMD = $(SVG2PDF) --export-type=pdf --export-filename
else
  SVG2PDF_CMD = $(SVG2PDF) -f pdf -o
endif

LATEX := pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX := bibtex

.PHONY: all clean figures view

all: $(PAPER).pdf

figures: $(PDF_FIGS)

%.pdf: %.svg
ifeq ($(SVG2PDF),)
	$(error No SVG converter found. Install rsvg-convert (brew install librsvg) or inkscape)
endif
	$(SVG2PDF_CMD) $@ $<

$(PAPER).pdf: $(PAPER).tex refs.bib $(PDF_FIGS)
	$(LATEX) $(PAPER).tex
	$(BIBTEX) $(PAPER) || true
	$(LATEX) $(PAPER).tex
	$(LATEX) $(PAPER).tex

clean:
	rm -f $(PAPER).{aux,bbl,blg,log,out,toc,pdf} $(PDF_FIGS)

view: $(PAPER).pdf
	open $(PAPER).pdf
