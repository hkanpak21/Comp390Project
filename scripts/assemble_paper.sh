#!/usr/bin/env bash
# Concatenates the discrete section drafts in paper/sections/ into one
# review-friendly paper.md. Each section is preceded by a horizontal rule
# for visual separation. The frontmatter blocks (> Status / Slice / Depends-on)
# inside each section are preserved as a paper-trail of which slices produced
# which section.
set -e
SECTIONS_DIR="paper/sections"
OUT="paper/paper.md"

ORDER=(
  "01_abstract.md"
  "02_introduction.md"
  "03_background.md"
  "04_identifying_nexus.md"
  "05_strategies.md"
  "06_per_op_typology.md"
  "07_end_to_end.md"
  "08_discussion.md"
  "09_conclusion.md"
  "appendix_a.md"
)

{
  echo "# multiNEXUS — Multi-GPU FHE Inference for BERT on H100"
  echo ""
  echo "> Auto-assembled from \`paper/sections/*.md\` on $(date +%Y-%m-%d)."
  echo "> Canonical source is the discrete section files; regenerate with"
  echo "> \`scripts/assemble_paper.sh\`."
  echo ""
  echo "---"
  echo ""
  for s in "${ORDER[@]}"; do
    if [ -f "${SECTIONS_DIR}/${s}" ]; then
      cat "${SECTIONS_DIR}/${s}"
      echo ""
      echo "---"
      echo ""
    else
      echo "<!-- MISSING SECTION: ${s} -->"
    fi
  done
} > "${OUT}"

WORDS=$(wc -w < "${OUT}")
LINES=$(wc -l < "${OUT}")
echo "Wrote ${OUT}: ${WORDS} words, ${LINES} lines."
