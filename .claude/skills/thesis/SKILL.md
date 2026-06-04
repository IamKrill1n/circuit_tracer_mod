---
name: thesis-writing
description: Improve graduation-thesis (SOICT/HUST DATN) writing quality for a research-track thesis with the chapter structure Introduction, Literature Review, Methodology, Theoretical Analysis, Numerical Results, Conclusions. Use when drafting or revising the Abstract or any chapter, writing per-chapter Overview/Summary sections, checking paragraph flow and scientific tone, aligning claims with evidence, or doing a final self-review before submission.
---
# Thesis Writing (SOICT DATN, research track)

## Overview

Use this skill to draft and revise a graduation thesis that follows the
`SOICT_DATN_Research_ENG_Template/` template. Target a defense-committee reader:
prioritize a clear thesis story, the template's mandated chapter structure and claims that
are backed by results.

The template is the source of truth for structure and formatting. Read the
relevant chapter file under `SOICT_DATN_Research_ENG_Template/Chapter/` before
editing that chapter, and keep the existing LaTeX section scaffolding.

## Thesis Structure

The thesis is a `report`-class document with these front-matter and chapter
files (see `SOICT_DATN_Research_ENG_Template/main.tex`):

- `Chapter/0_2_Acknowledgment.tex` — acknowledgment.
- `Chapter/0_3_Abstract.tex` — Abstract, 200–350 words, prose only.
- `Chapter/1_Introduction.tex` — Chapter 1, 3–6 pages.
- `Chapter/2_Literature_review.tex` — Chapter 2, ≤10 pages.
- `Chapter/3_Methodology.tex` — Chapter 3.
- `Chapter/4_Theoretical_Analysis.tex` — Chapter 4 (optional; omit if the work has no theory).
- `Chapter/5_Numerical_results.tex` — Chapter 5 (optional; omit if the work has no experiments).
- `Chapter/6_Conclusions.tex` — Chapter 6.
- `Chapter/7_Reference.tex` — short notices on references.

## Global Writing Rules (from the template)

These rules come directly from the template instructions and override generic
paper-writing habits.

1. One paragraph carries exactly one main idea, stated up front, then supporting
   analysis that refines it. Do not let a paragraph run long or mix ideas.
2. Every sentence has a full subject and predicate and serves the paragraph's
   shared topic. Each sentence links to the previous one; each paragraph links to
   the previous one.
3. Use objective scientific register. Never use spoken-language words, hyperbole,
   or subjective/emotional praise ("amazing", "extremely useful", "very cool").
4. Tighten every sentence until it is hard to add or remove even one word. Be
   concise; avoid padding.
5. Keep terminology stable across the whole thesis; define a term before reusing
   it.
6. If a claim cannot be supported by results, weaken or remove it.
7. Treat figures and tables as core content: clean teaser/pipeline figures,
   readable minimal-ink tables, consistent formatting.

## Per-Chapter Overview and Summary (mandatory)

Every main chapter (2–6) opens with an **Overview** paragraph and closes with a
**Chapter Summary** paragraph, both in normal body text (no bold, no boxes, no
bullets).

- **Overview** of chapter N: connect back to chapter N−1, justify why chapter N
  exists and why it is needed, then introduce what this chapter presents and
  under which top-level sections. Chapter 1 does not need an Overview.
- **Chapter Summary**: state the chapter's key conclusions, recap how the
  problems opened in the Overview were resolved, and add a linking sentence to
  the next chapter. The Summary must not duplicate the Overview verbatim.

## Section Guides

### Abstract (`0_3_Abstract.tex`)
200–350 words, continuous prose, never bullets. Cover, in order: (i) the problem
and why it matters, what current approaches exist and their limitations; (ii) the
chosen approach and why; (iii) an overview of the proposed solution; (iv) the
main contributions and final results.

### Chapter 1 — Introduction (3–6 pages)
Follow the template sections:

- **Problem Statement**: describe the problem, why it was chosen, and its
  importance. The section title may instead name the concrete problem.
- **Background and Problems of Research**: survey current results for the problem,
  then expose the limitations of existing solutions.
- **Research Objectives and Conceptual Framework**: state the thesis objectives,
  then propose the solution direction, ideally one response per limitation raised
  above.
- **Contributions**: list the concrete, concise contributions of the thesis.
- **Organization of Thesis**: describe the remaining chapters as full prose
  paragraphs — absolutely no bullet points or sentence fragments. For each
  chapter give a sentence or two on its content; chapter 1 is not described here.

### Chapter 2 — Literature Review (≤10 pages)
- **Scope of Research**: bound what the thesis covers.
- **Related Works**: present related work and analyze its strengths and
  weaknesses, then derive the motivation for this thesis from those gaps.
- **Background knowledge sections**: include only foundations tightly coupled to
  the thesis; do not pad with general textbook material.

### Chapter 3 — Methodology
Present the proposed method. For each component include motivation, design, and
its technical advantage. State non-obvious invariants (shapes, conventions,
thresholds) explicitly, consistent with the codebase conventions.

### Chapter 4 — Theoretical Analysis (optional)
Include only if the thesis has theoretical results (complexity analysis,
performance-ratio proofs, etc.). Omit the chapter otherwise.

### Chapter 5 — Numerical Results (optional)
- **Evaluation Methodology**: define the metrics and parameters used.
- **Simulation/Experiment Method**: baselines and why they were chosen, number of
  experiments and repetitions, parameter selection, scenarios, data handling.
- **Experiment results**: one result per section, each with tables/figures,
  detailed commentary, method-to-method comparison, and explanation of why the
  results came out as they did.

### Chapter 6 — Conclusions
- **Discussion and Limitations**: honest discussion of what the results mean and
  where the method falls short.
- **Conclusion**: summarize contributions and outcomes; note future directions.

## Paragraph Clarity Check

Use this whenever asked whether a paragraph "flows" or is clear.

1. Read as an external reader:
   - Does the paragraph have one explicit main idea?
   - Does the first sentence state it?
   - Are all key terms readable without hidden context?
   - Does each sentence connect to the previous one (cause, contrast, consequence,
     refinement, example)?
2. Reverse-outline the section: write the thesis/main claim, each paragraph's
   topic sentence, and the evidence under each. Check that topic sentences map to
   the chapter goal and evidence maps to its topic sentence. Revise or remove any
   paragraph that does not map cleanly.
3. If flow is still weak, add temporary transition phrases during revision, then
   remove scaffolding before finalizing.

## Final Self-Review

Before finalizing, append and answer a self-review across five dimensions, then
revise based on unresolved items:

1. Contribution — are the contributions concrete and clearly stated?
2. Writing clarity — one idea per paragraph, scientific tone, tight sentences?
3. Experimental strength — are claims backed by Chapter 5 results?
4. Evaluation completeness — metrics, baselines, repetitions adequate?
5. Method soundness — motivation, design, and advantage stated per component?

Treat claim–evidence alignment as a hard constraint, especially for the Abstract
and Chapter 1. Review as a skeptical committee member and resolve every
high-risk question.

## Execution Rules

1. Read the target chapter file and keep its LaTeX scaffolding; do not invent a
   new structure without reason.
2. Build a mini-outline before drafting prose.
3. Edit one chapter at a time; do not load all chapter files at once.
4. Keep terminology stable and tone scientific across the whole thesis.
5. Ensure every chapter 2–6 has an Overview and a Chapter Summary.
6. Write the Organization-of-Thesis and Abstract as prose, never bullets.

## Output Contract

When asked to rewrite or draft a section, return:

1. A compact section outline (3–7 bullets).
2. Revised paragraphs with explicit paragraph roles (overview / problem / design /
   advantage / evidence / limitation / summary).
3. A short self-review checklist covering clarity, flow, terminology consistency,
   unsupported claims, and missing evidence.
4. A claim–evidence map for each major claim using
   `Claim: ... | Evidence: ... | Status: supported/needs evidence`.
