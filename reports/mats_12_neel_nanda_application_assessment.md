# Neel Nanda MATS 12.0 application assessment

Research date: 2026-08-11

Primary source: [Neel Nanda MATS 12.0 (Winter 2026–27) application document](https://docs.google.com/document/d/1p-ggQV3vVWIQuCccXEl1fD0thJOgXimlbBpGk6FI32I/edit?tab=t.0#heading=h.y0ohi6l5z9qn)

Supporting program-wide sources: [MATS FAQ](https://www.matsprogram.org/faq) and
[MATS selection guidance](https://www.matsprogram.org/faq/getting-into-mats).

## Bottom line

This is Neel Nanda's mentor stream for MATS 12.0. A circuit-tracing or attribution-graph
submission is a credible fit only when it answers a concrete AI-safety question and demonstrates
pragmatic value beyond simpler baselines. The document explicitly invites work testing whether
attribution graphs reveal useful model-biology insights that prompting, guessing-and-checking,
chain-of-thought inspection, or probes cannot. It also explicitly discourages circuit finding for
its own sake, IOI-style circuit discovery on arbitrary tasks, basic SAE work, toy models, purely
theoretical work, and projects limited to old models such as GPT-2, Pythia, or Gemma 2.

## Dates and selection funnel

- Application: **Friday, September 4, 2026, 11:59 p.m. Pacific Time**. Extensions are available
  through **September 11**. The year is inferred from the Winter 2026–27 title and calendar.
- Exploration offers: **September 15**.
- Paid online exploration: **September 28–October 30**; three weeks part-time, followed by a
  two-week full-time paired research sprint.
- Research-phase decisions: **November 6**.
- Paid research phase: **January 19–April 10, 2027**; 12 weeks, full-time, normally in Berkeley.
- Approximately 34 applicants enter exploration and approximately 8 advance to research, so only
  about 24% of the exploration cohort advances. MATS reports a program-wide ultimate selection
  rate of roughly 4–7%, not specific to this stream.

## Eligibility

- All backgrounds and experience levels are welcome; prior mechanistic-interpretability or
  AI-safety experience is not required.
- Central MATS rules require applicants to be at least 18 by program start. Both US and non-US
  citizens may apply.
- US work authorization is unnecessary. Exploration is remote, MATS can assist with a J-1 visa,
  and remote research-phase participation is possible when needed, though strongly discouraged.
- The final exploration sprint and main research phase require full-time participation. The main
  phase cannot be done alongside a full-time job.

## Required application

1. Complete a mini-project on an interesting AI-safety problem in roughly 16 hours, with an
   absolute maximum of **20 active project hours**. Project-specific reading, planning, coding,
   analysis, and the main report count; prior general learning, generic setup, breaks, and
   unattended training do not.
2. Submit the linked application form. Its concise project-summary answers are the preliminary
   filter and should name the model, experiment, concrete result, importance, and main limitation.
3. Submit a Google Doc accessible to anyone with the link. It must begin with a self-contained
   executive summary of **1–3 pages and at most 600 words**; about one page including graphs is
   preferred. The report should be understandable without reading the code.
4. Include graphs. If results depend on generated data, an LLM judge, or subjective labels,
   manually inspect the data and show randomly selected, non-cherry-picked examples.
5. Code is encouraged but not required.

The detailed rules grant two extra hours for the executive summary and do not count application-
form answers, although an earlier summary loosely describes the extra two hours as covering both.
The safest practice is to track each category separately and disclose the accounting.

Existing relevant work may replace a fresh project if it is a (co-)first-author paper, a paper
with a significant personal contribution, or a high-effort safety blog post. The applicant must
link it, write an executive summary, estimate total hours, explain their contribution, and explain
its relevance if not obviously mechanistic interpretability. Existing work is judged more harshly
because it normally received much more time.

## Evaluation criteria

The document emphasizes:

- clear claims, methods, metrics, graphs, and evidence;
- interesting, original, tractable problem choice aligned with current mentor interests;
- skepticism, controls, alternative explanations, and honest limitations;
- technical depth and practical understanding rather than blindly following an agent;
- simple methods before complex ones;
- strong prioritization, productivity, and willingness to pivot;
- visible research reasoning, especially for negative or inconclusive results; and
- concise, readable communication.

Neel reads form summaries first and cannot read every full report. The most damaging mistakes are
unverified agent output, unsupported or inflated claims, failure to inspect data, missing cheap
controls, generic problem choice, use of an incapable or outdated model, superficial breadth, and
poor writing.

LLMs and coding agents are explicitly encouraged for this work test, but every claim and result
must be checked. Raw LLM-written summaries or executive-summary prose are a strong negative signal,
and fabricated or plainly unverified results are disqualifying. The central MATS application may
have stricter LLM rules, so its form-specific instructions must also be followed.

## Program obligations and benefits

- Exploration is deliberately self-directed. The sprint is normally completed in a self-chosen
  pair and ends with a presentation; advancement is based mainly on sprint output.
- The research phase is full-time, normally paired, with about 1.5 hours of mentor check-in per
  pair each week plus Slack support.
- There is no obligation to conduct research in the roughly 2.5-month gap before the main phase.
- Exploration stipend: **$4,200 for five weeks**.
- Research stipend: **$19,200 for 12 weeks**, with housing support. Central MATS materials also
  list travel, office space, and weekday meals.
- An optional extension may support finishing or publishing the work.

## IP and publication

- Scholars own their work's IP; Neel, MATS, Google, and Google DeepMind do not.
- Scholars are strongly encouraged to publish and open-source under a permissive license.
- Recent scholars commonly produce a co-first-author paper at a top ML venue, but the document
  says publication is a bonus rather than a stated contractual requirement.
- The public document states no confidentiality, publication-review, or IP-assignment restriction.
  Any later participation agreement should be checked before accepting an offer.

## Implications for a circuit-tracing submission

A strong submission should start with a safety-relevant model behavior and use circuit tracing as
a means, not the objective. It should use a current capable model, compare attribution graphs with
cheap black-box and probe baselines, show a downstream benefit such as model forensics, model
diffing, alignment-training analysis, or hypothesis validation, and test load-bearing conclusions
with interventions, resampling, counterfactuals, or careful qualitative inspection. Faithfulness,
replacement-model error, data quality, and cherry-picking should be addressed explicitly.

The fit is therefore **promising but conditional**: attribution graphs are named as an encouraged
research direction, while generic circuit discovery is named as a weak one. Evidence that the
method enables a safety-relevant finding unavailable from simpler techniques is the central
differentiator.
