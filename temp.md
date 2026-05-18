L_cplx is not well-posed as "average path length"
You're right that the metric doesn't have a clean meaning here. In a CLT attribution graph:

Every feature has edges to features in every later layer it survives into, plus often direct edges to logits.
After pruning, the supernode graph still has many parallel emb→…→logit routes, so the shortest path from a given emb to a given logit is usually just whatever skips the most levels — frequently 2 (one hop into a feature supernode, one hop out to the logit) or even 1 if the edge exists.
"Average shortest-path length over (emb, logit) pairs" thus mostly measures how many emb's have a direct shortcut, not how complex the summary is. It saturates near 1–2 across very different partitions.
There's also the tension you noted with L_cons. Collapsing pushes:

$D_{\text{agg}} \uparrow$ (bad — penalised),
average path length $\downarrow$ (rewarded by L_cplx),
$L_{\text{coh}} \uparrow$ (bad — penalised).
So L_cplx and D_agg are pulling against each other, which sounds right in principle. The problem is just that L_cplx doesn't actually measure "summary complexity" the way we want.

Cleaner alternatives for L_cplx (none involve flow or paths):

Option	Formula	Reads as
(a) Supernode count	$\lvert \pi(\varphi) \rvert$	"how many boxes the reader has to look at"
(b) Edge count	$\lvert E^{SN} \rvert$	"how many arrows the reader has to trace"
(c) Nodes + edges	$\lvert \pi(\varphi) \rvert + \lvert E^{SN} \rvert$	visual element count
(d) DAG depth	longest shortest-path from $V_{\text{emb}}$ to $V_{\text{logit}}$	"how many levels deep" — but same saturation problem
I'd push for (a). It's what every graph-summarization paper uses, it's monotone in "how much there is to read", and it's a clean counterpart to the granularity hyperparameter $K$ each solver already picks.

Ranges are not commensurate
Cosine distance for L_coh helps partially:

Pro: scale-invariant — no dependence on $\lVert W \rVert$.
Pro: each term in $[0, 1]$ (with rectified cosines), so the sum over members lives in $[0, |V'_{\text{mid}}|]$.
Con: still a sum, not a mean, so it grows with cluster size and isn't directly comparable to D_agg ∈ [0, 1].
Con: a normalised version ($\frac{1}{|V'_{\text{mid}}|}\sum (1-\cos)$ ∈ [0,1]) fixes the scale but doesn't fix the deeper problem.
The deeper problem is this: even if every $L_\ast$ lives in [0, 1], adding them assumes a 1-unit-of-coherence-badness ≡ 1-unit-of-conservation-badness ≡ 1-unit-of-complexity-badness exchange rate, which is arbitrary. You can normalise the scales but you can't normalise the semantics.

Two honest responses:

(A) Drop the sum; make L a vector. State the problem as multi-objective:
$$\min_{\varphi};(L_{\text{coh}}(\varphi),;L_{\text{cons}}(\varphi),;L_{\text{cplx}}(\varphi))$$
with Pareto dominance as the comparison rule. This is what the evaluation methodology already does in practice (report each, dominate = Pareto-better). Solvers minimise their own scalarisation; the problem doesn't need to.

(B) Keep the sum but add explicit weights with a stated calibration. $L = \lambda_1 L_{\text{coh}} + \lambda_2 L_{\text{cons}} + \lambda_3 L_{\text{cplx}}$, with $\lambda_i$ chosen so that "one $\sigma$ of badness on each axis" trades off equally on a calibration set. This is more honest than just summing but introduces three knobs and a calibration story.

(A) is simpler and matches what the eval already does. Cosine on L_coh is then a nice-to-have for scale invariance rather than a load-bearing fix.

What I'd suggest
Change L_cplx to $\lvert \pi(\varphi) \rvert$ (supernode count). Drop the path-length framing entirely.
Switch L_coh to mean rectified cosine distance for scale invariance — but flag in the text that this is normalisation, not a fix for the cross-axis exchange rate.
Reframe the objective as vector-valued / multi-objective, with Pareto dominance as the evaluation rule. Keep $L = L_{\text{coh}} + L_{\text{cons}} + L_{\text{cplx}}$ as a notational shorthand if useful, but don't depend on the sum being meaningful.
Want me to proceed with all three, or push back on any of them? In particular:

Are you attached to keeping L_cplx path-flavored (e.g. depth instead of count)?
Do you want the formulation to stay scalar-objective ($L = \sum L_\ast$) for the paper's narrative arc, or is multi-objective acceptable?