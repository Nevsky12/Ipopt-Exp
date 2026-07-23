# Experimental C++23 API

This directory is an opt-in, standalone prototype for the next C++ ABI. It
does not alter the stable C or TNLP interfaces.

The first vertical slice is an AnyAny-erased NLP model with:

- `std::span` inputs and outputs;
- `std::expected` callback failures;
- optional native Jacobian-vector and transposed-Jacobian-vector products,
  plus an optional fused pair callback;
- an optional native Hessian-of-the-Lagrangian product;
- cached materialized sparse fallbacks for every missing product direction;
  the paired fallback evaluates Jacobian values once and visits each sparse
  entry once for both products;
- alias-safe transactional products and validation before entering user
  callbacks;
- a two-word fingerprint over dimensions, both sparsity patterns, derivative
  capabilities, and a model revision;
- a reusable operator for `[H + D_x, J^T; J, -D_y]` with strong output
  guarantees when a callback fails;
- an index-map adapter for Ipopt's complete eight-block
  `x/s/y_c/y_d/z_L/z_U/v_L/v_U` primal-dual product, including bound
  complementarity and all four regularization terms;
- a compatibility adapter that composes TNLP fixed-variable expansion,
  equality/inequality ordering, equality right-hand-side subtraction, and
  `OrigIpoptNLP` objective/row/variable scaling for values, `J*v`, `J^T*v`,
  and `H*v`;
- an owning bridge from the initialized stable `TNLPAdapter`/`OrigIpoptNLP`
  stack that snapshots those maps and scales, wraps the same underlying TNLP,
  converts C/Fortran sparsity, and propagates legacy callback failures;
- a non-owning AnyAny problem reference for synchronous consumers, preserving
  the owner's validated derivative-structure and fingerprint caches without
  copying or exposing a stable-ABI object;
- an owning snapshot of arbitrary stable `W/J_c/J_d` matrices through public
  triplet extraction, used for compound formulations such as restoration
  without allowing a stable matrix view to escape;
- a movable, non-copyable solve session that gates Krylov-workspace reuse on
  the complete KKT structure fingerprint while accepting fresh numeric model
  state on every solve;
- a move-only, type-erased direct-preconditioner contract that validates the
  KKT structure and caller-owned numeric revision, factorizes exactly once,
  and exposes only `solve_rhs` applications to FGMRES;
- an opt-in adapter from the stable four-block `AugSystemSolver` to that
  prepared backend, including exact RHS condensation and reconstruction of
  all four bound-complementarity direction blocks;
- a move-only AnyAny `CandidateFirstBackend` contract whose implementation
  owns perturbation retries, exact-inertia certification, refinement, and
  quality escalation, and reports factorization/backsolve work;
- a lazy typed-backend factory that constructs only after the live KKT exists,
  reuses a backend for the same complete structure/restoration role, and
  changes configuration transactionally;
- an allocation-free-after-construction symmetric variable-block
  tridiagonal solver with factor-once/solve-many lifecycle, fused contiguous
  multi-RHS passes, true-residual refinement, certified inertia, and an
  opt-in faster numeric-pivot mode that never claims an inertia proof;
- a symmetric arrowhead solver for `[A B; B^T C]` that factors the stage chain
  once, solves all border columns in one fused traversal, forms the small dense
  Schur complement, and refines against the original arrowhead matrix;
- a compile-time stage-assembler contract and independent candidate-first
  backend that compose exact inertia across congruent condensation,
  reconstruct the complete direction, and own regularization retries without
  calling the stable solver;
- move-only topology-sized workspaces for preparing the equality permutation,
  packed block storage, solver storage, and full-direction storage before a
  live KKT is bound;
- an immutable validated OCP-stage topology, canonical-to-generic primal and
  constraint permutations, and a compile-time derivative provider with owned
  packed Hessian/dynamics/path storage;
- an equality-only full-KKT stage assembler that handles arbitrary generic NLP
  orderings in either stage direction, caches derivatives by numeric revision,
  proves exact full-KKT inertia for its checked DTOC-like PSD-Hessian/full-row-
  rank family, and rejects bounds, slacks, inequalities, restoration
  auxiliaries, and nonidentity next-state dynamics explicitly;
- a full primal-dual stage assembler that accepts explicit variable,
  equality/inequality, and bound maps, eliminates the four complementarity
  direction blocks into the symmetric `[x,s,y_c,y_d]` system, reconstructs the
  complete eight-block direction exactly, and keeps the inertia target on the
  reduced symmetric system;
- a strict equality-stage derivative scatter that consumes each live KKT's
  current Hessian/Jacobian values in linear nonzero work, rejects cross-stage
  structure and noncanonical next-state derivatives, and never retains the
  short-lived matrix snapshot;
- an explicit bordered topology and sparse derivative scatter for local-stage
  variables plus global primals, adjacent-stage Hessian blocks, dense
  next-state dynamics Jacobians, path inequalities, and arbitrary bound maps;
- an explicit `RestoIpoptNLP` topology transform for `[x,n_c,p_c,n_d,p_d]`
  that validates compound variable, constraint-role, and bound maps and keeps
  each restoration pair local to its path or outgoing-dynamics row;
- an explicit `AlgorithmBuilder` runtime gate that supports reference-first
  shadow/replacement checks and an opt-in candidate-first transaction with the
  complete stable `PDSystemSolver` as fallback;
- restarted right-preconditioned FGMRES with a changing preconditioner,
  constructor-owned workspace, true-residual checks, and explicit convergence,
  limit, breakdown, nonfinite, and invalid-configuration statuses.

The adapters and KKT operator deliberately own their sparse values and product
workspaces. After construction and the first source fallback evaluation,
repeated products do not resize those workspaces. A solve session must own one
adapter/operator and must not invoke it concurrently. The full operator wraps
the reusable NLP saddle-point block and matches
`PDFullSpaceSolver::ComputeResiduals` without the right-hand-side subtraction.

`MakeLegacyCoordinateProblem` takes the same expanded-position maps represented
by `TNLPAdapter`'s `ExpansionMatrix` objects. It exposes scaled internal
coordinates with `x_s = D_x x`, `c_s = D_c(c-rhs)`, `d_s = D_d d`, and
`f_s = d_f f`. Consequently its products are
`J_s v = D_g J D_x^-1 v`, `J_s^T w = D_x^-1 J^T D_g w`, and
`H_s v = D_x^-1 H(d_f obj_factor, D_g y) D_x^-1 v`. Fixed variables removed
with `MAKE_PARAMETER` receive zero directions; `MAKE_CONSTRAINT` rows are
appended as scaled identity rows. The adapter validates every map and rejects
zero/nonfinite scaling before entering a model callback.

`MakeLegacyIpoptCoordinateProblem(adapter, orig_nlp, revision)` is the live
bridge. It verifies that `orig_nlp` is backed by exactly that adapter, copies
all legacy maps/RHS/scales after `InitializeStructures`, and owns the same TNLP
through `SmartPtr`. No stable `Vector`, `Matrix`, or borrowed map escapes into
the C++23 object. Legacy callbacks always receive `new_x=true` and Hessian
callbacks receive `new_lambda=true`; callback failure leaves the public output
unchanged. The caller-supplied revision must change when the TNLP structure
changes across reoptimization.

The direct canary path prepares that owning coordinate bridge once per
`PDSystemSolver::InitializeImpl` and gives each short-lived KKT operator a
`BorrowNlpProblem` forwarding value. This keeps the TNLP sparsity conversion,
coordinate scatter, and nested fingerprints cached across Newton systems while
numeric callbacks still observe the current `x` and multipliers. Reinitialize,
a changed adapter/original-NLP identity, and every restoration matrix snapshot
remain explicit cache boundaries.

`PrimalDualSolveSession` can be reused with a different operator instance only
when its dimensions, ordered sparsity, derivative capabilities, structural
revision, and all layout index maps match. Numeric values are never cached by
the session. A mismatch is rejected before a Krylov step and leaves the
caller's solution unchanged.

`PreparedDirectPreconditioner` separates matrix preparation from RHS solves.
Its backend reports a dimension, structural fingerprint, and nonzero numeric
revision. `PrepareDirectPreconditioner` validates all three, validates the KKT
state dimensions, and calls `factorize()` once. Later applications call only
`solve_rhs()` into constructor-owned scratch storage and commit a finite result
atomically. The caller must increment `PrimalDualState::numeric_revision`
whenever any numeric coefficient of the KKT system or any dual/slack value
reachable through the state view changes; it is deliberately separate from
the structural revision. Reusing a prepared factor with a
different numeric revision is rejected before entering either FGMRES or the
direct backend.

`PrepareLegacyAugSystemPreconditioner` binds one initialized stable
`AugSystemSolver` to one numeric KKT revision. It derives the existing
`sigma_x/sigma_s` diagonals from the eight-block state and can bind the live
`IpCq` diagonal vectors so their existing tags are preserved. It performs one
zero-RHS `MultiSolve` to prepare the factor, and then reuses the same stable
matrices, vectors, and tags for every RHS. The reduction and reconstruction
are the formulas used by `PDFullSpaceSolver::SolveOnce`. A changed
`W/J_c/J_d/sigma_x/sigma_s` tag,
nonpositive complementarity slack, wrong inertia, stable exception, or
nonfinite result is rejected without committing the caller output. The
adapter intentionally does not own perturbation retries or `IncreaseQuality`;
those remain policy of `PDFullSpaceSolver`, and a quality change requires a
new numeric revision and prepared backend. The bound stable solver must not be
used concurrently during a prepared session.

`LegacyAlgorithmCanaryBuilder` is the first runtime integration seam. It is
passed explicitly to `IpoptApplication::OptimizeNLP`. `AlgorithmBuilder` now
routes restoration construction through the existing virtual factory with a
`resto.` prefix. Direct exact-Hessian `OrigIpoptNLP`/`TNLPAdapter` solves use
the callback bridge; wrapped restoration formulations use an owning snapshot
of their materialized compound matrices. Inexact, limited-memory, nonzero-beta,
and refinement-only calls are counted as skipped and delegated to the stable
solver. Precision floors are derived from the linked stable `Ipopt::Number`,
so the same gate supports double- and single-precision Ipopt while the C++23
operator remains double precision.

`LegacyAlgorithmCanaryMode::shadow` remains the default. The opt-in
`validated_replacement` mode converts all eight candidate blocks into a new,
detached stable `IteratesVector` only after convergence, finite-direction,
true-residual, and direction-equivalence checks have passed. A conversion,
callback, solve, or equivalence failure leaves the already computed reference
direction untouched; the final `result.Copy` is the sole commit. Replacement
requests, commits, failures, and restoration commits are counted separately.
This mode validates the replacement boundary, but cannot accelerate Ipopt yet:
the complete reference solve still runs first to preserve perturbation,
inertia, refinement, and `IncreaseQuality` policy.

`LegacyAlgorithmCanaryMode::candidate_first` is the first mode that can avoid
the stable factorization. It requires a shared `AnyCandidateFirstBackend` made
with `MakeCandidateFirstBackend`. The backend synchronously receives immutable
views of the current full KKT state and RHS; it must not retain them. It owns
its complete perturbation/inertia/retry/refinement/quality policy and returns
an unscaled eight-block direction, its accepted nonnegative regularization, an
exact negative-eigenvalue certificate, and work counters. The wrapper rejects
a missing/wrong certificate, invalid dimensions, nonfinite values, or a true
residual above the configured limit. Only then does it convert all blocks in a
detached `IteratesVector`, perform the sole `result.Copy`, and publish the four
accepted `PDPert` values. Any backend exception, reported failure, inertia or
residual rejection, or commit failure invokes the complete stable solver
exactly once. Ineligible calls go directly to stable code.

`MakeLazyCandidateFirstBackend(factory)` closes the lifecycle gap between that
stable gate and a topology-specific typed backend. The factory receives the
live full-KKT operator and restoration flag only when a candidate request is
made. Its backend is reused while both the complete KKT fingerprint and the
restoration role are unchanged. A failed reconfiguration does not destroy the
previous valid backend, and the usual candidate-first path still owns the
exactly-once stable fallback.

`SymmetricBlockTridiagonalSolver` and
`StageStructuredCandidateBackend<Assembler>` are a real reference-free
structured kernel and policy layer, rather than the dense canary test double.
The packed topology is fixed at construction; factorization owns its numeric
copy, certifies each Schur-block inertia from explicit eigensystem backward
and orthogonality errors, and reuses factors for scalar or fused RHS passes. A
successful assembler must account exactly for every condensed sign and
reconstruct the complete flat KKT direction. `StageNlpTopology` and
`PreparedStageDerivativeProvider` expose validated `[u_k,x_k]`, dynamics, and
path blocks without type erasure inside stage loops.
`EqualityStageKktAssembler` now maps an equality-only OCP into the complete
flat KKT, including arbitrary primal/equality permutations and incoming
dynamics multipliers, without condensation or stable-solver calls. Its
optional numeric-pivot factor is deliberately marked inexact. The stage
backend may promote it only when the assembler independently proves the exact
inertia of the complete KKT; missing, inexact, zero-separated, or
wrong-dimension certificates are rejected. The current theorem covers an SPD
regularized KKT and an unregularized DTOC-like diagonal-PSD Hessian whose
initial path rows pivot on every Hessian-null direction and whose dynamics
contain the checked canonical next-state `-I` chain. True structured residual,
reconstruction, and the outer full-KKT residual remain mandatory.

`PrimalDualStageKktLayoutWorkspace` and
`PrimalDualStageKktAssembler` extend that path to Ipopt's full
`x/s/y_c/y_d/z_L/z_U/v_L/v_U` direction. The topology and expected
`PrimalDualLayout` are explicit production metadata; binding never infers
roles from variable names or sparsity. The assembler supports arbitrary
generic primal and constraint order, path equalities and inequalities, both
sides of primal and slack bounds, and either stage direction. It uses the
same complementarity RHS condensation and direction reconstruction as
`PDFullSpaceSolver::SolveOnce`. The block factorization has dimension
`x+s+y_c+y_d`, while the returned direction retains all eight blocks. An
independent strict-diagonal-dominance theorem may certify the reduced inertia
when the primal diagonal is positive and the corresponding dual
regularizations are positive; an inexact numeric factor alone is never an
inertia certificate. Layout fingerprints, map coverage, derivative revisions,
positive complementarity slacks, structured residual, reconstruction, and the
outer nonsymmetric full-KKT residual are all checked before commit.

`BorderedStageNlpTopology`, `SparseBorderedStageDerivativeProvider`, and
`PrimalDualBorderedStageKktAssembler` cover the corresponding arrowhead case.
Every global primal is declared explicitly and forms the dense border; local
Hessian entries may stay within one stage or connect adjacent stages. Dense
`D_k = partial g_k / partial x_(k+1)` blocks can either remain natively inside
the next stage or be transformed by the exact congruence
`L_k = -D_k^-1`. The normalized mode transforms the dynamics RHS, global and
current-stage Jacobians, dual regularization, and reconstructed multipliers
consistently; its inverse has pivot and backward-error gates. The native mode
avoids that scaling when the generic block solver does not need canonical
`-I` dynamics.

`SymmetricBorderedBlockTridiagonalSolver` never promotes the floating dense
Schur complement to an exact inertia claim. Its candidate backend therefore
requires a separate full reduced-KKT certificate, plus successful numeric
factorization, structured refinement, reconstruction, and the outer full-KKT
residual. Proof failures are rejected before an expensive factorization. An
opt-in regularization warm start reuses only a previously fully accepted
perturbation; every new numeric matrix still passes all gates.

Both structured candidate backends expose an adaptive `K_delta / M = K_delta`
FGMRES correction after the outer full-KKT residual gate. The accepted direct
step is the initial guess, and `M^-1` is a RHS-only condensation,
`solve_rhs()` with the current stage/arrowhead factors, and full direction
reconstruction. No matrix assembly, perturbation retry, quality improvement,
or factorization is reachable from the Krylov preconditioner. All FGMRES
storage is topology-sized and constructor-owned. Accurate direct steps take
the existing one-apply validation path; only a failed residual requests the
extra Krylov work, and the canary rejects any refinement reporting a new
factorization.

The request role is part of the assembler metadata: a main-problem assembler
is still rejected for restoration and a restoration assembler is rejected for
the main NLP. `MakeRestorationStageNlpTopology` instead builds a separate
topology for the exact `RestoIpoptNLP` compound form. It validates explicit
maps for the original primal and all `n_c/p_c/n_d/p_d` variables, requires the
original bounds plus a lower bound for every restoration variable, and maps
constraint roles through both primal-dual layouts. Restoration variables for
a dynamics row are placed in the previous stage, so their `+I/-I` entries are
ordinary outgoing-dynamics coefficients and the only next-stage derivative
remains canonical `-I`.

`SparseStageDerivativeProvider` prepares scatter maps from one exemplar KKT
but evaluates values through the KKT supplied with each solve request. This is
important for restoration: the stable bridge materializes a new owning
`W/J_c/J_d` snapshot each iteration, so retaining the factory's KKT reference
would dangle. Structure fingerprints, same-stage Hessian/path entries, and the
numeric next-state `-I` are revalidated. The live forced-restoration canary now
accepts this separate structured backend with zero candidate fallback. A
noncanonical scaled dynamics form or any unmodeled cross-stage Hessian remains
an explicit rejection rather than an inferred approximation.

The CUTEst research executable supplies exact DTOC3 and CAR2 blueprints and
binds them through the lazy candidate-first factory. DTOC3 remains the narrow
equality-only canonical-dynamics gate. CAR2 is the representative nonlinear
inequality OCP: 1000 stages, four states, two controls, one global time
parameter, 3996 dynamics equalities, 1000 path inequalities, eight fixed
endpoint states, adjacent-stage Hessian entries, and dense `4x4` implicit
next-state blocks. Its live 5999-variable solve accepts every one of 171
candidate requests with zero fallback and a full-KKT residual gate. After one
warmup, seven interleaved CPU-4-pinned Release samples give a 1580.238 ms
median versus 1937.905 ms for the coordinate-matched MUMPS baseline: the
current candidate is 18.46% faster while taking 171 rather than 236 nonlinear
iterations. A second seven-pair run alternated matched `x86-64-v3` binaries
against the immediately preceding candidate. It gives 1632.929 versus
1636.303 ms (`-0.21%`) end to end and 620.574 versus 628.900 ms (`-1.32%`)
inside the backend for the preceding numeric-factor milestone. The
solve-layout candidate gives 1623.054 versus 1626.942 ms (`-0.24%`) end to
end and 616.363 versus 621.074 ms (`-0.76%`) inside the backend. The subsequent
single-pass-validation candidate gives 1586.014 versus 1636.395 ms (`-3.08%`)
end to end and 577.131 versus 620.433 ms (`-6.98%`) inside the backend; both
were faster in all seven pairs. The edge-once-inertia milestone gives
1579.065 versus 1585.204 ms (`-0.39%`) end to end, 569.456 versus 575.126 ms
(`-0.99%`) inside the backend, and 206.448 versus 209.093 ms (`-1.27%`) in
assembly. End-to-end time improves in six of seven pairs, while backend and
assembly improve in all seven. Reconstruction moves from 6.791 to 6.976 ms
(`+2.72%`, a 0.185 ms median penalty), which is outweighed by the backend
gain.

The current matrix-application kernel exploits the exact symmetry projection
performed while factorizing each diagonal block: one off-diagonal edge now
updates both output rows, with fixed 12x12 and 14x14 dispatch. Bordered
refinement calls an internal nonalias path and avoids the public transactional
workspace copies; the public `apply` methods remain alias-safe and leave caller
output unchanged on failure. In seven order-alternated microbenchmark pairs,
all twenty numeric and pivot-heavy size-12/14 case medians improve by
4.16%--13.11%; 139 of 140 individual pairs are faster and every checksum
matches. The matched CAR2 wall-clock series is deliberately not reported as a
speedup: its median is 1578.408 versus 1573.817 ms (`+0.29%`) and it wins only
three of seven pairs, although backend and assembly medians improve by 0.12%
and 0.46%. The deterministic profile below establishes the isolated kernel
gain. The current protocol median is 25.61% below the historical initial
2124.192 ms candidate. The record is
`graphify-out/car2-bordered-stage-candidate-benchmark.json`. The
separate forced-restoration TNLP still covers exact main/restoration topology
transformation. Public TNLP stage metadata and a bordered restoration
transform remain absent; benchmark name parsing is not a production inference
mechanism.

Normalized bordered assembly now exploits the same prepared-zero contract in
two additional structural blocks. The dual regularization congruence forms
only one triangle of `T T^T` and mirrors it exactly; canonical next-state
coupling writes only the diagonal pairs of `-I`. A C++23 fixed-extent dispatch
unrolls the profile-dominant four-state Gram and current-stage dynamics
products. The same checked partial-pivot Gauss--Jordan inverse is now
instantiated at compile-time for four-state dynamics; arbitrary state counts
retain the generic loops. Four sequential seven-pair native comparisons
isolate those changes. Triangular
Gram formation reduces assembly by 0.59% but remains end-to-end noise-scale;
the sparse `-I` step reduces assembly by 0.78% and total time by 0.32%; the
fixed-four-state step reduces assembly by 1.25% in all seven pairs, backend by
0.33%, and total time by 0.26% in six pairs. The checked-inverse step then
reduces assembly from 208.298 to 204.508 ms (`-1.82%`) and backend from
556.930 to 550.542 ms (`-1.15%`) in all seven pairs; total time moves from
1552.367 to 1546.659 ms (`-0.37%`) with five paired wins. All runs accept
171/171 requests with zero fallback, unchanged objective, and a final
constraint violation of `1.62e-13`. Release and ASan/UBSan both pass all 15
tests; the assembler tests also cover a real four-state row pivot, `T D = -I`,
the exact mirrored Gram and canonical coupling, singular rejection, persistent
structural zeros, and the generic two-state path.

The four-state Gram and normalized-dynamics helpers now also emit their fixed
output coordinates directly instead of retaining runtime row/state loops
around fixed dot products. Deterministic profiling accepts this step, but its
native timing series is excluded: total CV was 2.25%--2.36%, assembly CV was
2.92%--3.24%, and assembly improved in only three of seven pairs. It therefore
makes no new wall-clock claim.

For fixed-size numeric factors, Gauss--Jordan now keeps the augmented identity
in pivot-column order after the first real row pivot. Its exact support is
therefore the contiguous processed prefix; scratch storage and the final
column restoration are activated lazily only when a permutation occurs. The
latest kernel also omits dead normalized-pivot and eliminated-column stores,
and fuses pivot-column restoration with the symmetry projection. A previously
missing fixed 14x14 dispatch now covers both multiplier and Schur formation.
Partial pivot selection, the numeric pivot margin, finite-output checks, the
independent exact full-KKT inertia proof, and both residual gates are unchanged.
At that milestone, seven-sample CPU-4 microbenchmarks across block sizes 12/14
and 16--256 stages showed a further 0.49%--2.28% reduction for 12x12 blocks
and 21.42%--22.24% for 14x14 blocks versus the preceding symmetric-Schur
milestone. Pivot-heavy
12-to-14-to-12 and 14-to-14-to-14 regressions cover the fused restoration and
the new dispatch. The earlier pivot-column change reduced factorization
medians by 16.19%--18.47% against its own predecessor.

The benchmark now also has a dedicated `numeric_pivoted` mode for block sizes
12 and 14 over 16--256 stages. It exposed a proposed lazy augmented-identity
initialization as slower: ordinary size-12/14 cases regressed by up to
3.58%/5.97%, while pivot-heavy size-14 cases regressed by 2.35%--4.34%. That
solver change was fully reverted; the benchmark and a
diagonal-to-pivot-heavy-to-diagonal factor/solve regression remain as guards
against both performance loss and stale inverse state across pivot-pattern
changes.

The same guard rejected splitting the Gauss--Jordan elimination rows into two
ranges around the pivot row. Every one of the twenty size/stage medians
regressed: 2.75%--5.45% and 2.73%--3.41% for ordinary size 12/14, and
0.87%--3.07% and 0.08%--1.08% for pivot-heavy size 12/14. Only 25 of 140
samples were faster, with zero checksum mismatch. The change was fully
reverted and the rebuilt benchmark SHA-256 exactly matched its baseline.

Multiplier blocks are now formed directly in the column-major layout consumed
by the forward and backward solves; the Schur update reads the same storage,
so no transpose pass or duplicate multiplier buffer is needed. Fixed-size
single-right-hand-side stages use local stack storage, and the fixed dispatch
tests the profile-dominant 12x12 shape first. In seven order-alternated pairs
over 16--256 stages, sequential eight-right-hand-side solve medians fall by
31.39%--33.29% for size 5, 25.75%--26.99% for size 8, 19.76%--23.55% for size
12, and 20.16%--22.93% for size 14. The fused `solve_many` path improves by
2.68%--4.90%, 3.80%--5.22%, 6.63%--8.08%, and 2.41%--5.25%, respectively.
All 560 paired checksum comparisons match; factor timings move between
`-2.53%` and `+3.84%`, with no systematic factorization claim.

Packed stage inputs are now validated while they are copied into owning
storage. Each diagonal pair is checked and symmetrically projected once rather
than once per orientation, lower-block finite validation is fused with its
copy, and the bordered wrapper delegates stage-block validation to the stage
solver instead of rescanning it. Halving each symmetric operand before adding
also keeps finite same-sign values near `Number::max()` from overflowing during
projection. Seven paired numeric-factor microbenchmarks over 16--256 stages
improve size-12 medians by 9.20%--11.35% and size-14 medians by 7.51%--10.53%,
with no checksum mismatch. Dedicated `NaN` and maximum-finite regressions cover
the fused validation and the overflow-safe projection.

The numeric inverse is now projected once onto the symmetry invariant shared
by multiplier formation, Schur formation, and stage solves. Consequently
`L A^-1 L^T` is evaluated like a symmetric rank-k update: only one triangle is
formed and the result is mirrored. The certified Jacobi path retains its
two-sided update and error-bound protocol. Against the immediately preceding
numeric kernel, seven paired
microbenchmarks reduce 12x12 factorization by 10.93%--12.83% and 14x14 by
9.65%--10.84%.

The bordered backend now explicitly prepares its stable packed matrix
workspace. A new or changed workspace is cleared once; subsequent assemblies
overwrite every structurally live coefficient and retain only permanent zero
slots, while arbitrary external spans keep from-scratch clearing semantics.
The adjacent-stage Hessian rectangle and incoming-dynamics rectangle are
disjoint, so the former is a complete assignment rather than a stale-value
accumulation. A regression test compares repeated assembly at changed
regularization against intentionally `NaN`-poisoned fresh storage. In the
order-alternated CAR2 comparison this removes 3.59% of measured assembly time
without changing inertia or either residual gate.

The independent full-inertia proof now traverses each symmetric Hessian,
cross-stage, and border edge once and adds its magnitude to both endpoint
Gershgorin row sums. Its scratch vector is allocated with the configured
assembler workspace and reused. Exact symmetry checks, roundoff margins,
slack and diagonal positivity tests, the required inertia signature, pivot
checks, and both residual gates are unchanged.

A matched portable `x86-64-v3` Callgrind profile moved the candidate from
32.606 to 21.212 billion instructions (`-34.94%`), versus 31.078 billion for
the ordinary baseline (`-31.75%`). The reusable-workspace change moved the
preceding 24.955 billion profile to 24.513 billion (`-1.77%`) and removed all
419.954 million instructions attributed to repeated packed-storage `memset`
calls. The symmetric Schur change then moved 24.513 to 24.302 billion
(`-0.86%`); Schur-update self cost fell from 786.0 to 472.6 million
instructions (`-39.88%`). Dead-store elimination, fused permutation/symmetry
restoration, and 14x14 dispatch then move 24.302 to 24.052 billion (`-1.03%`):
numeric-stage-factor self cost falls from 1.919 to 1.815 billion (`-5.43%`),
and the former 143.2-million-instruction symmetry pass is no longer separate.
The solve-layout and fixed single-RHS changes then move 24.052 to 23.682
billion (`-1.54%`); `SolveOne` self cost falls from 705.1 to 384.4 million
instructions (`-45.48%`), while numeric-stage-factor self cost is unchanged
within 0.05%.
Single-pass validation and copy then move 23.682 to 22.959 billion
instructions (`-3.05%`). The complete stage-solver `factorize` aggregate falls
from 953.7 to 579.4 million instructions (`-39.25%`), while the bordered
wrapper aggregate falls from 380.3 to 31.5 million (`-91.71%`); the numeric
elimination kernel itself is unchanged within 0.01%.
The edge-once independent-inertia proof moves 22.959 to 22.918 billion
instructions (`-0.18%`). Assembler-family instructions fall from 1.307 to
1.267 billion (`-3.08%`, 40.270 million saved), and assembler self cost falls
from 1.189 to 1.159 billion (`-2.52%`); numeric factorization, Schur update,
and `SolveOne` counts are unchanged exactly.
The symmetric matrix-application and internal nonalias refinement paths then
move the complete profile from 22.918 to 22.710 billion instructions
(`-0.91%`). Inclusive `solve_refined_rhs` cost falls from 1.142 to 0.972
billion (`-14.95%`). Within matched builds, the final size-12/14 dispatch alone
moves 22.751 to 22.710 billion (`-0.18%`) and reduces
`solve_refined_rhs` from 1.012 to 0.972 billion (`-3.98%`).
The normalized-assembler changes then move 22.710 to 22.683 billion for the
mirrored Gram, 22.659 billion for sparse `-I` writes, and 22.550 billion for
fixed four-state products. Compile-time instantiation of the unchanged checked
four-state inverse then moves the profile to 22.414 billion, and straight-line
four-state Gram/state outputs move it to 22.344 billion. Together the five
steps save 365.908 million instructions globally (`-1.61%`) and move inclusive
assembler cost from 2.481 to 2.101 billion (`-15.29%`). The inverse step alone
saves 135.439 million instructions globally (`-0.60%`), reduces inclusive
assembler cost by 137.365 million (`-5.96%`), and reduces assembler self cost
from 956.510 to 822.732 million (`-13.99%`). Pivot choice, eager augmented
identity initialization, finite and singularity checks, and the normalized
inverse backward-error gate retain the same code and operation order. The
straight-line product step saves another 69.588 million global instructions
(`-0.31%`) and 67.662 million inclusive assembler instructions (`-3.12%`);
the fixed Gram helper falls by 71.70%, fixed normalized dynamics by 17.73%,
and their combined instruction count by 50.90%. Finally, the off-diagonal
inertia scan is fused with the matrix-copy passes without changing the
edge-addition order. This moves 22.344 to 22.259 billion instructions
(`-0.38%`) and inclusive assembler cost from 2.101 to 2.014 billion
(`-4.17%`); the remaining diagonal-only certificate falls from 195.268 to
30.274 million instructions (`-84.50%`). Both Hessian orientations are still
copied independently, so exact asymmetry rejection is not a symmetry
projection. The CAR2 trace contains 175 assemblies, two pre-dual-regularization
misses, 173 exact certificates, 173 factorizations, and 171 accepted solves.
A regression requires the symmetric four-state case to return exact inertia
`(9,4,0)`, then verifies that changing only `H[0,1] != H[1,0]` preserves both
stored values and suppresses the certificate. The associated native series is
excluded from wall-clock claims because total CV is 3.25%--4.07% and assembly
CV is 4.92%--5.40%, despite lower assembly time in all seven pairs.

The sparse derivative providers then prepare immutable scatter plans from the
validated structure. When every source coordinate has a unique packed target,
each evaluation overwrites mapped coefficients and clears only structural
holes; duplicate-target plans retain the former full clear and `+=` order.
The unconditional source-finite gate remains, while the packed-output scan is
skipped only when unique writes make accumulation overflow impossible. This
moves the complete profile from 22.259 to 21.947 billion instructions
(`-1.40%`), inclusive prepared-provider cost from 1.168 to 0.851 billion
(`-27.15%`), and inclusive assembler cost from 2.014 to 1.697 billion
(`-15.75%`). Clean and `NaN`-poisoned external buffers produce identical
outputs in both stage and bordered-stage regressions; signed zero, duplicate
coordinates, and invalid targets have focused coverage. CAR2 still accepts
171/171 requests with zero fallback, while DTOC3 N=5000 and N=30000 each
accept their single nonbordered candidate with constraint violation below
`9.4e-16`. No wall-clock claim is made from the single native correctness runs.

The reusable KKT operator now obtains `J*v` and `J^T*w` through one
transactional AnyAny operation. Models may implement the fused callback
directly; a fully materialized model evaluates Jacobian values once and
updates both products in source-entry order, while mixed/native models retain
the former transpose-then-forward callback order. Every input may alias either
output, overlapping outputs are rejected, and scratch results are committed
only after the complete pair succeeds. The legacy coordinate adapter expands
the point and both directions once before forwarding the pair. On CAR2 this
moves the portable profile from 21.947 to 21.826 billion instructions
(`-0.55%`), reduces inclusive `NlpKktOperator::apply` cost from 735.114 to
613.792 million (`-16.50%`), and replaces two outer AnyAny product chains
costing 407.583 million instructions with one costing 286.260 million
(`-29.77%`). Jacobian cache hits fall from 513 to 342 while the 172 actual
Jacobian evaluations and the nonlinear trajectory remain unchanged. CAR2 is
still 171/171 accepted with no fallback, and both large DTOC3 correctness
cases retain 1/1 acceptance and sub-`9.4e-16` violation. These single native
runs add no wall-clock claim.

The legacy canary now retains the full `PrimalDualKktOperator` beside its
already persistent coordinate problem for one `PDSystemSolver::InitializeImpl`
lifetime. This preserves the live TNLP callbacks while reusing the validated
layout, bound maps, structure fingerprint, and KKT scratch vectors. The cached
operator is reset before its borrowed problem owner; reoptimization constructs
one fresh operator after reinitialization. Restoration and other
matrix-snapshot paths remain transient so that numeric snapshots never cross
solves. On CAR2 this moves the portable profile from 21.826 to 21.555 billion
instructions (`-1.24%`). KKT construction falls from 171 calls to one and from
124.508 to 0.728 million inclusive instructions (`-99.42%`); the four bound-map
queries per solve fall from 684 calls to four and from 108.087 to 0.634 million
instructions (`-99.41%`). Inclusive `EvaluateSolve` cost falls by 270.426
million instructions (`-3.44%`). CAR2 retains 171/171 acceptance, zero
fallback, the same factorization/backsolve/quality counts, objective, and
nonlinear trajectory. DTOC3 N=5000 and N=30000 again retain 1/1 acceptance and
sub-`9.4e-16` violation. Lifecycle accounting is exercised across ordinary,
`ReOptimizeNLP`, forced-restoration, and injected candidate-failure tests; the
full Release and ASan+UBSan+leak suites remain 15/15. These single native runs
add no wall-clock claim.

The legacy evaluation bridge now writes its eleven current-state blocks and
flat eight-block RHS/reference directions directly into per-`PDSystemSolver`
contiguous workspaces. Equal-precision builds fill the final C++23 storage
without the former stable-vector/conversion-vector pair; mixed-precision builds
retain one monotonically grown stable-number scratch. Equality and inequality
multipliers are adjacent and borrowed as one span, reconstructed residual
storage is reused, and the accepted direction is scaled in place after the
unscaled true-residual gate. Workspaces retain capacity across reinitialization,
but every span is rebuilt after resize and the synchronous backend may not
retain any view. Main and restoration solvers own separate storage, while a
failed private copy or backend request cannot publish partially written data.
On CAR2 this moves the portable profile from 21.555 to 21.371 billion
instructions (`-0.85%`). Inclusive `EvaluateSolve` cost falls by 183.266
million instructions (`-2.41%`), and the flat-RHS path falls from 94.973 to
28.116 million (`-70.40%`). The 3,249 owning `CopyVector` calls disappear;
profile-wide `memcpy` falls by 48.426 million instructions (`-5.17%`). CAR2
retains 171/171 acceptance, zero fallback, the same work counts, objective,
and trajectory; both DTOC3 sizes remain 1/1 with sub-`9.4e-16` violation.
Release and debug `.text` are identical and their Callgrind totals differ by
21 instructions. Growth/reuse accounting covers ordinary solves,
`ReOptimizeNLP`, separate restoration storage, and recovery after injected
backend failure; Release and ASan+UBSan+leak remain 15/15. The single native
runs remain correctness checks and add no wall-clock claim.

`CommitDirection` now consumes the scaled direction already checked finite by
`EvaluateSolve`. When stable and C++23 scalar types are identical, it passes
that immutable storage directly to the eight `PutValuesInVector` calls. A
mixed-precision build still converts into the solver-owned stable scratch and
checks representability completely before the first detached write. The
replacement `IteratesVector` remains detached, all eight blocks are populated,
and the legacy result remains the fallback until the final `result.Copy`.
This removes 171 temporary full-direction vectors and their duplicate identity
conversion/finite scan. On CAR2 the portable profile moves from 21.371 to
21.325 billion instructions (`-0.22%`); inclusive commit cost falls from
57.354 to 10.704 million (`-81.34%`) and commit self cost falls by 99.85%.
CAR2 remains 171/171 with zero fallback and unchanged work counts, objective,
and trajectory; DTOC3 N=5000 and N=30000 remain 1/1 with sub-`9.4e-16`
violation. Release/debug `.text` is identical with a 61-instruction Callgrind
difference, and both 15-test validation suites remain clean. The native runs
remain correctness-only and make no new wall-clock claim.

The bordered assembler now records a successful complementarity-state check
only after the complete assembly returns successfully. Reconstruction reuses
that proof only for the same nonzero caller-owned numeric revision; every
assembly attempt still clears the proof and scans all eight dual/slack spans,
while revision zero or a new revision is checked normally. On CAR2 this moves
the portable profile from 21.325 to 21.310 billion instructions (`-0.07%`).
The 171 redundant reconstruction scans disappear, reducing total
`ValidComplementarityState` calls from 346 to 175 and its inclusive work from
29.411 to 14.876 million instructions; all 175 assembly checks remain. CAR2
retains 171/171 acceptance, zero fallback, unchanged work counts, objective,
and trajectory; both DTOC3 sizes remain 1/1 with sub-`9.4e-16` violation.
Release/debug `.text` is identical with a one-instruction Callgrind difference,
and a focused changed-revision regression plus both 15-test validation suites
remain clean. The native runs remain correctness-only and make no new
wall-clock claim.

Four-state normalized reconstruction now loads the four dynamics constraint
positions and four normalized direction values once, then emits the four
original directions through a C++23 fixed-size helper. Every output retains
the prior sequential accumulation order `0 + term0 + term1 + term2 + term3`;
all other state counts retain the generic nested-loop fallback. On CAR2 this
moves the portable profile from 21.310 to 21.264 billion instructions
(`-0.22%`). Inclusive candidate-solve cost falls by 46.285 million
instructions (`-0.71%`), and its self cost falls from 112.660 to 71.319
million (`-36.70%`). All CUTEst Hessian, Jacobian, constraint, and objective
callback counts and instruction costs remain identical. A `NaN`-poisoned
13-entry regression requires exact equality to an independent scalar
reconstruction. CAR2 retains 171/171 acceptance, zero fallback, unchanged work
counts and objective, while both DTOC3 sizes remain 1/1 with sub-`9.4e-16`
violation. Release/debug `.text` is identical with a 57-instruction Callgrind
difference, both 15-test suites remain clean, and the single native runs add no
wall-clock claim.

Full-direction reconstruction now carries an explicit, conservative overwrite
certificate in both structured layouts. The three built-in assemblers publish
it only after live binding/configuration has proved a complete reduced
permutation and complementarity reconstruction. Certified `NDEBUG` backends
skip the otherwise redundant full-vector prefill; unknown assemblers retain
`NaN` poisoning, and every non-`NDEBUG` build poisons even certified output so
an omitted write is rejected by the existing finite-output gate. On CAR2 this
moves the portable profile from 21.264 to 21.252 billion instructions
(`-0.05%`). Inclusive candidate-solve cost falls by 11.622 million instructions
(`-0.18%`), while its self cost falls from 71.319 to 59.690 million
(`-16.31%`). CAR2 retains 171/171 acceptance, zero fallback, unchanged work
counts, objective, and Callgrind residual; DTOC3 N=5000 and N=30000 remain 1/1
with their prior sub-`9.4e-16` violations. Release/debug `.text` is identical
with a 24-instruction Callgrind difference. A deliberately incomplete generic
assembler and `NaN`-poisoned full bordered reconstruction cover both sides of
the capability, and both 15-test suites remain clean. The single native runs
remain correctness-only and make no new wall-clock claim.

Candidate-first requests now optionally carry caller-owned mutable direction
storage. A structured backend writes the complete reconstructed direction into
that span synchronously and returns an explicit marker with an empty owning
result; callers that do not provide storage retain the original owning-vector
fallback. The backend never retains the request span. The legacy bridge lends
its existing flat-reference workspace only when no reference direction is
needed, then performs the unchanged dimension, finite, true-residual, scaling,
and detached-commit gates before the next solve can reuse it. Failed requests
may clobber private scratch but cannot publish it. On CAR2 this moves the
portable profile from 21.252 to 21.235 billion instructions (`-0.08%`). The 171
hot full-vector copies disappear: profile-wide const-vector assignment falls
from 173 calls and 23.411 million inclusive instructions to two calls and
0.097 million. Inclusive candidate-solve cost falls from 6.478 to 6.458 billion
instructions (`-0.32%`), and inclusive `EvaluateSolve` cost falls from 7.336
to 7.319 billion (`-0.24%`). The added ownership/lifetime checks account for
about 2.9 million extra self instructions in each enclosing layer; assembly
and prepared-derivative-provider costs are unchanged exactly. CAR2 retains
171/171 acceptance, zero fallback, unchanged work counts, objective, and
Callgrind residual; both DTOC3 sizes remain 1/1 with their prior sub-`9.4e-16`
violations. Release/debug `.text` is identical with a 148-instruction
Callgrind difference, and both 15-test suites remain clean. The single native
runs remain correctness-only and make no new wall-clock claim.

The final true-residual reduction, direction scaling, and post-scale finite
gate now share one index-ordered traversal in `EvaluateSolve`. The two
residual maxima retain the exact operation order of `RelativeInfinityError`,
every direction entry is still scaled, and the finite result is rejected only
after the traversal. The independent unscaled-direction finite gate before
the true KKT application and the later reference-direction comparison remain
unchanged. On CAR2 this moves the portable profile from 21.235 to 21.212
billion instructions (`-0.11%`). The former 171 standalone residual-reduction
calls and their 29.072 million instructions disappear; fused work adds 5.813
million self instructions to `EvaluateSolve`, whose inclusive cost nevertheless
falls from 7.319 to 7.296 billion (`-0.31%`). `apply_flat`, `FlattenInto`,
assembly, the prepared derivative provider, and every CUTEst callback count
and cost remain unchanged. CAR2 retains 171/171 acceptance, zero fallback,
unchanged work counts, objective, and Callgrind residual; both DTOC3 sizes
remain 1/1 with their prior sub-`9.4e-16` violations. A focused regression
checks exact residual equality, complete scaling, and overflow rejection;
Release/debug `.text` is identical with a 259-instruction Callgrind difference,
and both 15-test suites remain clean. The single native runs remain
correctness-only and make no new wall-clock claim.

Coordinate initialization and nested
fingerprint self costs each fell by 99.42%; numeric stage-factor self cost fell
from 2.961 to 1.773 billion instructions before the symmetry projection
(`-40.12%`), including `-20.45%` from the pivot-column change. These
instruction counts explain the
optimization; the CPU-pinned Release series above remains the authoritative
wall-clock gate.

This addresses concrete problems in the experimental `jgillis/Ipopt-1:jac_vp`
branch: its callback is pure virtual, callback failures are ignored, the
forward path can use stale `x`, and each product allocates temporary arrays.
The façade instead gives every model a correct fallback while preserving a
native fast path.

The FGMRES implementation follows Saad's flexible method: it stores every
preconditioned direction `Z[k]`, solves the complete projected triangular
system, and forms `x0 + sum(y[k] * Z[k])` from an immutable restart base. This
was cross-checked against the
[original paper](https://doi.org/10.1137/0914028) and the
[PETSc FGMRES implementation](https://petsc.org/release/src/ksp/ksp/impls/gmres/fgmres/fgmres.c.html).
Unlike PR #773, convergence exactly at the iteration limit succeeds and a zero
maximum performs no Krylov step.

## Full-space FGMRES refinement

The stable `PDFullSpaceSolver` also has an opt-in FGMRES implementation in its
normal post-direct-solve refinement phase. `linear_system_refinement` keeps
`iterative-refinement` as the default and adds `fgmres-kdelta`, for
`K_delta / M = K_delta`, and `fgmres-k`, for `K / M = K_delta`. Both variants
take the already computed direct step as `x0`, apply the complete eight-block
primal-dual KKT operator, and use the exact current tagged `K_delta`
factorization as a right preconditioner. A stale factorization is rejected;
the preconditioner path cannot enter perturbation selection, quality
improvement, or factorization retry. `fgmres_restart`,
`fgmres_breakdown_tolerance`, and `fgmres_reorthogonalization` control the
Arnoldi cycle, while the existing refinement iteration and residual options
retain their outer-solver meaning.

The focused live-Ipopt test forces nonzero equality/inequality perturbations,
checks both operator choices against their corresponding full residual, and
requires zero preconditioner cache misses. It also checks the work identities:
classic refinement performs one full-KKT residual per phase and iteration,
whereas FGMRES performs the phase residual plus an Arnoldi KKT application and
a true-residual validation per Krylov iteration.

Interleaved CPU-2 measurements used MUMPS, one OpenBLAS thread, the canonical
unscaled `make_constraint` setup, seven natural runs, and five forced-`cd`
runs. Values below are medians; each triple is
classic / `fgmres-kdelta` / `fgmres-k`.

| problem and perturbation | Optimize ms | refinement ms | Ipopt iterations |
| --- | ---: | ---: | ---: |
| CAR2, natural | 1941.76 / 2002.18 / 1998.57 | 234.40 / 289.91 / 289.46 | 236 / 236 / 236 |
| DTOC3 N=5000, natural | 25.58 / 25.29 / 25.32 | 1.61 / 1.81 / 1.80 | 1 / 1 / 1 |
| CAR2, forced `cd` | 2493.22 / 1670.81 / 1412.42 | 1070.83 / 775.08 / 664.77 | 157 / 114 / 96 |
| DTOC3 N=5000, forced `cd` | 33.27 / 34.07 / 25.32 | 3.26 / 3.67 / 1.88 | 2 / 2 / 1 |

| problem and perturbation | backsolves | KKT applications | factorization attempts |
| --- | ---: | ---: | ---: |
| CAR2, natural | 240 / 241 / 241 | 476 / 718 / 718 | 236 / 236 / 236 |
| DTOC3 N=5000, natural | 1 / 1 / 1 | 2 / 3 / 3 | 1 / 1 / 1 |
| CAR2, forced `cd` | 1305 / 726 / 622 | 1499 / 1574 / 1346 | 244 / 130 / 104 |
| DTOC3 N=5000, forced `cd` | 2 / 2 / 1 | 4 / 6 / 3 | 2 / 2 / 1 |

Every measured solve converged and every preconditioner cache-miss count was
zero. With zero perturbations the two FGMRES modes have identical trajectories.
DTOC3's natural trajectory is also bit-for-bit identical to classic
refinement. CAR2 keeps the same 237 points and discrete line-search path; its
largest aligned differences are `5.1e-8` in objective/infeasibility and
`5.2e-7` in direction norm. Thus natural CAR2 exposes FGMRES's extra KKT apply
as a 23.5%--23.7% refinement-time cost and about a 3% total cost; DTOC3 total
times overlap and support no speed claim. Forced perturbations deliberately
separate the operators: CAR2 takes 158 / 115 / 97 trajectory points and DTOC3
takes 3 / 3 / 2. Those path changes explain the large forced-case time gains;
they are not isolated kernel speedups. The raw samples, counters, convergence
values, and trajectory comparison are recorded in
`graphify-out/pdfullspace-fgmres-refinement-benchmark.json`.

AnyAny is pinned to commit `560301c278e46bb527e49b1739dcb841035a9932`
(release v1.2.1) and the source archive is verified by SHA-256.

Build and test:

```sh
cmake -S cxx23 -B build-cxx23 \
  -DCMAKE_BUILD_TYPE=Release \
  -DIPOPT_CXX23_BUILD_BENCHMARKS=ON
cmake --build build-cxx23 -j
ctest --test-dir build-cxx23 --output-on-failure
taskset -c 4 \
  build-cxx23/ipopt_cxx23_anyany_jacobian_product_benchmark
taskset -c 4 \
  build-cxx23/ipopt_cxx23_fgmres_reuse_benchmark
taskset -c 4 \
  build-cxx23/ipopt_cxx23_block_tridiagonal_solver_benchmark
```

The live bridge, augmented-system, and runtime-canary tests are opt-in because
they link the stable Ipopt library. With
an installed `ipopt.pc`, add
`-DIPOPT_CXX23_BUILD_LEGACY_BRIDGE_TESTS=ON`. A source-tree build can instead
set both `IPOPT_CXX23_LEGACY_IPOPT_INCLUDE_DIRECTORIES` and
`IPOPT_CXX23_LEGACY_IPOPT_LIBRARY`. Put the generated build-tree
`src/Common` directory in the include list; CMake reads its
`config_ipopt.h` and propagates the `IPOPT_SINGLE`/`IPOPT_INT64` ABI macros.

The runtime gate covers user scaling, all four fixed-variable treatments,
forced restoration, `ReOptimizeNLP`, and an injected non-fatal mismatch. The
source-tree C++23 gate currently passes 16/16 in both Release and
ASan+UBSan+leak; the earlier broad gate also passes against genuine
single-precision Ipopt. An installed unmodified Ipopt 3.14.20 remains source
and binary compatible and passes its 11/11 compatible subset in both Release
and ASan+UBSan+leak, but
cannot expose its internally constructed
restoration solver to the new factory seam. Both normal/reoptimized and forced
restoration solves commit in `validated_replacement`; an injected mismatch
proves fallback. Candidate-first additionally covers all fixed-variable
treatments, normal/reoptimized/restoration commits, zero-reference happy paths,
and exactly-one-reference backend, inertia, and residual fallbacks. Reported
factorization/backsolve/KKT-product totals are checked against the injected
backend even for rejected attempts; the exact ASan run is recorded in
`graphify-out/candidate-first-canary-counts.json`. The interleaved
C++23 live primal-dual stage test additionally accepts the structured backend
for both the bounded main NLP and forced `RestoIpoptNLP`; every candidate
request is accepted and the outer full-KKT residual remains mandatory. The
interleaved
CUTEst/`fatrop-research` gate is recorded in
`graphify-out/cutest-ipopt-comparison.json`: the ABI-safe adapter change is
neutral on full solves, while structured FATROP is 1.57x to 7.93x faster on
the measured DTOC cases. The separate live DTOC3 candidate-first gate is in
`graphify-out/dtoc3-stage-candidate-benchmark.json`. Across N=50, 5000, and
30000, its seven-run interleaved Optimize-only medians are 7.86%--10.83%
lower than the same executable without the stage backend, with one request,
one accepted candidate, and zero fallback/factory failures in every run.
Candidate callback caching preserves the inner structured and outer true-KKT
residual checks while keeping actual CUTEst work at two Jacobian and one
Hessian evaluations. Cold preparation is reported separately and outweighs
the first large solve; its measured break-even is the third repeated solve for
N=5000 and N=30000. The synthetic block benchmark independently shows linear
numeric factorization at 0.248--0.255 microseconds/stage for 5x5 blocks and
0.766--0.775 microseconds/stage for 8x8 blocks, 1.80x--1.94x faster than the
certified factor in five pinned Release runs. Scope, medians, checksums, and
limitations are recorded in `graphify-out/block-tridiagonal-benchmark.json`.
The code from PR #773 was not copied directly:
its incremental solution update does not apply changed coefficients of prior
Krylov basis vectors, its zero-iteration limit can fail to make progress, and
its norm protocol widens the old matrix hierarchy substantially.
