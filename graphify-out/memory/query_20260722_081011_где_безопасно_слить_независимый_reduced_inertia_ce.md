---
type: "query"
date: "2026-07-22T08:10:11.954441+00:00"
question: "Где безопасно слить независимый reduced-inertia certificate с копированием bordered KKT, сохранив асимметрию, порядок сумм и residual gates?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["PrimalDualBorderedStageKktAssembler", ".CertifyReducedInertia()", ".assemble_bordered_stage_system()", "BorderedStageStructuredCandidateBackend"]
---

# Q: Где безопасно слить независимый reduced-inertia certificate с копированием bordered KKT, сохранив асимметрию, порядок сумм и residual gates?

## Answer

Expanded from graph vocabulary: [independent, inertia, certificate, reduced, hessian, cross, border, copy, symmetric, proof, validation, assembler]. DFS isolated PrimalDualBorderedStageKktAssembler and CertifyReducedInertia as the remaining duplicate scan. The accepted implementation accumulates global, physical-stage, local-global, and cross-stage Gershgorin edges during their existing copy passes in the same floating-point order. It stores both Hessian orientations independently and carries an exact-equality flag, so asymmetry is preserved and rejects certification. CAR2 remains 171/171 accepted with zero fallback, 171 iterations, objective 2.66608466558, and 173 exact certificate-bearing factorizations after two initial non-certifiable dual-regularization assemblies. Portable Callgrind falls 22,344,481,894 to 22,258,654,203 instructions (-0.3841%); inclusive assembler cost falls 2,101,409,728 to 2,013,700,743 (-4.1738%); certificate self cost falls 195,268,031 to 30,274,233 (-84.4961%). Fixed four-state Gram and dynamics counts remain exact. Release and ASan/UBSan pass 15/15. A regression requires symmetric exact inertia (9,4,0), then verifies unequal H[0,1]/H[1,0] values survive while the certificate is absent. The native series is excluded because CV is 3.25%-5.40%; no wall-clock claim is made.

## Outcome

- Signal: useful

## Source Nodes

- PrimalDualBorderedStageKktAssembler
- .CertifyReducedInertia()
- .assemble_bordered_stage_system()
- BorderedStageStructuredCandidateBackend