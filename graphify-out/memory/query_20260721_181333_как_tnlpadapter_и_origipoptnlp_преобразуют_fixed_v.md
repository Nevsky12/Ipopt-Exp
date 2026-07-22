---
type: "query"
date: "2026-07-21T18:13:33.317790+00:00"
question: "Как TNLPAdapter и OrigIpoptNLP преобразуют fixed variables, ограничения и scaling для Jv, JTv и Hv?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["TNLPAdapter::GetSpaces()", "TNLPAdapter::ResortX()", "TNLPAdapter::ResortG()", "TNLPAdapter::Eval_h()", "TNLPAdapter::GetScalingParameters()", "ExpansionMatrix::MultVectorImpl()", "ExpansionMatrix::TransMultVectorImpl()", "IpOrigIpoptNLP.cpp"]
---

# Q: Как TNLPAdapter и OrigIpoptNLP преобразуют fixed variables, ограничения и scaling для Jv, JTv и Hv?

## Answer

Expanded from the query via graph vocabulary: [adapter, bounds, constraint, expansion, fixed, gradient, hessian, jacobian, multiplier, scaling, tnlp, unscaled]. Graph traversal identified TNLPAdapter::GetSpaces, ResortX, ResortG, Eval_jac_c, Eval_jac_d, Eval_h, GetScalingParameters, ExpansionMatrix Mult/TransMult, GradientScaling, and IpOrigIpoptNLP.cpp. Source verification established a two-layer composition: TNLPAdapter first expands reduced variables, partitions full g into c and d, subtracts equality RHS, filters Jacobian/Hessian entries, and appends identity rows for MAKE_CONSTRAINT; OrigIpoptNLP then uses x_s=Dx*x, c_s=Dc*c, d_s=Dd*d and f_s=df*f. Therefore J_s*v=Dg*J*Dx^-1*v, J_s^T*w=Dx^-1*J^T*Dg*w, and H_s*v=Dx^-1*H(df*objective_factor,Dg*y)*Dx^-1*v. A header-only LegacyCoordinateAdapter now implements this contract with construction-owned workspaces, strong output guarantees, map/scaling validation, source/map fingerprinting, and native/fallback capability preservation. Dense 4x3 and KKT regression tests plus Release Werror and ASan/UBSan/LeakSanitizer pass.

## Outcome

- Signal: useful

## Source Nodes

- TNLPAdapter::GetSpaces()
- TNLPAdapter::ResortX()
- TNLPAdapter::ResortG()
- TNLPAdapter::Eval_h()
- TNLPAdapter::GetScalingParameters()
- ExpansionMatrix::MultVectorImpl()
- ExpansionMatrix::TransMultVectorImpl()
- IpOrigIpoptNLP.cpp