---
type: "query"
date: "2026-07-21T18:43:38.761839+00:00"
question: "Какие живые maps fixed variables, equality/inequality ordering и NLPScalingObject scales должен экспортировать TNLPAdapter/OrigIpoptNLP в тонкий C++23 bridge, кто ими владеет и когда они становятся валидны?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["TNLPAdapter()", "OrigIpoptNLP()", "NLPScalingObject", "IpTNLPAdapter.cpp", "IpOrigIpoptNLP.cpp", "IpNLPScaling.hpp"]
---

# Q: Какие живые maps fixed variables, equality/inequality ordering и NLPScalingObject scales должен экспортировать TNLPAdapter/OrigIpoptNLP в тонкий C++23 bridge, кто ими владеет и когда они становятся валидны?

## Answer

После успешного OrigIpoptNLP::InitializeStructures TNLPAdapter уже публично отдает GetPermutationMatrices, GetFixedVariables, GetC_Rhs и ResortX. GetSpaces создает P_x_full_x/P_c_g/P_d_g, GetBoundsInformation заполняет fixed values и c_rhs, затем OrigIpoptNLP вызывает NLPScalingObject::DetermineScaling. Фактические post-DetermineScaling диагонали извлекаются без downcast применением apply_vector_scaling_x/c/d к векторам единиц; objective factor равен apply_obj_scaling(1). Bridge обязан проверить, что OrigIpoptNLP::nlp() указывает на тот же TNLPAdapter, скопировать все positions/RHS/scales в owning vectors и держать исходный TNLP через SmartPtr. MAKE_PARAMETER и MAKE_PARAMETER_NODUAL используют P_x_full_x и fixed template, MAKE_CONSTRAINT добавляет fixed equality rows, RELAX_BOUNDS оставляет identity map. Ни один legacy Vector/Matrix view не должен переживать export.

## Outcome

- Signal: useful

## Source Nodes

- TNLPAdapter()
- OrigIpoptNLP()
- NLPScalingObject
- IpTNLPAdapter.cpp
- IpOrigIpoptNLP.cpp
- IpNLPScaling.hpp