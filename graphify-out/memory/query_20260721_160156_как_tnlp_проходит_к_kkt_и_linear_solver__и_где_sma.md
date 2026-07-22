---
type: "architecture"
date: "2026-07-21T16:01:56.317982+00:00"
question: "Как TNLP проходит к KKT и linear solver, и где SmartPtr создаёт связанность?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["TNLPAdapter()", "AlgorithmBuilder::BuildIpoptObjects()", "AlgorithmBuilder::SymLinearSolverFactory()", "IpoptCalculatedQuantities()", "SmartPtr", "TSymLinearSolver"]
---

# Q: Как TNLP проходит к KKT и linear solver, и где SmartPtr создаёт связанность?

## Answer

Расширение запроса: tnlp adapter calculated quantities kkt solver smart ptr linear augmented application algorithm. Публичный TNLP проходит через TNLPAdapter; AlgorithmBuilder собирает OrigIpoptNLP, IpoptData и IpoptCalculatedQuantities и выбирает backend симметричного линейного решателя; IpoptCalculatedQuantities является узлом кэширования вычисляемых величин. SmartPtr — крупный узел владения и связанности, но одна центральность графа не доказывает runtime-hotspot. Из-за предупреждений о dangling edges и схлопнутых endpoints выводы требуют проверки в исходниках.

## Outcome

- Signal: useful

## Source Nodes

- TNLPAdapter()
- AlgorithmBuilder::BuildIpoptObjects()
- AlgorithmBuilder::SymLinearSolverFactory()
- IpoptCalculatedQuantities()
- SmartPtr
- TSymLinearSolver