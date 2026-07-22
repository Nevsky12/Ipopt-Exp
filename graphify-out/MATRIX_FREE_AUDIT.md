# Аудит `jac_vp` и FGMRES для Ipopt

Срез выполнен 2026-07-21 относительно локального `stable/3.14`
`72a29c9aab198afa0dbb940339022a22c415a4eb`. Были извлечены точные head
ветки, а не только патчи из веб-интерфейса:

| Источник | Head | Merge base | Дифф |
|---|---|---|---:|
| [`jgillis/Ipopt-1:jac_vp`](https://github.com/jgillis/Ipopt-1/tree/jac_vp) | `480f2f05722139648db7e905367614f50f8869e0` | Ipopt 3.14.11, `8b042fb3cccb372683cdb0631cb3db32916a9cbb` | 18 файлов, +518/-22 |
| [Ipopt PR #773](https://github.com/coin-or/Ipopt/pull/773), `pghysels:GMRESPDFull` | `2aa4d4447eb343139bd617fe7c23a6d82effee67` | Ipopt 3.14.14, `5fd746337021543e78dc2d167cb1c15db81f68e1` | 39 файлов, +1427/-25 |

## Итог

Обе ветки содержат полезные идеи, но ни одну нельзя переносить целиком.
`jac_vp` задаёт нужную operator capability, а #773 добавляет backward-error
criterion и FGMRES refinement. Однако текущая комбинация:

- не является полностью matrix-free;
- ломает существующий TNLP source/API contract;
- теряет ошибки callback;
- имеет correctness-дефект в обновлении FGMRES-решения;
- выполняет аллокации внутри каждого operator/Krylov шага;
- не имеет достаточной regression matrix для scaling, fixed variables,
  restoration и граничных значений iteration limits.

Поэтому из веток следует извлечь контракт и тестовые случаи, а operator layer
и FGMRES написать заново за стабильным TNLP adapter.

## Ветка `jac_vp`

### Полезное

- Проводит `J*v` и `J^T*v` через TNLP adapter, original NLP, restoration и
  primal-dual residual computation.
- Разделяет equality/inequality части после полного пользовательского
  Jacobian product.
- Содержит режим сравнения product callback с материализованным Jacobian.

### Блокирующие проблемы

1. `TNLP::eval_jac_g_vp` добавлен как pure virtual. Все существующие TNLP
   subclasses перестают собираться, даже когда matrix-free capability им не
   нужна. Capability должна быть optional и иметь корректный sparse fallback.
2. Forward `TNLPAdapter::Eval_jac_vp` не вызывает `update_local_x(x)` и передаёт
   callback поле `full_x_`. После изменения iterate это может быть устаревший
   `x`; сам параметр `x` функции фактически игнорируется.
3. Возврат `bool` обоих пользовательских product callbacks игнорируется, а
   adapter всегда возвращает `true`. Ошибка производной превращается в
   продолжение solve с неопределённым результатом.
4. Каждый forward product выделяет `sx` и `vp`; transpose product выделяет `s`
   и иногда ещё `vp`. Это аллокации в самом частом пути Krylov.
5. Реализация делает unchecked release-casts к `DenseVector` и
   `GenTMatrixSpace`; `dynamic_cast` находится только внутри debug assert.
6. `OrigIpoptNLP` сохраняет результат callback в локальный `success`, но не
   использует его. Variable scaling запрещён assertion вместо корректного
   преобразования направления/результата.
7. В transpose path вычисляется `new_x`, однако новый callback не принимает
   этот флаг; значение остаётся неиспользованным.

Даже при `jac_vp=yes` ветка продолжает получать `curr_jac_c()` и
`curr_jac_d()` для direct augmented-system solve и для test/fallback paths.
Она заменяет products при вычислении residual, но не устраняет materialization
и factorization.

## PR #773 / `GMRESPDFull`

### Полезное

- Flexible right preconditioning естественно допускает изменение direct-solve
  preconditioner между итерациями.
- Modified Gram-Schmidt выполняется дважды.
- Критерий использует norm-wise relative backward error, а не только
  евклидову норму residual.
- Есть restart и проверки малого residual/happy breakdown.

### Correctness blockers

Главная ошибка находится в inner iteration. На шаге `j` код заново решает
least-squares и получает весь вектор коэффициентов `y[0..j]`, но применяет
только:

```cpp
res.Axpy(y[j], *Z[j]);
```

После второго шага коэффициенты прежних `Z[0..j-1]` обычно изменились. В
`res` остаются их старые значения, поэтому это уже не FGMRES candidate
`x0 + sum(y[k] * Z[k])`; последующий true residual вычисляется для другого
вектора. Исправление должно либо собирать candidate из неизменного `x0` и всех
текущих коэффициентов, либо применять дельты относительно предыдущего `y`.

Остальные граничные дефекты:

- `gmres_refinement` включён по умолчанию, хотя путь экспериментальный;
- `restart` жёстко равен 3 и только обрезается по maximum iterations;
- допустимые `min_refinement_steps=0, max_refinement_steps=0` дают
  `restart=0`: inner loop пуст, а внешний `while (!done)` не прогрессирует;
- convergence ровно на `max_refinement_steps` всё равно заканчивается
  `return (totit < max_refinement_steps_)`, то есть возвращает failure;
- перед делением `givens_c/s = ... / delta` нет отдельной обработки
  `delta == 0`;
- initial diagnostic печатает `resid.Amax()`, хотя актуальный residual уже
  находится в `V[0]`, а `resid` до этого использовался как scratch для нормы.

### Стоимость и поверхность API

Из 1427 добавленных строк 790 находятся в 37 LinAlg-файлах. Они расширяют
`Matrix` операциями `ComputeRowA1/ComputeColA1`, которые реально используются
для оценки `||KKT||_inf`; это не мёртвый код. Но цена интеграции велика:

- новые pure virtual implementations проходят через почти всю matrix
  hierarchy;
- в восьми типах остаются 14 явных `UNIMPLEMENTED_LINALG_METHOD_CALLED`;
- только часть вызовов в `NrmInf` перехватывает это исключение и продолжает с
  консервативно заниженной оценкой нормы;
- четыре копии slack vectors создаются на каждую оценку нормы;
- arrays Hessenberg/Givens, `V`, `Z` и `IteratesVector` создаются заново на
  каждом restart/inner iteration.

Нужен отдельный operator-level norm capability с явным статусом exact/bound/
unavailable и session-owned workspaces, а не обязательное расширение каждого
старого matrix subclass.

## Почему это пока не matrix-free solver

```text
NLP J*v / J^T*v
       |
       v
primal-dual KKT operator ------> corrected FGMRES
       |                              |
       |                              v
       +---------------------- direct preconditioner
                                      |
                                      v
                         materialized Jc/Jd/W + factorization
```

Product callbacks сокращают sparse matrix-vector passes при residual
evaluation. Но #773 вызывает `SolveOnce` как preconditioner, а тот передаёт
материализованные `W`, `J_c` и `J_d` в `AugSystemSolver`. Такая схема может
улучшить точность direct solve и уменьшить число refactorizations, но сама по
себе не устраняет factorization.

Для настоящего factorization avoidance нужен новый iterative/structured
`AugSystemSolver` либо preconditioner, который строится из дешёвой
аппроксимации. Полный KKT product также требует optional Hessian-vector
capability; одного Jacobian product недостаточно.

### Жизненный цикл существующего direct solve

Повторный RHS уже не обязан означать повторную факторизацию. В
`PDFullSpaceSolver::SolveOnce` cache из тегов `W/J_c/J_d`, диагоналей и slack
данных определяет, совпадает ли augmented system; при совпадении вызывается
`AugSystemSolver::Solve` с текущими perturbations без повторного поиска
perturbation. `StdAugSystemSolver` пересобирает compound matrix только при
смене тегов или scalar deltas, а `TSymLinearSolver` передаёт
`MultiSolve(new_matrix=false)`, когда тег матрицы не изменился. Контракт
`SparseSymLinearSolverInterface` прямо разрешает в этом случае использовать
старую факторизацию; `IncreaseQuality` остаётся законным поводом для
refactorization. `GenAugSystemSolver` поддерживает ту же семантику
`new_matrix`, но пока выделяет общий `rhssol` на каждый вызов.

В FATROP это разделение выражено явнее парами `solve_in_place` /
`solve_in_place_rhs` и `solve` / `solve_rhs`; отдельный graph-solver test
проверяет reuse уже вычисленного фактора. Поэтому новый boundary моделирует
ровно `factorize` + `solve_rhs`, но дополняет неявные matrix tags двумя
проверяемыми ключами: полным structural fingerprint и обязательной ненулевой
numeric revision.

## Реализованный безопасный vertical slice

В `cxx23/` добавлен отдельный, opt-in прототип, не меняющий стабильные C/TNLP
интерфейсы:

- `std::span` для non-owning inputs/outputs;
- `std::expected` для полного распространения callback failures;
- независимые optional capabilities `J*v`, `J^T*v` и Hessian Лагранжиана
  `H*v`;
- materialized triplet fallback для каждого отсутствующего направления;
- проверка размерностей и sparsity до обращения по индексам;
- alias-safe `x`, multipliers, direction и result;
- кэш sparsity и session-owned value/scratch workspaces;
- fingerprint по dimensions, ordered Jacobian/Hessian sparsity, capabilities и
  model revision;
- reusable saddle-point operator `[H + D_x, J^T; J, -D_y]`, не изменяющий
  outputs при ошибке любого callback;
- полный восьмиблочный primal-dual wrapper
  `x/s/y_c/y_d/z_L/z_U/v_L/v_U`, который выражает `J_c/J_d` и bound expansion
  matrices индексными maps, включает complementarity и `delta_x/s/c/d` и
  сохраняет strong-output guarantee;
- `LegacyCoordinateAdapter`, который точно компонует редукцию fixed variables,
  перестановку `g -> [c,d]`, вычитание equality RHS и диагональный scaling
  `OrigIpoptNLP` для значений, `J*v`, `J^T*v` и `H*v`; его структуры сохраняют
  порядок nonzeros `TNLPAdapter`, а все workspaces выделяются при построении;
- owning `MakeLegacyIpoptCoordinateProblem`, который после
  `OrigIpoptNLP::InitializeStructures` копирует живые maps/RHS/scales, проверяет
  identity underlying `TNLPAdapter`, держит тот же `TNLP` через `SmartPtr`,
  переводит C/FORTRAN sparsity и не выпускает legacy views в новый слой;
- movable/non-copyable solve session, который сравнивает полный structural
  fingerprint до Krylov шага и переиспользует FGMRES workspace только для
  совместимого operator instance;
- move-only `PreparedDirectPreconditioner` с AnyAny-erased backend: подготовка
  проверяет размер, structural fingerprint, размеры KKT state и ненулевую
  numeric revision, один раз вызывает `factorize`, после чего Krylov вызывает
  только `solve_rhs`; stale factor, backend failure и nonfinite solution не
  изменяют caller output;
- `LegacyAugSystemDirectBackend`, который переводит полный восьмиблочный RHS в
  существующую четырёхблочную систему, готовит factor нулевым `MultiSolve`,
  повторно использует стабильные tags/vectors и восстанавливает
  `z_L/z_U/v_L/v_U` без передачи ему perturbation/quality policy;
- `LegacyAlgorithmCanaryBuilder`, который через виртуальную factory
  `AlgorithmBuilder` для main и restoration сначала выполняет неизменённый
  reference solve, а затем сверяет принятую direction/residual с C++23
  full-system путём; default shadow не изменяет результат, а opt-in validated
  replacement делает transactional commit только после всех проверок;
- owning `LegacyMatrixSnapshot`, который извлекает compound `W/J_c/J_d` через
  публичный `TripletHelper` и даёт restoration независимые `J*v`, `J^T*v` и
  `H*v` без заимствованного stable view;
- restarted flexible GMRES с constructor-owned workspace, immutable base на
  каждом restart, полной сборкой `x0 + sum(y[k] Z[k])` и true-residual check
  перед commit;
- AnyAny v1.2.1, зафиксированный SHA и SHA-256 архива.

Release-тест собран GCC 13.3 с C++23 и `-Wall -Wextra -Wpedantic -Werror`.
Проверены native, fallback, partial capability, симметричный off-diagonal
Hessian fallback, все виды aliasing, callback failure, dimension mismatch,
invalid sparsity, cache, capability/revision fingerprint и KKT strong-output
guarantee. Полный wrapper проверен ручным эталоном всех восьми блоков,
block/flat in-place aliasing, invalid layout, layout fingerprint, injected
callback failure и сквозным FGMRES solve. Session reuse проверен на двух
численно разных операторах с общей структурой, structural mismatch и callback
failure без изменения caller solution. Compatibility adapter сверен с плотным
4x3 эталоном при удалённой fixed variable, переставленных `c/d`, добавленной
`MAKE_CONSTRAINT`-строке, отрицательном objective scaling и ненулевых
`D_x/D_c/D_d`; отдельно проверены native/fallback capability, aliasing,
structural fingerprint, invalid maps/scales и strong failure guarantee.
Его результат также проходит сквозной `[H+D_x,J^T;J,-D_y]` KKT test. FGMRES
отдельно проверен на changing preconditioner, restart,
повторном использовании solver, семействе nonsymmetric dense matrices 2x2–8x8,
KKT integration, zero/exact maximum iterations, callback failure, NaN и
breakdown. Тот же набор проходит ASan+UBSan с leak detection.

Prepared-direct regression использует точный 3x3 LU backend как правый
preconditioner для двух разных RHS: оба solve сходятся за одну FGMRES-итерацию,
счётчик остаётся на одной факторизации и двух `solve_rhs`. Отдельно проверены
in-place RHS, stale numeric revision до входа в backend, неверные
dimension/structure/state, factorization/solve failures, nonfinite результат и
strong-output guarantee. Release `-Werror` и sanitizer-набор проходят 4/4
standalone tests.

Augmented-system regression связан с публичной библиотекой Ipopt и использует
установленные `SymTMatrix/GenTMatrix` и реализацию контракта
`AugSystemSolver`. Для двух полных 9-компонентных систем direct reduction,
direction и residual совпадают с matrix-free оператором, а exact right
preconditioner даёт одну FGMRES-итерацию. Нулевой prepare-RHS и три последующих
solve оставляют счётчик на одной факторизации. Отдельно проверены inertia/no
inertia, ограниченный `SYMSOLVER_CALL_AGAIN`, singular status, исключение,
nonfinite result, nonpositive slack, strong output и отклонение изменившегося
stable matrix tag. Согласованные живые `IpCq`-диагонали сохраняют исходные
`sigma_x/sigma_s` tags; изменение tag или несогласованное значение отклоняется
до входа в solver. Связанный набор проходит как с source-tree библиотекой, так
и через установленный `ipopt.pc`.

Live-bridge regression поднимает настоящие `TNLPAdapter`, user scaling и
`OrigIpoptNLP`. Для `make_parameter`, `make_parameter_nodual`,
`make_constraint` и `relax_bounds` сверены variable/constraint maps, fixed
template/equalities, equality RHS и `D_x/D_c/D_d/d_f`. На `make_parameter`
C++23 и `OrigIpoptNLP` дают одинаковые objective, constraints, gradient,
ordered scaled Jacobian и Hessian. Отдельно проверены FORTRAN_STYLE,
`new_x/new_lambda=true`, callback failure со strong output и отклонение чужого
owner. Opt-in связанный набор проходит Release `-Werror` и ASan+UBSan с leak
detection: 7/7.

Runtime-canary regression запускает настоящий `IpoptApplication` с явным
`LegacyAlgorithmCanaryBuilder`, exact Hessian и user scaling для
`make_parameter`, `make_parameter_nodual`, `make_constraint` и
`relax_bounds`. Reference `PDFullSpaceSolver` полностью завершает свой
perturbation/inertia/refinement/quality state machine первым; canary использует
уже принятые perturbations, `W/J_c/J_d` и живые `sigma_x/sigma_s`, после чего
сверяет full direction и reconstructed residual. Все проверенные solve
совпадают; в default shadow legacy result остаётся авторитетным.

`LegacyAlgorithmCanaryMode::validated_replacement` после тех же checks
конвертирует все восемь candidate-блоков в detached stable `IteratesVector` и
выполняет единственный финальный `Copy`. Normal, `ReOptimizeNLP` и forced
restoration проходят реальный commit. Callback/solve/conversion/equivalence
failure оставляет уже вычисленный reference result; намеренный mismatch
проверяет этот fallback. Режим пока не может ускорить solve, потому что полный
reference perturbation/inertia/refinement/quality path всегда выполняется
первым.

Добавлен отдельный opt-in `LegacyAlgorithmCanaryMode::candidate_first` с
move-only AnyAny backend contract. В отличие от legacy
`LegacyAugSystemDirectBackend`, этот backend не обязан и не должен вызывать
stable `AugSystemSolver`: он сам выбирает perturbations, выполняет retries,
refinement/quality escalation и возвращает exact inertia certificate вместе с
unscaled full direction и factorization/backsolve counters. Оболочка повторно
проверяет finite/dimensions, exact negative-eigenvalue count, nonnegative
stable-representable regularization и true KKT residual до единственного
detached commit. После commit публикуются `PDPert`; при любой ошибке полный
reference solver вызывается ровно один раз.

Интеграционный dense backend является только test double. Happy path для всех
fixed-variable treatments, normal, `ReOptimizeNLP` и forced restoration не
вызывает reference solve. Backend failure, неверная inertia и residual mismatch
отдельно подтверждают exactly-once fallback; work accounting включает
отклонённую факторизацию. Equality-only OCP assembler уже существует, но live
TNLP/CUTEst stage provider и canary binding ещё отсутствуют; bounds,
inequalities и restoration требуют полной восьмиблочной версии. Поэтому seam
не считается доказанным end-to-end ускорением.
Измеренные ASan counts сохранены в `candidate-first-canary-counts.json`.

За этим seam реализован независимый generic stage backend. Его
`SymmetricBlockTridiagonalSolver` выполняет variable-block `LDL^T`, сохраняет
factor state и обслуживает scalar/fused RHS без stable вызовов. Каждый Schur
block диагонализуется cyclic Jacobi; inertia считается exact только когда
явные backward error `||S-Q Lambda Q^T||_F`, orthogonality error
`||Q^T Q-I||_F` и roundoff margin доказывают, что ни одно собственное значение
не может пересечь ноль. Singular и плохо отделённые случаи отклоняются.

`StageStructuredCandidateBackend<Assembler>` добавляет к kernel полный policy
boundary: fingerprint, assembly с заданными perturbations, точный
congruence-inertia вклад исключённых блоков, primal/dual retry по направлению
inertia mismatch, true-residual refinement и reconstruction полного flat
direction. Tests проверяют indefinite variable blocks, factor-once/solve-many,
несколько RHS, retry, настоящую congruent condensation, неверный inertia
вклад, singular/nonfinite и transactional failure. `StageNlpTopology` теперь
валидирует stage dimensions, packed offsets и canonical-to-generic
permutations, а compile-time provider владеет packed
Hessian/dynamics/path buffers. `EqualityStageKktAssembler` собирает полный
equality-only KKT из `[u_k,x_k,path multiplier,incoming dynamics multiplier]`,
переставляет RHS/direction и кеширует derivatives по numeric revision. Bounds,
slacks, inequalities, restoration и неединичный next-state Jacobian он
намеренно отвергает. Это reference block elimination, не FATROP null-space
Riccati; generic TNLP пока не предоставляет эту stage capability live canary.

Расширенный CUTEst probe подтвердил точную DTOC3 `N=50` topology: 50 стадий,
49 controls, по два state на `Y1...Y50`, 98 линейных dynamics rows
`TT1...TT49`, два fixed initial states и stage-ordered row span 6. Для первого
end-to-end binding выбран `fixed_variable_treatment=make_constraint`, чтобы
оставить `Y1` в primal и представить две identity rows как stage-0 path
equalities.

`AlgorithmBuilder::BuildLineSearch` теперь повторно использует существующую
виртуальную `PDSystemSolverFactory` с префиксом `resto.`, поэтому тот же canary
видит restoration solver без нового virtual ABI slot. Для всех четырёх
fixed-variable treatments принудительно запускается restoration и проверяется
его compound matrix snapshot. Дополнительно проверены `ReOptimizeNLP` и
намеренный mismatch с `maximum_iterations=0`: в fail-open режиме Ipopt всё
равно успешно использует reference direction.

Исходный связанный набор проходил 7/7 с source double и настоящим
`--with-precision=single` + single SMUMPS; после добавления четырёх structured
tests расширенный набор с установленным Ipopt 3.14.20 проходит Release и
ASan+UBSan+leak 11/11. Precision floors выводятся из epsilon `Ipopt::Number`; CMake
считывает generated `config_ipopt.h` и автоматически передаёт ABI macros.
Установленный старый core не имеет restoration factory seam, поэтому его main
path проверяется, но внутренний restoration solver остаётся недоступен.

Формулы сверены с [оригинальной работой Саада](https://doi.org/10.1137/0914028)
и [реализацией PETSc](https://petsc.org/release/src/ksp/ksp/impls/gmres/fgmres/fgmres.c.html):
variable preconditioner применяется к каждому `V[k]`, preconditioned vectors
хранятся отдельно, а solution строится после решения всего projected upper
triangular system.

Минимальный 4x4 native `J*v`, 20 миллионов вызовов, 7 samples, CPU 4:

| Вызов | Median, нс | Диапазон, нс |
|---|---:|---:|
| Прямой `NlpModelAdapter` | 2.512 | 2.439–2.597 |
| Через `AnyNlpProblem` | 6.729 | 6.603–7.295 |

Измеренная цена AnyAny — 4.217 нс/call, или 2.68x на намеренно крошечном
kernel. Это аргумент держать type erasure на крупной operator boundary, а не
внутри scalar/nonzero loops. Данные: `anyany-jacobian-product-benchmark.json`.

Отдельный 8x8/8-step FGMRES benchmark сравнивает одинаковые solves с
constructor-owned workspace и с реконструкцией solver на каждый вызов. Reuse
показал median 2937.7 против 3126.3 нс/solve: экономия 188.6 нс, или 1.064x.
Это microbenchmark стоимости workspace lifecycle, не оценка будущего Ipopt
ускорения. Данные: `fgmres-workspace-reuse-benchmark.json`.

Synthetic block benchmark использует уже упорядоченные 8x8 stage blocks и
восемь RHS. Пять Release-прогонов на CPU 4 дали 1.566--1.609 мкс
factorization/stage для horizons 16--256 и 1.339x--1.468x ускорение fused
contiguous-RHS pass против восьми отдельных проходов. Решения совпали в
`1e-11`, checksums одинаковы. Это kernel result без derivative assembly,
globalization/restoration и Ipopt wall time; полные медианы и ограничения — в
`block-tridiagonal-benchmark.json`.

Добавлен process-isolated end-to-end runner для clean HEAD, текущей Ipopt и
FATROP. Он циклически меняет порядок вариантов, pin-ит один CPU, фиксирует BLAS
threads, записывает реально загруженную `libipopt` и отклоняет различия
status/iterations/objective/feasibility. Одинаковые release-сборки и семь
samples дали:

| Case | HEAD, мс | Current, мс | FATROP, мс | Current/FATROP |
|---|---:|---:|---:|---:|
| DTOC1L 1000/5/10 | 126.516 | 126.354 | 15.931 | 7.93x |
| DTOC1L 6667/5/10 | 890.047 | 892.342 | 141.798 | 6.29x |
| DTOC1L 9091/1/10 | 1024.981 | 1025.264 | 175.362 | 5.85x |
| DTOC3 50 | 1.048 | 1.016 | 0.142 | 7.14x |
| DTOC3 5000 | 25.395 | 25.352 | 7.253 | 3.50x |
| DTOC3 30000 | 159.777 | 159.893 | 53.759 | 2.97x |
| DTOC6 5001 | 65.083 | 65.159 | 35.981 | 1.81x |
| DTOC6 50000 | 878.454 | 881.557 | 562.885 | 1.57x |

Локальный adapter reuse нейтрален на полном solve, тогда как stage structure
FATROP даёт существенный, но уменьшающийся на DTOC6 выигрыш. Первоначальные
`std::vector` scratch-поля в `TNLPAdapter` меняли layout и воспроизводимо
замедляли DTOC3 на 1.02–1.19%; перенос scratch в хвост существующего `jac_g_`
вернул ABI/layout и дал +0.60%/+0.08% на 31-sample DTOC3 confirmation вместо
исходных +1.02%/+1.19%.
Данные: `cutest-ipopt-comparison.json` и
`cutest-ipopt-dtoc3-confirmation.json`.

## Следующая реализация

1. Связать DTOC3 CUTEst stage provider с готовым equality-only assembler и
   candidate-first canary, сохранив fixed initial state через
   `make_constraint`; stable solver остаётся fallback.
2. Добавить capability-aware norm estimate без расширения всей старой matrix
   hierarchy.
3. Добавить structural-change `ReOptimizeNLP`; deliberate failure,
   limited-memory/inexact skips и fail-open mismatch уже покрыты.
4. Расширить assembler на bounds/path inequalities/slacks/restoration, а
   end-to-end gate — candidate-first вариантом и счётчиками factorization,
   backsolve, callbacks, allocations и fallback rate.

Критерий продвижения — не скорость Krylov kernel, а одинаковое качество
решения при измеримом снижении factorization/solve cost без роста failure rate.
