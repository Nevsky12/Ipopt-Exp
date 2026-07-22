# Ipopt: аудит ускорения, C++23 и AnyAny

Срез выполнен 2026-07-21 для `coin-or/Ipopt` на локальном коммите
`72a29c9aab198afa0dbb940339022a22c415a4eb` (`stable/3.14`). Главный вывод:
текущий Ipopt уже собирается GCC 13.3 с `-std=c++23`; поэтому полезная
модернизация — не механическая замена синтаксиса, а измеряемое устранение
аллокаций и лишних проходов, современная внешняя модель объектов, явное
владение состоянием solve-session и новые структурированные linear-solver
capabilities.

В рамках аудита уже сделаны два изменения горячего пути: `TNLPAdapter` теперь
повторно использует Hessian и objective-gradient workspaces при удалении
фиксированных переменных. Они устраняют по паре `new[]`/`delete[]` на каждый
`Eval_h` и `Eval_grad_f`; финальные изолированные вызовы ускорились примерно в
1.01 и 1.94 раза соответственно. Это локальные microbenchmarks, а не обещание такого
же ускорения полного solve. Также добавлены воспроизводимый benchmark gate и
отдельный C++23/AnyAny vertical slice для безопасных derivative products;
стабильные C и TNLP API он не меняет.

## 1. Охват и воспроизводимость

Исследованы:

- исходники и тесты локального Ipopt;
- все 46 публичных pull request, которые GitHub API вернул для официального
  репозитория: 35 merged, 4 open и 7 closed-unmerged;
- 311 публично доступных форков из 317, указанных счётчиком GitHub;
- все 3372 branch refs этих форков, а не только default branches;
- 81 уникальная нестандартная head-ревизия в 60 форках, включая сравнение
  patch-equivalence с текущей локальной веткой;
- `~/projects/fatrop-research`, включая актуальные benchmark results;
- существующую интеграцию AnyAny/Ipopt в
  `~/projects/rocketsystem-v2/rocketsystem/psopt`;
- официальный AnyAny и релиз v1.2.1, актуальный на дату среза.

Сырые снимки находятся в `upstream-inventory/`. Их контрольные суммы:

| Файл | SHA-256 |
|---|---|
| `inventory.json` | `0706c9cb4cf28efd802bc6187fb67999aeff84388b3d77a7532e026a69a8dbb2` |
| `fork-branches.json` | `7b248a5c720a7b4c2d95553d7816ac25a6704634da394309b73b9cadbcc8754a` |
| `fork-branch-diffs.json` | `8d782601e34064ecc01f6666cf8979a6faaa14f5ead6bc841fa250c8d7855266` |

Ограничения полноты:

- шесть форков из счётчика GitHub были удалены, приватны или иначе не
  возвращались публичным API;
- `keonigama/Ipopt` был виден API, но не отвечал на `git ls-remote`;
- две из 81 custom heads имеют несвязанную историю;
- 311 уникальных novel commit SHA — это кандидаты, а не 311 полезных
  улучшений: среди них есть generated files, локальные сборочные артефакты,
  старые порты и уже концептуально устаревшие эксперименты;
- knowledge graph ускорил навигацию, но имеет 827 dangling edges и сотни
  collapsed endpoints. Поэтому архитектурные выводы проверялись по исходникам,
  а graph centrality нигде не трактуется как runtime profile.

## 2. Архитектура и места воздействия

Основной путь задачи:

```text
TNLP callbacks
    -> TNLPAdapter (bounds, fixed variables, scaling, Jacobian/Hessian mapping)
    -> OrigIpoptNLP / IpoptData / IpoptCalculatedQuantities
    -> SearchDirectionCalculator / PDSystemSolver
    -> AugSystemSolver
    -> SymLinearSolver backend
```

`AlgorithmBuilder` связывает алгоритмические компоненты и выбирает linear
solver. `IpoptCalculatedQuantities` — центр зависимостей и кэширования
производных величин. `SmartPtr` — важная граница владения и ABI, но сам по себе
не доказанный performance hotspot.

Отсюда четыре разных класса работ, которые нельзя смешивать в один rewrite:

1. **Низкорисковые hot-path исправления.** Workspace reuse, меньше копирований,
   более крупные batch-вызовы и кэширование структуры.
2. **Session reuse.** Отделить неизменную структуру NLP/KKT от меняющихся
   значений, сделать повторный solve безопасным и проверяемым.
3. **API/ownership modernization.** C++23 façade, spans, capabilities, AnyAny,
   явные lifetime и error contracts.
4. **Алгоритмическая линейная алгебра.** Matrix-free products, iterative
   refinement, structured Riccati/condensing, batched multi-RHS и только затем
   GPU.

## 3. Официальные pull request

### Открытые PR

| PR | Что даёт | Оценка |
|---|---|---|
| [#857](https://github.com/coin-or/Ipopt/pull/857) | Atomics для process-global состояния | Нужный robustness signal. В новом ABI лучше по возможности убрать mutable globals и владеть состоянием на уровне application/session, а не только сделать гонку менее опасной. |
| [#849](https://github.com/coin-or/Ipopt/pull/849) | cuDSS GPU backend | Полезный прототип, но не готовая архитектура и не должен приниматься как есть. |
| [#773](https://github.com/coin-or/Ipopt/pull/773) | Flexible GMRES refinement и backward-error checks | Полезный research input, но не merge candidate: найден correctness-дефект обновления FGMRES candidate и несколько iteration-limit failures. Нужна новая минимальная реализация. |
| [#709](https://github.com/coin-or/Ipopt/pull/709) | Leopard linear solver | Старый incomplete integration с проблемами воспроизводимости и распространения binary-only dependency. Низкий приоритет. |

У #849 есть конкретные correctness/performance blockers:

- матрица всё ещё передаётся как symmetric, хотя в обсуждении уже отмечена
  проблема устойчивости LDL с 1x1 pivoting для saddle-point KKT; general/LDU
  path оставлен TODO;
- `ProvidesInertia()` заявляет `true`, но результат фактически подменяется
  `max(actual, requested)`, поэтому недостаточное число отрицательных
  собственных значений может не быть обнаружено;
- статус `cudssDataGet` при чтении inertia не проверяется, из-за чего при
  ошибке возможно использование неинициализированных данных;
- несколько RHS решаются последовательным циклом с отдельными solve и
  device-to-host copy. Это противоречит наблюдениям FATROP: repeated sweeps
  могут уничтожить выигрыш, если их не fuse/batch;
- host/device ownership пока не является first-class частью matrix API.

### Уже влитые robustness-направления

Особенно полезны как regression corpus и требования к новому дизайну:

- [#848](https://github.com/coin-or/Ipopt/pull/848) — SPRAL use-after-free;
- [#846](https://github.com/coin-or/Ipopt/pull/846) — MPI SIGSEGV;
- [#843](https://github.com/coin-or/Ipopt/pull/843) — видимость исключений;
- [#837](https://github.com/coin-or/Ipopt/pull/837) — fallback regularization;
- [#834](https://github.com/coin-or/Ipopt/pull/834) — uninitialized state/Valgrind;
- [#697](https://github.com/coin-or/Ipopt/pull/697) — propagation ошибки
  `eval_jac_g`;
- [#671](https://github.com/coin-or/Ipopt/pull/671) — memory reporting MA27;
- [#645](https://github.com/coin-or/Ipopt/pull/645) — null TNLP callbacks в C API;
- #590 и #570 — корректность index mapping.

Закрытые невлитые #856, #850, #812, #744, #628, #582 и #414 в основном
относятся к cleanup, documentation и build portability. Их следует использовать
как список compatibility cases, но они не являются ядром ускорения.

## 4. Форки и внешние модификации

Из 3372 branch refs 425 указывают на предков текущей stable-ветки, 2856 — на
другие официальные upstream branches, 91 ref образует 81 уникальную custom
head. Для 79 связанных custom heads найдено 869 повторяющихся novel patch
entries, или 311 уникальных commit SHA; ещё 43 уникальных commit SHA
patch-equivalent тому, что уже есть upstream.

Приоритетные находки:

| Ветка | Содержание | Решение |
|---|---|---|
| [`jgillis/Ipopt-1:jac_vp`](https://github.com/jgillis/Ipopt-1/tree/jac_vp) | `jac_vp`/`jac_vpt` callbacks через TNLP, adapters, restoration, calculated quantities и primal-dual residuals | Извлечь capability, не код: pure virtual ломает TNLP, forward path может использовать stale `x`, ошибки callback теряются, каждый product аллоцирует память. |
| [`jgillis/Ipopt:3.14.11.deepdive`](https://github.com/jgillis/Ipopt/tree/3.14.11.deepdive) | MUMPS debug/stat CSV и dumps RHS | Извлечь instrumentation ideas в benchmark/trace layer; не тащить runtime patch целиком. |
| [`jgillis/Ipopt:3.14.19.mod`](https://github.com/jgillis/Ipopt/tree/3.14.19.mod) | CasADi fixes: `nonzeros + 1` для MA27-like arrays и portable `RTLD_DEEPBIND` gating | Проверить санитайзерами и минимальным reproducer. Не применять лишний элемент или `RTLD_DEEPBIND` без доказанной причины. |
| [`dinesh286/Ipopt:reopt-fix`](https://github.com/dinesh286/Ipopt/tree/reopt-fix) | Reset `curr_` до `InitializeStructures` | Включить сценарий в session-reuse tests; сначала проверить, не закрыт ли invariant современным upstream. |
| [`subedisuman/Ipopt:cacheClear`](https://github.com/subedisuman/Ipopt/tree/cacheClear) | Cache invalidation после intermediate callback | Извлечь идею и написать regression test; сама ветка загрязнена build artifacts. |
| IDAES `restoration_mod` и ветки `dthierry` | Исследования L1 exact-penalty restoration | Архив полезных алгоритмических идей, но старый и незавершённый integration path. |

GPU-ветки BenVanDerMeer, MattBolitho и старые PETSc/MAGMA/SyLVER prototypes не
дают готового пути: часть имеет несвязанную историю, часть — stubs, часть
вендорит CUDA или основана на старом API. Они подтверждают, что нужен новый
memory/inertia contract, но не заменяют его.

CMake/package forks полезны для distribution и C++23 CI, однако их нельзя
считать runtime acceleration. Свежая дата push также не означает свежую
модификацию solver: например, некоторые новые default branches содержат только
документацию или merge старых build patches.

## 5. Уроки `fatrop-research`

Локальные результаты на AMD Ryzen 9 9950X3D, GCC 13.3 и Ipopt 3.14.20/MUMPS
показывают:

- FATROP даёт примерно 1.06x–10.24x end-to-end против generic Ipopt/CUTEst на
  исследованных структурированных задачах;
- bordered solver для глобальных параметров даёт 2.07x–2.24x против naive
  state augmentation и до 15.97x против Ipopt на крупном synthetic case;
- на normalized direct collocation/DTOC bordered вариант иногда проигрывает
  naive (отношение 0.565–0.909), потому что отдельные RHS sweeps повторно
  проходят те же stage data.

Практический вывод: направление `structured KKT` подтверждено, но интерфейс
должен выражать batch RHS и позволять fused/blocked Riccati passes. Просто
добавить bordered Schur complement недостаточно. Следующие алгоритмические
эксперименты:

1. fused/blocked multi-RHS Riccati;
2. stage-local static condensation;
3. inequalities/slacks и nonlinear integration;
4. cache/memory counters и сравнение с несколькими sparse backends;
5. общий benchmark protocol с Ipopt, чтобы не сравнивать только kernel time с
   end-to-end time.

Источник чисел: `~/projects/fatrop-research/fatrop_implicit/research/RESULTS.md`.

## 6. C++23 и AnyAny: предлагаемый дизайн

Локальный `psopt` уже демонстрирует рабочую форму:

- `AnyNlpProblem = aa::any_with<...>` выражает только необходимые operations;
- входы/выходы передаются spans вместо владения массивами;
- `IpoptSession` хранит application и TNLP и использует `ReOptimizeTNLP` только
  для структурно той же задачи;
- stale barrier-state отдельно ограждён, а изменение размерности приводит к
  cold solve.

Это хороший **внешний façade**, но не основание заменять AnyAny каждый
внутренний virtual interface. Type erasure улучшает композицию и dependency
boundaries; ускорение нужно доказывать отдельно для каждого call pattern.

Рекомендуемая последовательность:

1. Оставить стабильные C и существующие TNLP API как compatibility adapters.
2. Ввести экспериментальный C++23 `NlpProblem` с non-owning `std::span`,
   `std::expected`-подобным error contract и capability discovery для Hessian,
   Jacobian products и batched evaluations.
3. Дать задаче явный `StructureFingerprint`: размеры, sparsity pattern,
   derivative capabilities и backend-relevant layout.
4. Ввести `SolveSession`, которому принадлежат workspaces, symbolic
   factorization, caches и backend state. Reuse разрешать только при совпадении
   fingerprint и options, влияющих на структуру.
5. Выразить linear backends через узкий AnyAny registry/capabilities:
   inertia, symmetric/general, host/device memory, multi-RHS, precision и
   iterative refinement.
6. Только после профилирования решать, какие внутренние virtual hierarchies
   стоит заменить value/type-erased objects в новом ABI epoch.

В локальном дереве `psopt` закреплён AnyAny v1.1.1. На дату аудита официальный
latest — [v1.2.1](https://github.com/kelbon/AnyAny/releases/tag/v1.2.1), где в
том числе исправлен return type `poly_ptr::operator->`. Обновление зависимости
нужно тестировать отдельно; официальный проект не предоставляет Ipopt-specific
benchmark evidence.

Первый такой façade теперь реализован в `cxx23/`: AnyAny v1.2.1 зафиксирован
commit SHA и SHA-256 архива, `J*v`, `J^T*v` и `H*v` обнаруживаются независимо,
а для отсутствующей capability работает проверенный materialized sparse
fallback. Ошибки возвращаются через `std::expected`; размеры, обе sparsity
структуры и aliasing проверяются до пользовательского callback. Поверх façade
есть reusable saddle-point operator `[H + D_x, J^T; J, -D_y]` и полный
восьмиблочный `x/s/y_c/y_d/z_L/z_U/v_L/v_U` product с complementarity,
bound index maps и четырьмя регуляризациями. `PrimalDualSolveSession`
переиспользует Krylov workspace только при совпадении полного fingerprint,
но принимает свежие численные model/state данные на каждый solve.

## 7. Реализованное изменение

До изменения `TNLPAdapter::Eval_h` при наличии `h_idx_map_` на каждом вызове
выполнял:

```cpp
Number* full_h = new Number[nz_full_h_];
// eval_h + mapping
delete[] full_h;
```

Теперь существующая `jac_g_` allocation получает дополнительный tail размером
`max(n_full_x_, nz_full_h_)`, но только если `P_x_full_x_` действительно удаляет
fixed variables. `Eval_h`, `Eval_grad_f` и финальное восстановление bound
multipliers используют этот непересекающийся tail последовательно. Это
сохраняет прежнюю mapping/cache семантику, не создаёт общего workspace и, в
отличие от добавления нового поля, не меняет layout `TNLPAdapter`.

Microbenchmark: задача с тремя переменными, одна фиксирована, exact Hessian из
трёх nonzeros, 20 миллионов вызовов, семь независимых samples.

| Версия | Median ns/`Eval_h` | Samples, ns |
|---|---:|---|
| Clean HEAD | 10.2255 | 10.1119, 10.1148, 10.1306, 10.2255, 10.2744, 10.3760, 10.5024 |
| ABI-safe tail | 10.1094 | 10.0865, 10.0897, 10.0926, 10.1094, 10.1108, 10.1233, 10.1860 |

Результат: median latency `-1.14%`, или около `1.011x` throughput в
allocation-dominated microbenchmark.

Тот же allocation pattern был найден в exact `Eval_grad_f` при удалённых
fixed variables. Full-size gradient использует тот же conditional tail. На той
же задаче, CPU и протоколе median снизился с 15.3770 до 7.9143 нс/call:
`-48.53%`, или `1.94x` throughput.
Benchmark дополнительно проверяет callback count и итоговый mapped gradient.
Этот же buffer теперь используется при восстановлении bound multipliers для
fixed variables.

Проверки после полного rebuild:

- GCC 13.3 release microbenchmark и отдельные `-O2 -g` core builds;
- полный `make test`: double, genuine single и ASan+UBSan; AMPL, C++, C,
  Fortran, sIPOPT, EmptyNLP и GetCurr прошли;
- Java test пропущен конфигурацией;
- C++23 linked gates: source double, single, установленный 3.14.20 и
  ASan+UBSan — 7/7; standalone — 4/4;
- `git diff --check`: без ошибок.

Отдельно проверена ABI-оговорка. Первый вариант с двумя `std::vector` менял
layout экспортируемого C++ класса и дал solve-level regression 1.02–1.19% на
DTOC3. Он был отвергнут. Финальный patch не добавляет полей: header diff меняет
только комментарий к существующему `jac_g_`. ASan также поймал ожидаемый
648/600-byte new/delete mismatch, когда объектные файлы от двух промежуточных
layout были намеренно смешаны сборкой с disabled dependency tracking; полный
clean rebuild проходит. Это ещё один аргумент никогда не принимать
инкрементальный green build как ABI-проверку.

### Benchmark gate

Добавлен opt-in `--enable-benchmarks`, отдельная цель `make benchmark` и JSON
runner. Он фиксирует CPU affinity, принудительно оставляет BLAS/OpenMP по одному
потоку, проверяет число callback-вызовов и завершает CI с ошибкой при превышении
заданного regression threshold. Runner теперь умеет отдельно или вместе
запускать Hessian и gradient paths. В свежей C++23 `-O3 -march=native` сборке
финальный общий gate показал 10.1094 нс для `Eval_h` (`-1.14%` к принятому
baseline) и 7.9143 нс для `Eval_grad_f` (`-48.53%`); оба прошли порог 5%.

### C++23/AnyAny product slice

Standalone Release-сборка проходит GCC 13.3 с
`-Wall -Wextra -Wpedantic -Werror`; тесты дополнительно проходят ASan+UBSan и
leak detection. Покрыты native/fallback/partial capabilities, симметричный
off-diagonal Hessian fallback, `x`/multipliers/direction aliasing, callback
failure, invalid dimensions/sparsity, cache, capability/revision fingerprint и
KKT strong-output guarantee.

Полный primal-dual wrapper сверяет все восемь блоков с вручную рассчитанным
эталоном формул `PDFullSpaceSolver::ComputeResiduals`, поддерживает block и
flat in-place вызовы, включает layout в structural fingerprint и отвергает
неполные/повторяющиеся index maps до callback. Отдельный интеграционный тест
решает этот полный оператор через FGMRES. Session-тест повторно использует тот
же workspace для другого численного экземпляра с той же структурой, отклоняет
изменённую structural revision и сохраняет solution при callback failure.

Добавлен отдельный `LegacyCoordinateAdapter`, воспроизводящий композицию двух
старых слоёв без зависимости нового ABI от `Matrix`: `TNLPAdapter` сначала
расширяет редуцированный `x`, делит `g` на `[c,d]`, вычитает equality RHS и при
`MAKE_CONSTRAINT` добавляет identity rows; затем `OrigIpoptNLP` применяет
`x_s=D_x x`, `c_s=D_c c`, `d_s=D_d d`, `f_s=d_f f`. Поэтому adapter реализует
`J_s v=D_g J D_x^-1 v`, `J_s^T w=D_x^-1 J^T D_g w` и
`H_s v=D_x^-1 H(d_f obj_factor,D_g y)D_x^-1 v`. Удалённые fixed directions
заполняются нулями. Map/scaling проверяются до callback, sparse order совпадает
с `TNLPAdapter`, numeric RHS/fixed values не загрязняют structural fingerprint,
а source structure и maps входят в него.

Плотный 4x3 regression test одновременно покрывает удалённую fixed variable,
перестановку equality/inequality rows, добавленную fixed equality, отрицательный
objective scaling, разные `D_x/D_c/D_d`, materialized/native products, aliasing
и injected failures. Результат также сверен внутри полного saddle-point KKT.
Release `-Werror` и ASan+UBSan+LeakSanitizer проходят.

Добавлен отдельный prepared-direct boundary. AnyAny-erased backend сообщает
dimension, structural fingerprint и numeric revision, затем разделяет один
`factorize()` и произвольное число `solve_rhs()`. Подготовка отклоняет неверную
структуру, состояние или нулевую/stale numeric revision до факторизации;
каждый RHS решается в заранее выделенный scratch, а failure или nonfinite
результат не меняет caller output. `PrimalDualSolveSession` валидирует
prepared factor до Krylov шага. Точный 3x3 reference backend подтвердил две
FGMRES-задачи с одной факторизацией, включая in-place RHS; тот же тест проходит
Release `-Werror`, ASan, UBSan и LeakSanitizer.

Тонкий live bridge теперь также реализован. Он строится только после
`OrigIpoptNLP::InitializeStructures`, проверяет identity underlying
`TNLPAdapter`, копирует `ExpansionMatrix` positions, fixed values/equalities,
`c_rhs` и фактические post-`DetermineScaling` диагонали. Исходный TNLP хранится
через `SmartPtr`; C/FORTRAN sparsity нормализуется, legacy callbacks всегда
получают `new_x/new_lambda=true`, а `false` превращается в `std::expected`
failure со strong output. Ни один stable `Vector`/`Matrix` view не переживает
export.

Live regression покрывает все четыре fixed-variable treatments и на
`make_parameter` сквозно сравнивает objective, constraints, gradient, ordered
scaled Jacobian и Hessian с настоящим `OrigIpoptNLP`. Также проверены чужой
owner, FORTRAN_STYLE и callback failure. Связанные Release и ASan+UBSan+leak
наборы проходят 7/7.

Текущий `AugSystemSolver` теперь подключён к prepared-direct boundary через
opt-in `LegacyAugSystemDirectBackend`. Он вычисляет `sigma_x/sigma_s` из
полного complementarity state, сводит восемь RHS-блоков к существующим
`x/s/c/d`, один раз подготавливает factor нулевым `MultiSolve` и после каждого
backsolve восстанавливает `z_L/z_U/v_L/v_U`. Structural fingerprint и numeric
revision дополнены проверкой живых tags `W/J_c/J_d/sigma_x/sigma_s`; переданные
живые `IpCq`-диагонали также численно сверяются с complementarity state. Stable
failure, exception, wrong inertia, nonfinite result и stale tag сохраняют
caller output.
Perturbation, inertia retry, iterative refinement и `IncreaseQuality`
намеренно остаются во внешнем `PDFullSpaceSolver`.

Связанный 9-компонентный regression сравнивает direct direction и residual с
полным matrix-free оператором и получает одну FGMRES-итерацию с exact right
preconditioner. Два независимых RHS и прямой контрольный solve используют одну
факторизацию; тест также проходит через установленный `ipopt.pc`, то есть не
зависит от приватных source-tree matrix headers.

Добавлен безопасный runtime seam без изменения stable ABI/default solver:
явный `LegacyAlgorithmCanaryBuilder` переопределяет виртуальную
`PDSystemSolverFactory`, получает тот же `AugSystemSolver` и оборачивает
reference solver. Reference solve всегда выполняется первым и полностью
сохраняет perturbation/inertia/refinement/quality state machine. После успеха
canary читает уже принятые perturbations и живые matrix/vector tags, независимо
решает восьмиблочную систему и сверяет direction/residual, не изменяя legacy
result. Настоящий `IpoptApplication` проходит этот gate с exact Hessian, user
scaling и всеми четырьмя fixed-variable treatments; source-tree, установленный
Ipopt 3.14.20 и ASan+UBSan+leak дают 7/7.

Seam теперь охватывает и restoration без нового virtual slot или изменения
layout. `AlgorithmBuilder::BuildLineSearch` создаёт restoration solver через
существующую factory с префиксом `resto.`, default factory сохраняет
`AugRestoSystemSolver`, а невиртуальный accessor `PDFullSpaceSolver` позволяет
decorator удержать именно его cache. Для прямого `OrigIpoptNLP` используется
TNLP callback bridge; для compound `RestoIpoptNLP` снимается owning triplet
snapshot `W/J_c/J_d` через публичный `TripletHelper`, без долгоживущих stable
views. Все четыре fixed-variable treatments принудительно проходят
restoration; отдельно проходят `ReOptimizeNLP` и намеренный fail-open mismatch,
при котором reference result остаётся авторитетным.

Поверх shadow gate реализован opt-in
`LegacyAlgorithmCanaryMode::validated_replacement`. После полного reference
solve candidate обязан сойтись и пройти finite, direction-equivalence и true
residual limits. Все восемь блоков сначала конвертируются в отдельный stable
`IteratesVector`; только затем один финальный `Copy` заменяет result. Любая
ошибка оставляет reference direction без изменений. Отдельные счётчики
фиксируют requests, commits, failures и restoration commits. Normal,
`ReOptimizeNLP` и forced-restoration regressions действительно выполняют
commit, а намеренный mismatch проверяет fallback. Это проверка безопасной
границы замены, не ускорение: reference factorization всё ещё оплачивается
первой.

Следующий opt-in слой теперь также реализован:
`LegacyAlgorithmCanaryMode::candidate_first` принимает shared move-only
AnyAny backend. Backend синхронно получает current full-KKT state/RHS и сам
владеет perturbation sequence, inertia retries, refinement и quality
escalation; наружу возвращает unscaled direction, четыре принятые
regularization, exact inertia certificate и work counters. Wrapper независимо
проверяет размер/finite, неотрицательность и representability regularization,
точное число negative eigenvalues и true residual. Только затем detached
`IteratesVector` одним `Copy` публикует направление, после чего обновляются
`PDPert`/`info_regu_x`. При любой ошибке, неверной inertia, плохом residual или
commit failure полный stable solver вызывается ровно один раз; inexact,
limited-memory/refinement и nonzero-beta вызовы сразу делегируются ему.

Happy path действительно не вызывает `PDFullSpaceSolver::Solve` и потому не
факторизует stable `AugSystemSolver`. Runtime-canary regression по-прежнему
использует маленький dense backend-double: все четыре fixed-variable treatments, normal,
`ReOptimizeNLP` и forced restoration выполняют candidate commit; отдельные
инъекции backend failure, неверной inertia и плохого residual дают по одному
stable fallback. Factorization/backsolve accounting сверяется с backend и
учитывает также отклонённые попытки. Это настоящий performance seam, но ещё не
end-to-end performance result: equality-only OCP assembler уже реализован и
проверен независимо, но live TNLP/CUTEst stage provider и его binding к canary
пока не подключены, а bounds/inequalities/restoration требуют полной
восьмиблочной версии. Поэтому новое CUTEst wall-time число не заявляется.
Точные counts factorization,
backsolve, backend KKT/derivative-product requests и validation applications
сохранены в `candidate-first-canary-counts.json`.

За seam теперь добавлены два независимых production-oriented слоя, которые не
вызывают stable solver:

- `SymmetricBlockTridiagonalSolver` принимает variable-size stage blocks,
  выполняет block-`LDL^T`, факторизует каждый Schur block cyclic Jacobi и
  переиспользует factor для scalar или fused RHS;
- inertia принимается только если явные нормы `S-Q Lambda Q^T` и
  `Q^T Q-I` доказывают отделение спектра от нуля; по закону инерции Сильвестра
  знаки Schur blocks суммируются в сертификат исходной матрицы;
- topology и максимальный RHS batch фиксируются конструктором, поэтому happy
  factor/solve/refinement paths не меняют capacity и не удерживают caller
  views;
- `StageStructuredCandidateBackend<Assembler>` проверяет fingerprint,
  требует точный inertia-вклад всех исключённых congruence blocks,
  реконструирует полный восьмиблочный direction и сам увеличивает primal или
  dual regularization в зависимости от направления inertia mismatch;
- singular/uncertified factor, refinement limit/increase, неверный condensed
  dimension, несертифицированный inertia-вклад и nonfinite reconstruction
  являются явными отказами, после которых внешний candidate-first gate всё
  ещё может ровно один раз вызвать stable fallback.
- `StageNlpTopology` валидирует полные stage dimensions, canonical-to-generic
  permutations и packed offsets; compile-time `StageDerivativeProvider`
  заполняет owned Hessian/dynamics/path buffers и переиспользуется только при
  совпадающей numeric revision;
- `EqualityStageKktAssembler` собирает без condensation полный KKT из блоков
  `[u_k,x_k,path multiplier,incoming dynamics multiplier]`, точно переставляет
  RHS/direction и намеренно отвергает bounds, slacks, inequalities,
  restoration auxiliaries и неединичный next-state Jacobian.

CUTEst probe теперь печатает имена constraints и отмечает fixed variables.
Для DTOC3 `N=50` он подтвердил 50 стадий, 49 controls `X1...X49`, состояния
`Y1...Y50` размерности 2, 98 линейных dynamics rows `TT1...TT49` и ровно два
fixed initial-state variables `Y1,1/Y1,2`; после stage ordering максимальный
row span равен 6. Выбранный следующий binding использует
`fixed_variable_treatment=make_constraint`: начальные states остаются в
primal, а две identity rows становятся stage-0 path equalities, что попадает
в уже реализованный equality-only контракт.

Это generic block-elimination reference backend, а не копия FATROP null-space
Riccati. Synthetic regression охватывает variable blocks, indefinite systems,
точную inertia, singular/near-singular rejection, congruent condensation,
retry, factor-once/solve-many и transactional failures. Расширенный набор с
установленным Ipopt 3.14.20 проходит Release и ASan+UBSan+leak 11/11.

Предшествующий legacy gate также собран с настоящим
`--with-precision=single` и single SMUMPS.
Допуски FGMRES/direction/residual выводятся из epsilon стабильного
`Ipopt::Number`, а CMake читает generated `config_ipopt.h`, чтобы не получить
случайный double ABI. Этот source double/single набор проходит 7/7; текущий
установленный-Ipopt набор с четырьмя structured tests проходит 11/11 в Release
и ASan+UBSan+leak. Старый установленный core не имеет нового
restoration factory seam, поэтому совместим и проверяет main path, но не может
отдать наружу свой внутренний restoration solver.

При полном single `make test` обнаружен независимый false positive в
`GetCurr`: Lagrangian gradient после сокращения членов порядка `2e5`
сравнивался с допуском, масштабированным только малым результатом. Чистый HEAD
воспроизвёл сбой бит-в-бит. Проверка теперь дополнительно учитывает roundoff от
суммы модулей членов; полный double и single regression проходят.

Поверх operator layer также реализован corrected restarted FGMRES: workspace
выделяется конструктором, changing preconditioner получает глобальный номер
итерации, restart base не мутирует, candidate собирается по всем текущим
коэффициентам, а перед commit проверяется true residual. Callback error
оставляет caller solution неизменным. Формулы сверены с
[Saad (1993)](https://doi.org/10.1137/0914028) и
[PETSc FGMRES](https://petsc.org/release/src/ksp/ksp/impls/gmres/fgmres/fgmres.c.html).

На минимальном 4x4 native `J*v` AnyAny boundary имеет median 6.729 нс против
2.512 нс прямого adapter-вызова: +4.217 нс, или 2.68x для намеренно крошечного
kernel. Поэтому type erasure уместен на крупной operator boundary, но не в
поэлементных циклах. Полный разбор веток, алгоритмических дефектов и следующего
дизайна находится в `MATRIX_FREE_AUDIT.md`.

В 8x8/8-step FGMRES microbenchmark повторное использование constructor-owned
workspace дало 2937.7 против 3126.3 нс/solve при реконструкции solver:
`-188.6` нс, или 1.064x. Это подтверждает выбранный lifecycle на малом kernel,
но не заменяет full Ipopt benchmark.

Новый synthetic structured benchmark измеряет уже готовые stage blocks
размера 8 и восемь RHS, пять Release-прогонов на pinned CPU 4. Медианы:

| Stages | Dimension | Factor, мкс | Factor, мкс/stage | 8 sequential RHS, мкс | Fused 8 RHS, мкс | Fused speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 128 | 25.263 | 1.579 | 14.378 | 9.797 | 1.468x |
| 32 | 256 | 50.110 | 1.566 | 28.343 | 20.247 | 1.400x |
| 64 | 512 | 100.800 | 1.575 | 56.790 | 41.380 | 1.372x |
| 128 | 1024 | 205.971 | 1.609 | 113.025 | 84.380 | 1.339x |
| 256 | 2048 | 409.553 | 1.600 | 232.196 | 164.246 | 1.414x |

Factor cost с propagated inverse-error certification остаётся в диапазоне
1.566--1.609 мкс/stage; contiguous-RHS microkernel даёт 1.339x--1.468x против
восьми отдельных horizon sweeps.
Checksums совпали во всех повторах. Это только linear-algebra kernel с уже
известной ordering: derivative assembly, globalization, restoration и Ipopt
wall time сюда не входят. Сырые условия, медианы и ограничения сохранены в
`block-tridiagonal-benchmark.json`.

End-to-end gate теперь запускает отдельные процессы для clean HEAD, текущей
`libipopt` и FATROP, циклически меняет их порядок, pin-ит CPU 4 и ограничивает
BLAS одним потоком. Обе Ipopt-библиотеки собраны GCC с одинаковыми
`-O3 -DNDEBUG -march=native` и зависимостями. Семь измерений после одного
warmup дали:

| CUTEst case | HEAD, мс | Current, мс | Current, % | FATROP, мс | Current/FATROP |
|---|---:|---:|---:|---:|---:|
| DTOC1L N=1000, X=5, Y=10 | 126.516 | 126.354 | -0.13 | 15.931 | 7.93x |
| DTOC1L N=6667, X=5, Y=10 | 890.047 | 892.342 | +0.26 | 141.798 | 6.29x |
| DTOC1L N=9091, X=1, Y=10 | 1024.981 | 1025.264 | +0.03 | 175.362 | 5.85x |
| DTOC3 N=50 | 1.048 | 1.016 | -3.05 | 0.142 | 7.14x |
| DTOC3 N=5000 | 25.395 | 25.352 | -0.17 | 7.253 | 3.50x |
| DTOC3 N=30000 | 159.777 | 159.893 | +0.07 | 53.759 | 2.97x |
| DTOC6 N=5001 | 65.083 | 65.159 | +0.12 | 35.981 | 1.81x |
| DTOC6 N=50000 | 878.454 | 881.557 | +0.35 | 562.885 | 1.57x |

Статусы, iterations внутри каждого backend, objectives и feasibility прошли
gate; максимальная violation была `1.90e-11`. У текущего Ipopt нет
последовательного solve-level выигрыша: factorization-dominated задачи скрывают
локальную экономию adapter allocations. Первоначальный вариант scratch с
двумя `std::vector` в `TNLPAdapter` дал воспроизводимый regression 1.02–1.19%
на DTOC3 и менял C++ layout класса. Scratch перенесён в хвост существующей
`jac_g_` allocation только для удалённых fixed variables. После этого
31-повторное подтверждение дало +0.60% на DTOC3/5000 и +0.08% на
DTOC3/30000, вместо исходных +1.02%/+1.19%. Сырые данные находятся в `cutest-ipopt-comparison.json` и
`cutest-ipopt-dtoc3-confirmation.json`.

## 8. Приоритетный roadmap

### P0 — измерительная база

- Поддерживать добавленный JSON gate CUTEst + `fatrop-research` как
  обязательную проверку с одинаковыми compiler flags, solver options и
  correctness tolerances.
- Разделять setup, callbacks, symbolic factorization, numeric factorization,
  triangular solves и globalization.
- Добавить allocations, peak RSS, cache misses, branch misses и multi-RHS
  counters.
- Запускать microbenchmarks только вместе с end-to-end solve benchmarks.

### P1 — безопасные локальные ускорения и session reuse

- Сохранять выбранный ABI-safe conditional-tail design Hessian/gradient
  workspace и проверять его только через clean rebuild.
- Профилировать оставшиеся per-iteration allocations/copies в TNLPAdapter и
  calculated quantities.
- Расширить уже проходящий `ReOptimizeNLP` regression на фактическое изменение
  структуры/ревизии и cache invalidation cases из `reopt-fix`.
- Перенести process-global mutable state в application/session там, где это
  возможно; atomics использовать как compatibility fix, а не конечную модель.

### P2 — C++23/AnyAny façade

- Расширять opt-in `NlpProblem`, уже добавленный рядом с TNLP без изменения C
  ABI: bounds и batched evaluations; Hessian products уже входят в façade.
- Fingerprint-gated `SolveSession`, numeric-preconditioner lifecycle, live
  TNLP/scaling adapter и adapter существующего `AugSystemSolver` уже
  реализованы; shadow и validated-replacement modes уже проходят equivalence
  gate. AnyAny candidate-first registry/contract и reference-free opt-in mode
  также реализованы; generic stage backend готов, следующим подключить native
  CUTEst stage provider к готовому equality-only assembler, затем полную
  восьмиблочную stage layout.
- Явно моделировать optional capabilities и lifetimes; не обещать ускорение от
  type erasure без benchmark.

### P3 — matrix-free robustness

- Не сливать `jac_vp` и PR #773 напрямую: их regression cases уже перенесены в
  standalone product/KKT/FGMRES tests; полный primal-dual product, live bridge,
  prepared adapter и opt-in runtime canary для main/restoration уже есть.
  Guarded commit и reference-free contract уже реализованы; следующий риск —
  корректная production perturbation/inertia/retry policy конкретного backend.
- Добавить limited-memory/inexact skip и callback-failure regressions; double,
  single и fail-open mismatch уже проверены.
- Сравнить factorization avoidance против числа Krylov iterations и callback
  cost на реальных задачах.

### P4 — structured KKT

- Generic variable-block factor state, fused multi-RHS, certified congruent
  condensation contract и candidate retry policy уже реализованы.
- Equality-only full-KKT stage assembler, topology/permutation validation и
  numeric-revision derivative cache уже реализованы.
- Следующие цели — live CUTEst/canary binding, полный
  bounds/inequalities/restoration assembler, FATROP-подобная null-space
  Riccati вместо reference Jacobi blocks и stage-local static condensation.
- Поддерживать generic sparse fallback и одинаковую проверку residual/inertia.

### P5 — GPU

- Сначала определить symmetric/general contract, pivoting и честную inertia
  verification.
- Ввести явные host/device buffers и batched RHS.
- Сравнивать end-to-end, включая transfers и repeated solves; PR #849 считать
  прототипом для извлечения кода, а не готовым merge target.

## 9. Критерии отбраковки

Изменение не проходит в основную линию модернизации, если:

- оно показывает только kernel speedup без solve-level результата и проверки
  качества решения;
- ускорение требует последовательных повторных проходов по одним данным там,
  где доступен batch/fusion;
- backend заявляет inertia, но не может надёжно сообщить ошибку или фактический
  результат;
- session reuse не проверяет неизменность структуры;
- замена virtual на AnyAny обосновывается только теорией;
- нарушается C ABI без compatibility adapter либо C++ class layout без явно
  объявленного ABI epoch;
- generated artifacts и локальные build fixes форка принимаются как solver
  modification без извлечения минимального патча и regression test.

## 10. Следующий конкретный эксперимент

Наилучший следующий vertical slice — связать DTOC3 CUTEst provider с уже
готовыми `StageNlpTopology`, `EqualityStageKktAssembler` и
`StageStructuredCandidateBackend`, сохранив фиксированные `Y1` через
`make_constraint`, и добавить lazy candidate construction в canary. После
этого расширить assembler до path inequalities, bounds/slacks и restoration,
а reference Jacobi factor внутри stage blocks заменить на профильный dense
kernel/null-space Riccati. Измерять нужно assembly, factorization,
backsolve, derivative callbacks, fallback rate и wall time рядом со stable
Ipopt и `fatrop-research`. Эксперимент считается успешным только при неизменных
status/iterations/objective/feasibility и снижении end-to-end времени без роста
fallback rate.
