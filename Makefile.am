# Copyright (C) 2004, 2008 International Business Machines and others.
# All Rights Reserved.
# This file is distributed under the Eclipse Public License.
#
# Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

SUBDIRS = src

if BUILD_SIPOPT
  SUBDIRS += contrib/sIPOPT
endif

if BUILD_BENCHMARKS
  SUBDIRS += benchmark
endif

DIST_SUBDIRS = src contrib/sIPOPT benchmark

doc_DATA = README.md AUTHORS LICENSE ChangeLog.md

EXTRA_DIST = \
  cxx23/CMakeLists.txt \
  cxx23/README.md \
  cxx23/benchmarks/anyany_jacobian_product_benchmark.cpp \
  cxx23/benchmarks/block_tridiagonal_solver_benchmark.cpp \
  cxx23/benchmarks/fgmres_reuse_benchmark.cpp \
  cxx23/include/ipopt/cxx23/any_nlp_problem.hpp \
  cxx23/include/ipopt/cxx23/block_tridiagonal_solver.hpp \
  cxx23/include/ipopt/cxx23/bordered_block_tridiagonal_solver.hpp \
  cxx23/include/ipopt/cxx23/bordered_stage_structured_nlp.hpp \
  cxx23/include/ipopt/cxx23/bordered_stage_structured_candidate_backend.hpp \
  cxx23/include/ipopt/cxx23/candidate_first_backend.hpp \
  cxx23/include/ipopt/cxx23/direct_preconditioner.hpp \
  cxx23/include/ipopt/cxx23/equality_stage_kkt_assembler.hpp \
  cxx23/include/ipopt/cxx23/fgmres.hpp \
  cxx23/include/ipopt/cxx23/kkt_operator.hpp \
  cxx23/include/ipopt/cxx23/legacy_algorithm_canary.hpp \
  cxx23/include/ipopt/cxx23/legacy_aug_system_backend.hpp \
  cxx23/include/ipopt/cxx23/legacy_coordinate_adapter.hpp \
  cxx23/include/ipopt/cxx23/legacy_ipopt_bridge.hpp \
  cxx23/include/ipopt/cxx23/legacy_matrix_snapshot.hpp \
  cxx23/include/ipopt/cxx23/primal_dual_kkt_operator.hpp \
  cxx23/include/ipopt/cxx23/primal_dual_bordered_stage_kkt_assembler.hpp \
  cxx23/include/ipopt/cxx23/primal_dual_stage_kkt_assembler.hpp \
  cxx23/include/ipopt/cxx23/restoration_stage_nlp.hpp \
  cxx23/include/ipopt/cxx23/solve_session.hpp \
  cxx23/include/ipopt/cxx23/sparse_bordered_stage_derivative_provider.hpp \
  cxx23/include/ipopt/cxx23/sparse_stage_derivative_provider.hpp \
  cxx23/include/ipopt/cxx23/stage_structured_candidate_backend.hpp \
  cxx23/include/ipopt/cxx23/stage_structured_nlp.hpp \
  cxx23/tests/block_tridiagonal_solver_test.cpp \
  cxx23/tests/bordered_block_tridiagonal_solver_test.cpp \
  cxx23/tests/direct_preconditioner_test.cpp \
  cxx23/tests/equality_stage_kkt_assembler_test.cpp \
  cxx23/tests/fgmres_test.cpp \
  cxx23/tests/any_nlp_problem_test.cpp \
  cxx23/tests/legacy_algorithm_canary_test.cpp \
  cxx23/tests/legacy_aug_system_backend_test.cpp \
  cxx23/tests/legacy_ipopt_bridge_test.cpp \
  cxx23/tests/legacy_primal_dual_stage_canary_test.cpp \
  cxx23/tests/legacy_coordinate_adapter_test.cpp \
  cxx23/tests/primal_dual_stage_kkt_assembler_test.cpp \
  cxx23/tests/primal_dual_bordered_stage_kkt_assembler_test.cpp \
  cxx23/tests/pdfullspace_fgmres_refinement_test.cpp \
  cxx23/tests/stage_structured_candidate_backend_test.cpp \
  cxx23/tests/stage_structured_nlp_test.cpp

.PHONY: test benchmark doc javadoc astyle cppcheck cppcheck-config

test: all
	cd test; $(MAKE) test

if BUILD_BENCHMARKS
benchmark: all
	cd benchmark; $(MAKE) benchmark
else
benchmark:
	@echo "Benchmarks are disabled; configure with --enable-benchmarks." >&2
	@exit 1
endif

doc :
	cd doc && doxygen
	sed -i -e 's@<p><a class="anchor" id="OPT_\(.*\)"></a>@<p><a class="anchor" id="OPT_\1"></a><a href="#OPT_\1">\&#x25B8;\&nbsp;</a>@g' doc/html/OPTIONS.html

if BUILD_JAVA
javadoc :
	$(JAVADOC) -d javadoc -windowtitle "JIpopt API documentation" $(srcdir)/src/Interfaces/Ipopt.java
endif

clean-doc:
	cd doc && rm -rf html *.log *.tag
	rm -rf javadoc

clean-local : clean-doc

astyle:
	cd $(srcdir) && astyle --mode=c -A1 --indent=spaces=3 --indent-switches --min-conditional-indent=1 --convert-tabs --align-pointer=type --pad-oper --add-braces -n -r "*.hpp" "*.h"  "*.cpp" "*.c" --exclude=hsl_ma77d.h --exclude=hsl_ma77s.h --exclude=hsl_ma86d.h --exclude=hsl_ma86s.h --exclude=hsl_ma97d.h --exclude=hsl_ma97s.h --exclude=hsl_mc68i.h
	cd $(srcdir) && astyle --mode=java -A1 --indent=spaces=3 --indent-switches --min-conditional-indent=1 --convert-tabs --align-pointer=type --pad-oper --add-braces -n -r "*.java"

cppcheck:
	cppcheck -q -j4 --enable=all --suppress=uninitMemberVar --suppress=noExplicitConstructor --inline-suppr \
	  --suppress=cstyleCast:src/Apps/AmplSolver/AmplTNLP.cpp --suppress=constArgument:examples/hs071_c/hs071_c.c \
	  --addon=threadsafety.py --std=c++03 --std=c99 --error-exitcode=2 --relative-paths=$(srcdir) \
	  $(IPOPTLIB_CFLAGS) $(IPOPTAMPLINTERFACELIB_CFLAGS) $(addprefix -I,$(shell find $(srcdir)/src -type d)) -Isrc/Common \
	  -DHAVE_CONFIG_H $(srcdir)/src $(srcdir)/test $(srcdir)/examples

cppcheck-config:
	cppcheck -q --check-config --suppress=missingIncludeSystem --std=c++03 --std=c99 --error-exitcode=2 \
	  $(IPOPTLIB_CFLAGS) $(IPOPTAMPLINTERFACELIB_CFLAGS) $(addprefix -I,$(shell find $(srcdir)/src -type d)) -Isrc/Common \
	  -DHAVE_CONFIG_H $(srcdir)/src $(srcdir)/test $(srcdir)/examples
