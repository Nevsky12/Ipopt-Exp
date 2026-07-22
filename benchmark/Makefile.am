# Copyright (C) 2026 Ipopt contributors.
# All Rights Reserved.
# This file is distributed under the Eclipse Public License.

noinst_PROGRAMS = \
  tnlp_adapter_eval_h_benchmark \
  tnlp_adapter_eval_grad_f_benchmark

tnlp_adapter_eval_h_benchmark_SOURCES = tnlp_adapter_eval_h.cpp
tnlp_adapter_eval_h_benchmark_LDADD = ../src/libipopt.la
tnlp_adapter_eval_grad_f_benchmark_SOURCES = tnlp_adapter_eval_grad_f.cpp
tnlp_adapter_eval_grad_f_benchmark_LDADD = ../src/libipopt.la

AM_CPPFLAGS = \
  -I$(srcdir)/../src/Common \
  -I$(srcdir)/../src/LinAlg \
  -I$(srcdir)/../src/LinAlg/TMatrices \
  -I$(srcdir)/../src/Algorithm \
  -I$(srcdir)/../src/Interfaces

BENCHMARK_ARGS ?= --iterations 20000000 --repetitions 7 --warmup 1000

.PHONY: benchmark

benchmark: $(noinst_PROGRAMS)
	./tnlp_adapter_eval_h_benchmark $(BENCHMARK_ARGS)
	./tnlp_adapter_eval_grad_f_benchmark $(BENCHMARK_ARGS)

EXTRA_DIST = README.md run_benchmarks.py run_cutest_comparison.py
