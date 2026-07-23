cmake_minimum_required(VERSION 3.24)

project(ipopt_cxx23_prototype LANGUAGES CXX)

include(FetchContent)

option(IPOPT_CXX23_BUILD_TESTS "Build C++23 prototype tests" ON)
option(IPOPT_CXX23_BUILD_BENCHMARKS "Build C++23 prototype benchmarks" OFF)
option(
  IPOPT_CXX23_BUILD_LEGACY_BRIDGE_TESTS
  "Build C++23 compatibility tests against an installed or local Ipopt"
  OFF
)
set(
  IPOPT_CXX23_LEGACY_IPOPT_INCLUDE_DIRECTORIES
  ""
  CACHE STRING
  "Optional semicolon-separated include directories for a local Ipopt build"
)
set(
  IPOPT_CXX23_LEGACY_IPOPT_LIBRARY
  ""
  CACHE FILEPATH
  "Optional Ipopt library for the legacy bridge test"
)

FetchContent_Declare(
  anyany
  URL
    https://github.com/kelbon/AnyAny/archive/560301c278e46bb527e49b1739dcb841035a9932.tar.gz
  URL_HASH
    SHA256=c0a3d8a3e84a46fa506cc9604a3e31b971d9eeb32565f53f3a14e97243fb3eda
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(AA_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(anyany)

add_library(ipopt_cxx23 INTERFACE)
add_library(Ipopt::cxx23 ALIAS ipopt_cxx23)
target_compile_features(ipopt_cxx23 INTERFACE cxx_std_23)
target_include_directories(
  ipopt_cxx23
  INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
target_link_libraries(ipopt_cxx23 INTERFACE anyanylib)

function(ipopt_cxx23_enable_strict_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /WX)
  else()
    target_compile_options(
      ${target}
      PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
  endif()
endfunction()

if(IPOPT_CXX23_BUILD_TESTS)
  enable_testing()
  add_executable(
    ipopt_cxx23_any_nlp_problem_test
    tests/any_nlp_problem_test.cpp
  )
  target_link_libraries(ipopt_cxx23_any_nlp_problem_test PRIVATE Ipopt::cxx23)
  ipopt_cxx23_enable_strict_warnings(ipopt_cxx23_any_nlp_problem_test)
  add_test(
    NAME ipopt_cxx23_any_nlp_problem
    COMMAND ipopt_cxx23_any_nlp_problem_test
  )

  add_executable(
    ipopt_cxx23_legacy_coordinate_adapter_test
    tests/legacy_coordinate_adapter_test.cpp
  )
  target_link_libraries(
    ipopt_cxx23_legacy_coordinate_adapter_test
    PRIVATE Ipopt::cxx23
  )
  ipopt_cxx23_enable_strict_warnings(
    ipopt_cxx23_legacy_coordinate_adapter_test
  )
  add_test(
    NAME ipopt_cxx23_legacy_coordinate_adapter
    COMMAND ipopt_cxx23_legacy_coordinate_adapter_test
  )

  add_executable(
    ipopt_cxx23_direct_preconditioner_test
    tests/direct_preconditioner_test.cpp
  )
  target_link_libraries(
    ipopt_cxx23_direct_preconditioner_test
    PRIVATE Ipopt::cxx23
  )
  ipopt_cxx23_enable_strict_warnings(
    ipopt_cxx23_direct_preconditioner_test
  )
  add_test(
    NAME ipopt_cxx23_direct_preconditioner
    COMMAND ipopt_cxx23_direct_preconditioner_test
  )

  add_executable(
    ipopt_cxx23_block_tridiagonal_solver_test
    tests/block_tridiagonal_solver_test.cpp
  )
  target_link_libraries(
    ipopt_cxx23_block_tridiagonal_solver_test
    PRIVATE Ipopt::cxx23
  )
  ipopt_cxx23_enable_strict_warnings(
    ipopt_cxx23_block_tridiagonal_solver_test
  )
  add_test(
    NAME ipopt_cxx23_block_tridiagonal_solver
    COMMAND ipopt_cxx23_block_tridiagonal_solver_test
  )

  add_executable(
    ipopt_cxx23_bordered_block_tridiagonal_solver_test
    tests/bordered_block_tridiagonal_solver_test.cpp
  )
  target_link_libraries(
    ipopt_cxx23_bordered_block_tridiagonal_solver_test
    PRIVATE Ipopt::cxx23
  )
  ipopt_cxx23_enable_strict_warnings(
    ipopt_cxx23_bordered_block_tridiagonal_solver_test
  )
  add_test(
    NAME ipopt_cxx23_bordered_block_tridiagonal_solver
    COMMAND ipopt_cxx23_bordered_block_tridiagonal_solver_test
  )

  add_executable(
    ipopt_cxx23_stage_structured_candidate_backend_test
    tests/stage_structured_candidate_backend_test.cpp
  )
  target_link_libraries(
    ipopt_cxx23_stage_structured_candidate_backend_test
    PRIVATE Ipopt::cxx23
  )
  ipopt_cxx23_enable_strict_warnings(
    ipopt_cxx23_stage_structured_candidate_backend_test
  )
  add_test(
    NAME ipopt_cxx23_stage_structured_candidate_backend
    COMMAND ipopt_cxx23_stage_structured_candidate_backend_test
  )

  add_executable(
    ipopt_cxx23_stage_structured_nlp_test
    tests/stage_structured_nlp_test.cpp
  )
  target_link_libraries(
    ipopt_cxx23_stage_structured_nlp_test
    PRIVATE Ipopt::cxx23
  )
  ipopt_cxx23_enable_strict_warnings(
    ipopt_cxx23_stage_structured_nlp_test
  )
  add_test(
    NAME ipopt_cxx23_stage_structured_nlp
    COMMAND ipopt_cxx23_stage_structured_nlp_test
  )

  add_executable(
    ipopt_cxx23_equality_stage_kkt_assembler_test
    tests/equality_stage_kkt_assembler_test.cpp
  )
  target_link_libraries(
    ipopt_cxx23_equality_stage_kkt_assembler_test
    PRIVATE Ipopt::cxx23
  )
  ipopt_cxx23_enable_strict_warnings(
    ipopt_cxx23_equality_stage_kkt_assembler_test
  )
  add_test(
    NAME ipopt_cxx23_equality_stage_kkt_assembler
    COMMAND ipopt_cxx23_equality_stage_kkt_assembler_test
  )

  add_executable(
    ipopt_cxx23_primal_dual_stage_kkt_assembler_test
    tests/primal_dual_stage_kkt_assembler_test.cpp
  )
  target_link_libraries(
    ipopt_cxx23_primal_dual_stage_kkt_assembler_test
    PRIVATE Ipopt::cxx23
  )
  ipopt_cxx23_enable_strict_warnings(
    ipopt_cxx23_primal_dual_stage_kkt_assembler_test
  )
  add_test(
    NAME ipopt_cxx23_primal_dual_stage_kkt_assembler
    COMMAND ipopt_cxx23_primal_dual_stage_kkt_assembler_test
  )

  add_executable(
    ipopt_cxx23_primal_dual_bordered_stage_kkt_assembler_test
    tests/primal_dual_bordered_stage_kkt_assembler_test.cpp
  )
  target_link_libraries(
    ipopt_cxx23_primal_dual_bordered_stage_kkt_assembler_test
    PRIVATE Ipopt::cxx23
  )
  ipopt_cxx23_enable_strict_warnings(
    ipopt_cxx23_primal_dual_bordered_stage_kkt_assembler_test
  )
  add_test(
    NAME ipopt_cxx23_primal_dual_bordered_stage_kkt_assembler
    COMMAND ipopt_cxx23_primal_dual_bordered_stage_kkt_assembler_test
  )

  add_executable(
    ipopt_cxx23_fgmres_test
    tests/fgmres_test.cpp
  )
  target_link_libraries(ipopt_cxx23_fgmres_test PRIVATE Ipopt::cxx23)
  ipopt_cxx23_enable_strict_warnings(ipopt_cxx23_fgmres_test)
  add_test(
    NAME ipopt_cxx23_fgmres
    COMMAND ipopt_cxx23_fgmres_test
  )

  if(IPOPT_CXX23_BUILD_LEGACY_BRIDGE_TESTS)
    add_library(ipopt_cxx23_legacy_ipopt INTERFACE)
    if(
      IPOPT_CXX23_LEGACY_IPOPT_INCLUDE_DIRECTORIES
      OR IPOPT_CXX23_LEGACY_IPOPT_LIBRARY
    )
      if(
        NOT IPOPT_CXX23_LEGACY_IPOPT_INCLUDE_DIRECTORIES
        OR NOT IPOPT_CXX23_LEGACY_IPOPT_LIBRARY
      )
        message(
          FATAL_ERROR
          "Set both local legacy-Ipopt include directories and library"
        )
      endif()
      target_include_directories(
        ipopt_cxx23_legacy_ipopt
        INTERFACE ${IPOPT_CXX23_LEGACY_IPOPT_INCLUDE_DIRECTORIES}
      )
      set(ipopt_cxx23_legacy_config_header "")
      foreach(
        ipopt_cxx23_legacy_include_directory
        IN LISTS IPOPT_CXX23_LEGACY_IPOPT_INCLUDE_DIRECTORIES
      )
        if(
          EXISTS
            "${ipopt_cxx23_legacy_include_directory}/config_ipopt.h"
        )
          set(
            ipopt_cxx23_legacy_config_header
            "${ipopt_cxx23_legacy_include_directory}/config_ipopt.h"
          )
          break()
        endif()
      endforeach()
      if(ipopt_cxx23_legacy_config_header)
        file(
          STRINGS "${ipopt_cxx23_legacy_config_header}"
          ipopt_cxx23_legacy_single_definition
          REGEX "^#define[ \t]+IPOPT_SINGLE[ \t]+1"
          LIMIT_COUNT 1
        )
        file(
          STRINGS "${ipopt_cxx23_legacy_config_header}"
          ipopt_cxx23_legacy_int64_definition
          REGEX "^#define[ \t]+IPOPT_INT64[ \t]+1"
          LIMIT_COUNT 1
        )
        if(ipopt_cxx23_legacy_single_definition)
          target_compile_definitions(
            ipopt_cxx23_legacy_ipopt INTERFACE IPOPT_SINGLE=1
          )
        endif()
        if(ipopt_cxx23_legacy_int64_definition)
          target_compile_definitions(
            ipopt_cxx23_legacy_ipopt INTERFACE IPOPT_INT64=1
          )
        endif()
      endif()
      target_link_libraries(
        ipopt_cxx23_legacy_ipopt
        INTERFACE "${IPOPT_CXX23_LEGACY_IPOPT_LIBRARY}"
      )
    else()
      find_package(PkgConfig REQUIRED)
      pkg_check_modules(legacy_ipopt REQUIRED IMPORTED_TARGET ipopt)
      target_link_libraries(
        ipopt_cxx23_legacy_ipopt
        INTERFACE PkgConfig::legacy_ipopt
      )
    endif()

    add_executable(
      ipopt_cxx23_legacy_ipopt_bridge_test
      tests/legacy_ipopt_bridge_test.cpp
    )
    target_link_libraries(
      ipopt_cxx23_legacy_ipopt_bridge_test
      PRIVATE Ipopt::cxx23 ipopt_cxx23_legacy_ipopt
    )
    ipopt_cxx23_enable_strict_warnings(ipopt_cxx23_legacy_ipopt_bridge_test)
    add_test(
      NAME ipopt_cxx23_legacy_ipopt_bridge
      COMMAND ipopt_cxx23_legacy_ipopt_bridge_test
    )

    add_executable(
      ipopt_cxx23_legacy_aug_system_backend_test
      tests/legacy_aug_system_backend_test.cpp
    )
    target_link_libraries(
      ipopt_cxx23_legacy_aug_system_backend_test
      PRIVATE Ipopt::cxx23 ipopt_cxx23_legacy_ipopt
    )
    ipopt_cxx23_enable_strict_warnings(
      ipopt_cxx23_legacy_aug_system_backend_test
    )
    add_test(
      NAME ipopt_cxx23_legacy_aug_system_backend
      COMMAND ipopt_cxx23_legacy_aug_system_backend_test
    )

    add_executable(
      ipopt_cxx23_legacy_algorithm_canary_test
      tests/legacy_algorithm_canary_test.cpp
    )
    target_link_libraries(
      ipopt_cxx23_legacy_algorithm_canary_test
      PRIVATE Ipopt::cxx23 ipopt_cxx23_legacy_ipopt
    )
    ipopt_cxx23_enable_strict_warnings(
      ipopt_cxx23_legacy_algorithm_canary_test
    )
    add_test(
      NAME ipopt_cxx23_legacy_algorithm_canary
      COMMAND ipopt_cxx23_legacy_algorithm_canary_test
    )

    add_executable(
      ipopt_cxx23_legacy_primal_dual_stage_canary_test
      tests/legacy_primal_dual_stage_canary_test.cpp
    )
    target_link_libraries(
      ipopt_cxx23_legacy_primal_dual_stage_canary_test
      PRIVATE Ipopt::cxx23 ipopt_cxx23_legacy_ipopt
    )
    ipopt_cxx23_enable_strict_warnings(
      ipopt_cxx23_legacy_primal_dual_stage_canary_test
    )
    add_test(
      NAME ipopt_cxx23_legacy_primal_dual_stage_canary
      COMMAND ipopt_cxx23_legacy_primal_dual_stage_canary_test
    )

    add_executable(
      ipopt_cxx23_pdfullspace_fgmres_refinement_test
      tests/pdfullspace_fgmres_refinement_test.cpp
    )
    target_link_libraries(
      ipopt_cxx23_pdfullspace_fgmres_refinement_test
      PRIVATE Ipopt::cxx23 ipopt_cxx23_legacy_ipopt
    )
    ipopt_cxx23_enable_strict_warnings(
      ipopt_cxx23_pdfullspace_fgmres_refinement_test
    )
    add_test(
      NAME ipopt_cxx23_pdfullspace_fgmres_refinement
      COMMAND ipopt_cxx23_pdfullspace_fgmres_refinement_test
    )

    add_executable(
      ipopt_cxx23_numerical_primitives_test
      tests/numerical_primitives_test.cpp
    )
    target_link_libraries(
      ipopt_cxx23_numerical_primitives_test
      PRIVATE Ipopt::cxx23 ipopt_cxx23_legacy_ipopt
    )
    ipopt_cxx23_enable_strict_warnings(
      ipopt_cxx23_numerical_primitives_test
    )
    add_test(
      NAME ipopt_cxx23_numerical_primitives
      COMMAND ipopt_cxx23_numerical_primitives_test
    )

    add_executable(
      ipopt_cxx23_adversarial_nlp_test
      tests/adversarial_nlp_test.cpp
    )
    target_link_libraries(
      ipopt_cxx23_adversarial_nlp_test
      PRIVATE Ipopt::cxx23 ipopt_cxx23_legacy_ipopt
    )
    ipopt_cxx23_enable_strict_warnings(
      ipopt_cxx23_adversarial_nlp_test
    )
    add_test(
      NAME ipopt_cxx23_adversarial_nlp
      COMMAND ipopt_cxx23_adversarial_nlp_test
    )
  endif()
endif()

if(IPOPT_CXX23_BUILD_BENCHMARKS)
  add_executable(
    ipopt_cxx23_anyany_jacobian_product_benchmark
    benchmarks/anyany_jacobian_product_benchmark.cpp
  )
  target_link_libraries(
    ipopt_cxx23_anyany_jacobian_product_benchmark
    PRIVATE Ipopt::cxx23
  )
  ipopt_cxx23_enable_strict_warnings(
    ipopt_cxx23_anyany_jacobian_product_benchmark
  )

  add_executable(
    ipopt_cxx23_fgmres_reuse_benchmark
    benchmarks/fgmres_reuse_benchmark.cpp
  )
  target_link_libraries(
    ipopt_cxx23_fgmres_reuse_benchmark
    PRIVATE Ipopt::cxx23
  )
  ipopt_cxx23_enable_strict_warnings(
    ipopt_cxx23_fgmres_reuse_benchmark
  )

  add_executable(
    ipopt_cxx23_block_tridiagonal_solver_benchmark
    benchmarks/block_tridiagonal_solver_benchmark.cpp
  )
  target_link_libraries(
    ipopt_cxx23_block_tridiagonal_solver_benchmark
    PRIVATE Ipopt::cxx23
  )
  ipopt_cxx23_enable_strict_warnings(
    ipopt_cxx23_block_tridiagonal_solver_benchmark
  )
endif()
