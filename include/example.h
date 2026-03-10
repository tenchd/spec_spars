#pragma once
#include "spec_spars/src/main.rs.h"
#include "rust/cxx.h"
#include <iostream>
#include "spec_spars/include/custom_cg.hpp"


//rust::Vec<Shared> f(rust::Vec<Shared> elements);

FlattenedVec go(FlattenedVec shared_jl_cols);

FlattenedVec test_roll(FlattenedVec jl_cols);

void sprs_test(rust::Vec<size_t> rust_col_ptrs, rust::Vec<size_t> rust_row_indices, rust::Vec<double> rust_values);

void sprs_correctness_test(rust::Vec<int> rust_col_ptrs, rust::Vec<int> rust_row_indices, rust::Vec<double> rust_values);

FlattenedVec run_solve_lap(FlattenedVec shared_jl_cols, rust::Vec<int> rust_col_ptrs, rust::Vec<int> rust_row_indices, rust::Vec<double> rust_values, rust::Str solver_output_filename, int num_nodes, bool verbose);

void julia_test_solve(FlattenedVec interop_jl_cols, rust::Vec<int> rust_col_ptrs, rust::Vec<int> rust_row_indices, rust::Vec<double> rust_values, int num_nodes);

bool test_stager(FlattenedVec interop_jl_cols, rust::Vec<int> rust_col_ptrs, rust::Vec<int> rust_row_indices, rust::Vec<double> rust_values, rust::Str input_filename, rust::Str julia_lap_filename, rust::Str julia_sketch_product_filename, rust::Str solver_output_filename, int num_nodes, int test_selector, bool verbose);

FlattenedVec test_diff_norm(rust::Str rust_lap_filename, rust::Str julia_sketch_product_filename, rust::Str solver_output_filename);