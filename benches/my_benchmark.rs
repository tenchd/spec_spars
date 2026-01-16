#![allow(unused)]

use criterion::{criterion_group, criterion_main, Criterion};
//use std::hint::black_box;
use spec_spars::{sparsifier::Sparsifier, utils::Benchmarker, tests::make_random_evim_matrix, stream::InputStream};
use sprs::CsMat;

// fn criterion_benchmark(c: &mut Criterion) {
//     c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
// }

fn jl_sketch_naive_random(c: &mut Criterion) {
    let num_rows = 5005;
        let num_cols = 60000;
        let csc = true;
            
        let seed: u64 = 1;
        let jl_factor: f64 = 1.5;

        let epsilon = 0.5;
        let beta_constant = 4;
        let row_constant = 2;
        let verbose = false;
        let jl_dim = ((num_rows as f64).log2() *jl_factor).ceil() as usize;

        let benchmarker = Benchmarker::new(false);
        let sparsifier = Sparsifier::new(num_rows, epsilon, beta_constant, row_constant, verbose, jl_factor, seed, benchmarker);

        let input_matrix = make_random_evim_matrix(num_rows, num_cols, csc);
        c.bench_function("naive jl sketch, random input", |b| b.iter(|| sparsifier.jl_sketch_sparse(&input_matrix)));
}

fn jl_sketch_fast_random(c: &mut Criterion) {
    let num_rows = 5005;
    let num_cols = 60000;
    let csc = true;
        
    let seed: u64 = 1;
    let jl_factor: f64 = 1.5;

    let epsilon = 0.5;
    let beta_constant = 4;
    let row_constant = 2;
    let verbose = false;
    let jl_dim = ((num_rows as f64).log2() *jl_factor).ceil() as usize;

    let benchmarker = Benchmarker::new(false);
    let sparsifier = Sparsifier::new(num_rows, epsilon, beta_constant, row_constant, verbose, jl_factor, seed, benchmarker);

    let input_matrix = make_random_evim_matrix(num_rows, num_cols, csc);
    let mut colwise_batch_answer:CsMat<f64> = CsMat::zero((num_rows, jl_dim)).into_csc();
    c.bench_function("fast jl sketch, random input", |b| b.iter(|| sparsifier.jl_sketch_colwise_batch(&input_matrix, &mut colwise_batch_answer)));
}

fn jl_sketch_naive_virus(c: &mut Criterion) {
    let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/virus/virus.mtx";
    let seed: u64 = 1;
    let jl_factor: f64 = 1.5;

    let epsilon = 0.5;
    let beta_constant = 4;
    let row_constant = 2;
    let verbose = false;

    let stream = InputStream::new(input_filename);
    let num_rows = stream.num_nodes;
    let jl_dim = ((num_rows as f64).log2() *jl_factor).ceil() as usize;

    let benchmarker = Benchmarker::new(false);
    let mut sparsifier = Sparsifier::new(stream.num_nodes.try_into().unwrap(), epsilon, beta_constant, row_constant, verbose, jl_factor, seed, benchmarker);

    for (value, (row, col)) in stream.input_matrix.iter() {
        sparsifier.insert(row.try_into().unwrap(), col.try_into().unwrap(), *value);
    }

    let input_matrix: CsMat<f64> = sparsifier.new_entries.to_edge_vertex_incidence_matrix();
    c.bench_function("naive jl sketch, virus input", |b| b.iter(|| sparsifier.jl_sketch_sparse(&input_matrix)));
}

fn jl_sketch_fast_virus(c: &mut Criterion) {
    let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/virus/virus.mtx";
    let seed: u64 = 1;
    let jl_factor: f64 = 1.5;

    let epsilon = 0.5;
    let beta_constant = 4;
    let row_constant = 2;
    let verbose = false;

    let stream = InputStream::new(input_filename);
    let num_rows = stream.num_nodes;
    let jl_dim = ((num_rows as f64).log2() *jl_factor).ceil() as usize;

    let benchmarker = Benchmarker::new(false);
    let mut sparsifier = Sparsifier::new(stream.num_nodes.try_into().unwrap(), epsilon, beta_constant, row_constant, verbose, jl_factor, seed, benchmarker);

    for (value, (row, col)) in stream.input_matrix.iter() {
        sparsifier.insert(row.try_into().unwrap(), col.try_into().unwrap(), *value);
    }

    let input_matrix: CsMat<f64> = sparsifier.new_entries.to_edge_vertex_incidence_matrix();
    let mut colwise_batch_answer:CsMat<f64> = CsMat::zero((num_rows, jl_dim)).into_csc();
    c.bench_function("fast jl sketch, virus input", |b| b.iter(|| sparsifier.jl_sketch_colwise_batch(&input_matrix, &mut colwise_batch_answer)));
}

criterion_group!(benches, jl_sketch_naive_random, jl_sketch_fast_random, jl_sketch_naive_virus, jl_sketch_fast_virus);
criterion_main!(benches);