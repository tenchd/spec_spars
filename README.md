Prototype implementation of streaming spectral sparsifier, described in [A Framework for Analyzing Resparsification Algorithms](https://arxiv.org/abs/1611.06940) by Kyng et. al. 

This codebase is written to be run on [NERSC](https://www.nersc.gov/), the supercomputing cluster at Lawrence Berkeley National Lab. **It can be run on non-NERSC Linux systems, provided Intel MKL libraries are available, with some configuration** (see below).

## Quick Install/Demo (NERSC only)
1. Clone the [fast matrix market repo](https://github.com/alugowski/fast_matrix_market#) to any location you like.
2. Clone this project and `cd` into the resulting directory.
3. Run `bash setup.sh` to generate an example config file and download some example datasets to sparsify.
4. `cp example_config.toml config.toml`
5. Edit the 'fast_mtx_path' line in config.toml to point to the location of the include subdirectory in the fast_matrix_market directory (from step 1).
6. Run `module load intel` to load the intel mkl code.
7. Run `export CXX=/usr/bin/g++` which is needed for compilation of C++ code within Rust.
8. Run `cargo run --release -- -p` to sparsify the example datasets.

## A Brief Overview of Spectral Sparsification.

The goal of spectral sparsification is, given a graph $G$, to produce a sparser graph $H$ that approximately preserves the spectral properties of $G$. Specifically, for some $0 < \epsilon < 1$ we say that $H$ is an $\epsilon$-spectral sparsifier of $G$ iff

$$  
(1 - \epsilon) x^T L_G x \leq x^T L_H x \leq (1 + \epsilon) x^T L_G x \quad \forall x \in \mathbb{R}^n
$$

where $L_G$ denotes the $n \times n$ Laplacian matrix of $G$.

The goal of this code is to compute a spectral sparsifier of a given input graph with $O(n \log n \epsilon^{-2})$ edges. Further, the goal is to compute this sparsifier in a streaming fashion, meaning that the edges of the input graph are given to the system in an arbitrary order, and it must use no more than $O(n \log n \epsilon^{-2})$ words of space at any time during computation.

Spectral sparsifiers are useful for a number of reasons. They preserve important and commonly-used combinatorial graph properties such as connectivity and cuts. For many applications, computing a spectral sparsifier and using it for downstream analytic tasks in place of the original graph reducces both running time and space costs, increasing the scale at which the application can be used.

## Project Status

This codebase is an alpha prototype of an implementation of the streaming spectral sparsification algoritm of Kyng et. al. At the moment, the code is developed and documented well enough for research collaborators to use. It is not ready for general release, however. Currently, the codebase supports:

- reading in an input graph as a matrix market file, and computing a (non-streaming) spectral sparsifier.

- incomplete correctness testing and performance experiments

Upcoming features include:

- full streaming spectral sparsification from a matrix market file (including sublinear memory cost)

- support for more input formats

- complete correctness testing and performance benchmarking suites

- sample uses of spectral sparsification on several large-scale scientific analysis tasks.

## Installation (Linux, non-NERSC)

Requirements: 
[Rust](https://rust-lang.org/tools/install/)
[Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html) (I HIGHLY recommend you install via the instructions provided in the link, if you are not already familiar with MKL.)

1. Clone the [fast matrix market repo](https://github.com/alugowski/fast_matrix_market#) to any location you like.
2. Clone this spectral sparsification repo and `cd` into the resulting directory.
3. Run `bash setup.sh` to generate an example config file and download some example datasets to sparsify.
4. `cp example_config.toml config.toml`
5. Edit the 'fast_mtx_path' line in config.toml to point to the location of the include subdirectory in the fast_matrix_market directory (from step 1). Make sure to use an absolute file path.
6. Load the Intel MKL environment variables. If your system has a module manager that can do this, use it. Otherwise, find the install location of MKL on your system and run `source path-to-mkl-install/intel/oneapi/setvars.sh`. Verify that this worked with `which mkl_link_tool`; if the variables are loaded it will print out a filepath to mkl_link_tool.
7. Run `export CXX=/usr/bin/g++` which is needed for compilation of C++ code within Rust.
8. Run `cargo run --release -- -p` to sparsify the example datasets.

Note that steps 6 and 7 must be done *every time* you start a new session, or else the code will not compile. You can also add these commands to your bashrc file to avoid having to do these every time.

## Command-line arguments

You can pass optional command-line arguments after `cargo run --release --` to control program behavior. 

-i, --input_file: 
Allows you to specify the path to an input .mtx file you want to sparsify.

-o, --output_file:
Allows you to specify where to write the .mtx file for the sparsified graph. Default: "" which writes nothing.

-d --dataset_name: 
Allows you to assign a name to the dataset stored in the input file, which affects the name of the output files written after sparsification.

-e, --epsilon: 
Allows you to specify a value for epsilon, controlling approximation quality.

-v, --verbose: 
Invoking this prints out debug information during sparsification run, in particular the preconditioning and linear solve phases.

-s, --seed: 
Allows you to specify a random seed for experiment reproducibility.

-b, --benchmark_skip: 
By default, the program prints benchmarking information for each sparsification. Invoking this flag disables this behavior.

-p, --process_all: 
Invoking this ignores input_file and dataset_name parameters and sparsifies a set list of datasets according to other command line arguments. Edit `process_standard_datasets` in main.rs to change this list of datasets (if you do, make sure to assign a reasonable dataset name for each input file).

## Running Tests
Run `cargo test --release` to run a suite of unit and integration tests. 