Prototype implementation of streaming spectral sparsifier.

Note that, currently, only running the code on NERSC is supported.

## Installation
Clone the project and `cd` into the resulting directory.

Run `mkdir data`.

## Compiling and Running on NERSC
1) Run `module load intel` to load the intel mkl code.

2) Run `export CXX=/usr/bin/g++` which is needed for compilation of C++ code within Rust.

Note that 1) and 2) need to be done *every time* you start a new session, or else the code will not compile.

To run a demo, you can `cargo run --release -- -p`. This sparsifies a pre-specified group of datasets.

## Command-line arguments

You can pass optional command-line arguments after `cargo run --release --` to control program behavior. 

-i, --input_file: 
Allows you to specify the path to an input .mtx file you want to sparsify.

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

