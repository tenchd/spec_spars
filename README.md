Prototype implementation of streaming spectral sparsifier.

## Compiling and Running on NERSC
Clone the project and `cd` into it.

Run `module load intel` to load the intel mkl code.

Run `export CXX=/usr/bin/g++` which is needed for compilation of C++ code within Rust.

To run an example where the virus dataset is sparsified, you can `cargo run`. 

To compare performance of the two jl sketch methods, you can run `cargo test jl_sketch_equiv -- --nocapture`.
This computes the JL sketch of the EVIM via nonblocked and blocked approaches, and compares the time required.
The test logic is found in src/tests.rs in the function "jl_sketch_equiv".
The implementations of the jl sketch logic are found in src/jl_sketch.rs in the functions "jl_sketch_sparse" and "jl_sketch_sparse_blocked".
