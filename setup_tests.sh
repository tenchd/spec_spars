# write the intended output location below. Make sure to use the absolute file path andinclude the trailing slash.
OUTPUT_LOCATION="/global/homes/d/dtench/m1982/david/julia_test_output/"

echo "Installing required Julia packages for test setup."
julia reference_implementation/jsetup.jl
echo "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
echo "Package installation complete. Running reference implementation to generate output files for testing."
julia reference_implementation/stream_sparsify.jl $OUTPUT_LOCATION
echo "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
echo "Test setup complete. Reference files written to $OUTPUT_LOCATION"