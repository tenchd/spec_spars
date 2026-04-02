#!/bin/bash

# make a config.toml file with placeholder values, which the user will replace with the correct paths.
printf "fast_mtx_path = \"path/to/fast_matrix_market/include\" \ninput_files_path = \"path/to/input/files\" \noutput_files_path = \"path/to/output/files\"" > config.toml

# download the virus dataset and put it in the data directory.
mkdir -p data
cd data
wget https://portal.nersc.gov/project/m1982/HipMCL/viruses/vir_vs_vir_30_50length_propermm.mtx
mv vir_vs_vir_30_50length_propermm.mtx virus.mtx