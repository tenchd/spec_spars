#!/bin/bash

# make a config.toml file with placeholder values, which the user will replace with the correct paths.
printf "fast_mtx_path = \"path/to/fast_matrix_market/include\" \ninput_files_path = \"path/to/input/files\" \noutput_files_path = \"path/to/output/files\"" > example_config.toml

# download the virus dataset and put it in the data directory.
mkdir -p data
cd data
wget https://portal.nersc.gov/project/m1982/HipMCL/viruses/vir_vs_vir_30_50length_propermm.mtx
mv vir_vs_vir_30_50length_propermm.mtx virus.mtx
wget https://suitesparse-collection-website.herokuapp.com/MM/Belcastro/human_gene1.tar.gz
tar -xvzf human_gene1.tar.gz
mv human_gene1/human_gene1.mtx human1.mtx
rm human_gene1.tar.gz
rm -rf human_gene1/
wget https://suitesparse-collection-website.herokuapp.com/MM/Belcastro/human_gene2.tar.gz
tar -xvzf human_gene2.tar.gz
mv human_gene2/human_gene2.mtx human2.mtx
rm human_gene2.tar.gz
rm -rf human_gene2/
wget https://suitesparse-collection-website.herokuapp.com/MM/Belcastro/mouse_gene.tar.gz
tar -xvzf mouse_gene.tar.gz
mv mouse_gene/mouse_gene.mtx mouse.mtx
rm mouse_gene.tar.gz
rm -rf mouse_gene/
cd ..