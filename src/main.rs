use spec_spars::lap_test;

fn main() {
    //let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/virus/virus.mtx";
    //let dataset_name = "virus";
    //let output_file_prefix = "virus_sparse";
    let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/mouse_gene/mouse_gene.mtx";
    let dataset_name = "mouse_gene";
    lap_test(input_filename, dataset_name);
}