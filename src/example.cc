
#include "cxx-test/include/example.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <vector>
#include <chrono>
#include <atomic>

//MOST OF THIS FILE IS COPIED FROM DRIVER_LOCAL.CPP

typedef int custom_idx;

// for reading in jl columns in csv format
template <typename type_int>
void readVectorFromCSV(const std::string& filename, std::vector<type_int>& values) {
    
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string line;
    std::getline(file, line);

    std::stringstream ss(line);
    std::string token;
// TODO: add error checking to make sure that we don't go over the bounds
    while (std::getline(ss, token, ',')) {
        try {
            values.push_back(std::stof(token));
        } catch (const std::invalid_argument& e) {
            throw std::runtime_error("Invalid float value: " + token);
        } catch (const std::out_of_range& e) {
            throw std::runtime_error("Float value out of range: " + token);
        }
    }

    file.close();
}

template <typename type_int>
void readValuesFromFile(const std::string& filename, std::vector<std::vector<type_int>>& values) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<type_int> lineFloats;

        while (std::getline(ss, token, ',')) {
            try {
                lineFloats.push_back(std::stof(token));
            } catch (const std::invalid_argument& e) {
                throw std::runtime_error("Invalid float value: " + token);
            } catch (const std::out_of_range& e) {
                throw std::runtime_error("Float value out of range: " + token);
            }
        }
        values.push_back(lineFloats);
    }
    file.close();
}

template <typename type_int>
void writeVectorToFile(const std::vector<type_int>& vec, const std::string& filename) 
{
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    
    for (const type_int& element : vec) {
        outFile << element << "\n";
    }
    outFile.close();
}

template <typename type_int, typename type_data>
void perform_factorization_device(const custom_space::sparse_matrix<type_int, type_data> &spmat_device, type_int map_size,
    Edge<type_int, type_data> *device_edge_map, type_int output_size,
       type_int *device_min_dependency_count, Node<type_int, type_data> *device_node_list, type_int *device_output_position_idx, type_int *queue_device,
            type_int thread_id, type_int total_threads, int64_t *test_vec, int *schedule_id) 
{

    int gap = total_threads;
    //printf("id: %d, cpu: %d, gap: %d\n", thread_id, sched_getcpu(), gap);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    int neg_track = 0;
    int positive_track = 0;

    type_int num_cols = spmat_device.num_cols;
    type_int output_start_array[4];
    std::vector<Edge<type_int, type_data>> local_work_space_vec(2000);

    auto start = std::chrono::steady_clock::now();

    // assign the job id, will consider using a scheduler in the future
    // if id is within valid range of queue'
    type_int queue_access_index = thread_id;
    int64_t job_id = -1;
    std::atomic_ref<type_int> queue_size(queue_device[num_cols]);
    //std::atomic_ref<int64_t> test_size(test_vec[num_cols]);
    
    if(queue_access_index < queue_size.load())
    {
        std::atomic_ref<type_int> poll_id(queue_device[queue_access_index]);

        job_id = poll_id.load();
        // this means that even though the size was modified, the actual job haven't been inserted
        if(job_id == 0 && queue_access_index != 0)
        {
            job_id = -1;
        }
    }

    while(queue_access_index < num_cols)
    {
        if(job_id == -1)
        {
            std::atomic_ref<type_int> poll_id(queue_device[queue_access_index]);
            
            // this means that even though the size was modified, the actual job haven't been inserted
            while(poll_id.load(std::memory_order_acquire) == 0 && queue_access_index != 0)
            {
                job_id = -1;
            }
            job_id = poll_id.load(std::memory_order_acquire);
            neg_track++;
            continue;
        }
        positive_track++;

        // job successfully queued up, move to next search location
        queue_access_index += gap;
        // job id is the last column, skip
        if(job_id == num_cols - 1)
        {
            std::atomic_ref<type_int> last_col_loc_ref(*device_output_position_idx);
            type_int last_col_loc = last_col_loc_ref.fetch_add(1);

            device_node_list[num_cols - 1].start = last_col_loc;
            device_node_list[num_cols - 1].count = 0;
            device_node_list[num_cols - 1].sum = 0.0;
        
            job_id = -1;
            continue;
        }

        std::atomic_ref<type_int> atomic_dependency_count(device_min_dependency_count[job_id]);
        
        while(atomic_dependency_count.load() > 0)
        {
            //continue;
        }
            
        type_int left_bound_idx = spmat_device.col_ptrs[job_id];
        type_int num_of_original_nnz = spmat_device.col_ptrs[job_id + 1] - left_bound_idx;
        type_int update_neighbor_count = search_for_updates<type_int, type_data>(job_id, map_size, num_of_original_nnz,
            device_node_list, device_output_position_idx, output_start_array, device_edge_map, local_work_space_vec);
        
        type_int total_neighbor_count = num_of_original_nnz + update_neighbor_count;
        Edge<type_int, type_data> *local_work_space = local_work_space_vec.data();

        // if no entry in this column
        if(total_neighbor_count == 0)
        {
            
            device_node_list[job_id].count = 0;
            device_node_list[job_id].sum = 0.0;
            job_id = -1;
            continue;
        }

        {
            type_int edge_start = 0;

            // 1. read in the nonzeros in the original input and sort entire input first based on row value
            if(edge_start + total_neighbor_count >= local_work_space_vec.size())
            {
                local_work_space_vec.resize((edge_start + total_neighbor_count + 1) * 2);
                local_work_space = local_work_space_vec.data();
            }
            
            for(type_int i = edge_start + update_neighbor_count; i < edge_start + total_neighbor_count; i++)
            {
                local_work_space[i] = Edge<type_int, type_data>(spmat_device.row_indices[left_bound_idx + i - edge_start - update_neighbor_count], 
                    spmat_device.values[left_bound_idx + i - edge_start - update_neighbor_count], 1);
            }

            //odd_even_sort(device_factorization_output, edge_start, edge_start + total_neighbor_count, lane_id, compare_row);
            std::sort(local_work_space + edge_start, local_work_space + edge_start + total_neighbor_count, [](const Edge<type_int, type_data>& a, const Edge<type_int, type_data>& b) {
                return a.row < b.row;
            });
   
            // 3. merge entries with the same row value
            // count how many actual distinct elements there are after merging
            type_int actual_neighbor_count = 0;
            if(total_neighbor_count > 0)
            {
                Edge<type_int, type_data> *edge_ref = &local_work_space[edge_start];
                actual_neighbor_count++;
                
                for(type_int i = edge_start + 1; i < edge_start + total_neighbor_count; i++)
                {
                    if(local_work_space[i].row == edge_ref->row)
                    {
                        // merge the two
                        edge_ref->value += local_work_space[i].value;
                        edge_ref->multiplicity += local_work_space[i].multiplicity;
                    }
                    else
                    {
                        // update the chase pointer, shift the new distinct element to here, DON'T ACCUMULATE (i.e. no +=)
                        edge_ref++;
                        actual_neighbor_count++;
                        edge_ref->value = local_work_space[i].value;
                        edge_ref->multiplicity = local_work_space[i].multiplicity;
                        edge_ref->row = local_work_space[i].row;
                    }
                }
            }
            
            // update device_node_list with new locations after merging and update values after merging
            device_node_list[job_id].count = actual_neighbor_count;
            total_neighbor_count = actual_neighbor_count;

            // 5. sort input based on value
            //odd_even_sort(device_factorization_output, edge_start, edge_start + total_neighbor_count, lane_id, compare_value);
            std::sort(local_work_space + edge_start, local_work_space + edge_start + total_neighbor_count, [](const Edge<type_int, type_data>& a, const Edge<type_int, type_data>& b) {
                return a.value < b.value;
            });
            
            // compute cumulative sum
            type_data total_sum = 0.0;
            if(total_neighbor_count > 0)
            {
                local_work_space[edge_start].forward_cumulative_value = local_work_space[edge_start].value;
                total_sum += local_work_space[edge_start].value;
                for(type_int i = edge_start + 1; i < edge_start + total_neighbor_count; i++)
                {
                    local_work_space[i].forward_cumulative_value = local_work_space[i].value + local_work_space[i - 1].forward_cumulative_value;
                    total_sum += local_work_space[i].value;
                }
            }
            device_node_list[job_id].sum = total_sum;

            /* Generate Samples and set up links */
            for(type_int i = edge_start; i < edge_start + total_neighbor_count - 1; i++)
            {
                double number_decision = dis(gen) * (total_sum - local_work_space[i].forward_cumulative_value);
                Edge<type_int, type_data> *edge_iter = std::lower_bound(local_work_space + i + 1, local_work_space + edge_start + total_neighbor_count, number_decision, 
                    [local_work_space, i](const Edge<type_int, type_data>& element, const double& value) -> bool {
                        return (element.forward_cumulative_value - local_work_space[i].forward_cumulative_value) < value;
                    });
                type_int generated_row = std::max(edge_iter->row, local_work_space[i].row);
                type_int generated_col = std::min(edge_iter->row, local_work_space[i].row);
                type_data generated_value = local_work_space[i].value * (total_sum - local_work_space[i].forward_cumulative_value) / total_sum;

                // set up the generated edge
                local_work_space[i].sampled_row = generated_row;
                local_work_space[i].sampled_value = generated_value;

                // atomically update the link, must use atomics here
                std::atomic_ref<int64_t> link_ref(device_node_list[generated_col].prev);
                local_work_space[i].prev = link_ref.exchange(i + output_start_array[0]);

                // add count
                std::atomic_ref<type_int> col_count_ref(device_node_list[generated_col].count);
                col_count_ref.fetch_add(1);

                // add dependency update
                std::atomic_ref<type_int> dependency_update(device_min_dependency_count[generated_row]);
                dependency_update.fetch_add(1);
            }

            // scale by total sum
            for(type_int i = edge_start; i < edge_start + total_neighbor_count; i++)
            {
                local_work_space[i].value = local_work_space[i].value / total_sum;
            }
                
            // copy to global space
            for(type_int i = edge_start; i < edge_start + total_neighbor_count; i++)
            {
                device_edge_map[i + output_start_array[0]] = local_work_space[i]; 
            }            

            // update dependency by subtracting away from ones impacted by current node/column
            for(type_int i = edge_start + output_start_array[0]; i < edge_start + total_neighbor_count + output_start_array[0]; i++)
            {
                std::atomic_ref<type_int> dependency_ref(device_min_dependency_count[device_edge_map[i].row]);
                type_int old_dependency = dependency_ref.fetch_sub(device_edge_map[i].multiplicity);

                if(old_dependency == device_edge_map[i].multiplicity)
                {
                   type_int old_queue_size = queue_size.fetch_add(1);
                    //queue_size = atomicAdd(&queue_device[num_cols], 1);
                   std::atomic_ref<type_int> job_schedule_ref(queue_device[old_queue_size]);
                   job_schedule_ref.exchange(device_edge_map[i].row, std::memory_order_release);
                }
            }
        }

        // reset job id to -1, so it will look for a new job
        job_id = -1;
    }
    
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end - start;
}

bool compare(const Edge<int, double> &a, const Edge<int, double> &b)
{
    return a.row < b.row;  
};

template <typename type_int, typename type_data>
bool factorization_driver(sparse_matrix_processor<type_int, type_data> &processor, type_int num_threads, char* path, bool is_graph, \
    std::vector<std::vector<type_data>>& jl_cols, std::vector<std::vector<type_data>>& solution)
{
    assert(INT_MAX == 2147483647);
    int space_multiply_factor = 5;
    int edge_pool_size = processor.mat.nonZeros() * space_multiply_factor;
    if(edge_pool_size < 0)
    {
        printf("WARNING: allocation requirement became negative, indicating an int overflow");
        edge_pool_size = INT_MAX - 100000000;
        //assert(false);
    }
    printf("Edge size: %ld\n", sizeof(Edge<type_int, type_data>{}));

    // Allocate memory on the host (CPU)
    Edge<type_int, type_data> *host_edge_map = (Edge<type_int, type_data> *)malloc(edge_pool_size * sizeof(Edge<type_int, type_data>{}));
    for(size_t i = 0; i < edge_pool_size; i++) 
    {
        host_edge_map[i] = Edge<type_int, type_data>();
    }
    
    type_int host_output_position_idx = 0;

    // copy sparse matrix from cpu to gpu
    custom_space::sparse_matrix<type_int, type_data> spmat = processor.make_lower_triangular(processor.mat);

    // copy min_dependency_array
    if(spmat.num_cols != processor.min_dependency_count.size())
    {
        printf("processor.mat.num_cols: %d, spmat.num_cols: %d, min_dependency_count.size(): %ld\n", processor.mat.num_cols, spmat.num_cols, processor.min_dependency_count.size());
        assert(spmat.num_cols == processor.min_dependency_count.size());
    }
    
    std::vector<type_int> queue_cpu(spmat.num_cols + 2, 0); // last two element represent counters
    std::vector<int64_t> test_vec(spmat.num_cols + 1, 0);
    
    for(type_int i = 0; i < processor.min_dependency_count.size(); i++)
    {
        if(processor.min_dependency_count[i] == 0)
        {
            queue_cpu[queue_cpu[queue_cpu.size() - 2]] = i;
            queue_cpu[queue_cpu.size() - 2]++;

            test_vec[test_vec[test_vec.size() - 1]] = i;
            test_vec[test_vec.size() - 1]++;
       }
    }

    assert(queue_cpu[0] == 0);
    printf("initial queue size: %d\n", queue_cpu[queue_cpu.size() - 2]);

    std::vector<type_int> min_dependency_count = processor.min_dependency_count;

    // create node array
    std::vector<Node<type_int, type_data>> node_list_host(spmat.num_cols);
    for(size_t i = 0; i < spmat.num_cols; i++)
    {
        node_list_host[i] = Node<type_int, type_data>(0, 0, i);
    }
  
    printf("CURRENTLY, MULTIPLICITY STORAGE USES INT32, DON'T NEED 64 BIT UNLESS IN EXTREME CIRCUMSTANCES \n");
    printf("CURRENTLY, BINARY SEARCH USES INT32, DON'T NEED 64 BIT UNLESS IN EXTREME CIRCUMSTANCES \n");
    printf("defaulting row and col to 0 in map may cause problem for column 0 \n");
    
    int schedule_id[] = {0, 128, 1, 129, 2, 130, 3, 131, 4, 132, 5, 133, 6, 134, 7, 135, 8, 136, 9, 137, 10, 138, 11, 139, 12, 140, 13, 141, 14, 142, 15, 143};
    
    omp_set_num_threads(num_threads);

    // Parallel region starts
    #pragma omp parallel
    {
        auto start = std::chrono::steady_clock::now();

        perform_factorization_device<type_int, type_data>(spmat, edge_pool_size, host_edge_map, edge_pool_size, min_dependency_count.data(), node_list_host.data(), 
            &host_output_position_idx, queue_cpu.data(), omp_get_thread_num(), num_threads, test_vec.data(), schedule_id);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> duration = end - start;

        #pragma omp barrier

        if(omp_get_thread_num() == 0)
        {
            printf("Factorization execution time: %f seconds\n", duration.count());
            //std::cout << "Factorization execution time: " << duration.count() << " seconds" << std::endl;
        }
        
    }

    // make sure to add 1 to the size of each column since factorization did not include diagonal
    std::vector<type_int> csr_rowptr_host(spmat.num_cols + 1);
    std::vector<type_data> diagonal_entries(spmat.num_cols);
    csr_rowptr_host[0] = 0;
    size_t total_needed_size = 0;
    for(size_t i = 0; i < node_list_host.size(); i++)
    {
        
        total_needed_size = total_needed_size + node_list_host[i].count + 1;
        csr_rowptr_host[i + 1] = total_needed_size;

        // sort list and compute sum, also negate entries since the factorization was done with positive entries
        type_int col_start = node_list_host[i].start;
    
        // compute column sum and append diagonal, negate entries
        for(size_t j = col_start; j < col_start + node_list_host[i].count; j++)
        {
            host_edge_map[j].value = -host_edge_map[j].value;
        }
        diagonal_entries[i] = node_list_host[i].sum;

        // UPDATE COUNT
        host_edge_map[col_start + node_list_host[i].count] = Edge<type_int, type_data>(i, 1.0, 1);
        node_list_host[i].count++;
        std::sort(host_edge_map + col_start, host_edge_map + col_start + node_list_host[i].count, compare);
    }
    
    printf("nnz ratio: %f, factor nnz: %d, original lower triangular nnz (exclude diagonal): %d\n", double(total_needed_size) / double(spmat.nonZeros()), total_needed_size, spmat.nonZeros());
    std::vector<type_data> csr_val_host(total_needed_size);
    std::vector<type_int> csr_col_ind_host(total_needed_size);

    // start writing the result into a csr, preparing for cusparse operations
    for(size_t i = 0; i < node_list_host.size(); i++)
    {
        
        for(size_t j = csr_rowptr_host[i]; j < csr_rowptr_host[i + 1]; j++)
        {
            csr_col_ind_host[j] = host_edge_map[node_list_host[i].start + j - csr_rowptr_host[i]].row;
            csr_val_host[j] = host_edge_map[node_list_host[i].start + j - csr_rowptr_host[i]].value;

            // assert that diagonal elements are 1
            if(host_edge_map[node_list_host[i].start + j - csr_rowptr_host[i]].row == i)
            {
                assert(host_edge_map[node_list_host[i].start + j - csr_rowptr_host[i]].value == 1.0);
            }
            assert(host_edge_map[node_list_host[i].start + j - csr_rowptr_host[i]].row >= i);
        }
    }

    // write solution to file
    std::string prefix(path);
    if(prefix.length() > 0)
    {
        std::string filename = "l_sol.mtx";
        std::string wpath = path + filename;
        std::ofstream output_stream(wpath);
        if (!output_stream.is_open()) {
            std::cerr << "Failed to open file for writing." << std::endl;
            exit(1);
        }
        printf("rowptr size: %d\n", csr_rowptr_host.size());
        printf("col indices size: %d\n", csr_col_ind_host.size());
        // write_csr_to_matrix_market(csr_rowptr_host, csr_col_ind_host, csr_val_host, spmat.num_cols, spmat.num_cols, "c_sol.mtx");
        fast_matrix_market::matrix_market_header header(diagonal_entries.size(), diagonal_entries.size());
        header.object = fast_matrix_market::matrix;
        header.symmetry = fast_matrix_market::general;
        fast_matrix_market::write_options opts;
        opts.precision = 16;
        fast_matrix_market::write_matrix_market_csc(output_stream,
                                    header, 
                                    csr_rowptr_host,
                                    csr_col_ind_host,
                                    csr_val_host,
                                    false,
                                    opts);
        output_stream.flush();  // Ensure any buffered output is written to the file
        output_stream.close();  // Close the file stream when done
        std::string diagname = "sol_diag.txt";
        writeVectorToFile(diagonal_entries, (path + diagname).c_str());
    }

    // find the real e-tree
    std::vector<type_int> etree(spmat.num_cols, 0);
    
    for(size_t i = 0; i < spmat.num_cols; i++)
    {
        if(csr_rowptr_host[i] + 1 < csr_rowptr_host[i + 1])
        {
            etree[i] = csr_col_ind_host[csr_rowptr_host[i] + 1];
        }
    }
    std::vector<std::vector<type_int>> ftree = processor.create_factorization_tree_from_etree(etree);

    std::vector<type_int> layer_info = processor.layer_information(ftree);

    size_t verify_count = 0;
    for(size_t i = 0; i < layer_info.size(); i++)
    {
        verify_count += layer_info[i];
    }
    std::cout << "actual depth of tree after factorization: " << layer_info.size() - 1 << ", total count: " << verify_count << ", number of partitions: " << layer_info[1] << "\n";

    // triangular solve longest DAG path

    std::vector<size_t> max_path_dp(spmat.num_cols, 1);
    for(size_t i = 0; i < spmat.num_cols; i++)
    {
        size_t left_end = csr_rowptr_host[i];
        size_t right_end = csr_rowptr_host[i + 1];

        for(size_t j = left_end + 1; j < right_end; j++)
        {
            max_path_dp[csr_col_ind_host[j]] = std::max(max_path_dp[i] + 1, max_path_dp[csr_col_ind_host[j]]);
        }
        
    }
    auto max_path = std::max_element(max_path_dp.begin(), max_path_dp.end());

    printf("triangular solve max path: %d at index: %d\n", *max_path, std::distance(max_path_dp.begin(), max_path));
  
    

    custom_space::sparse_matrix<type_int, type_data> precond_M(processor.mat.rows(), processor.mat.cols(), std::move(csr_val_host), std::move(csr_col_ind_host), std::move(csr_rowptr_host));
    bool all_succeeded = true;
    auto num_solve = 1;
    for (std::vector<double> right_hand_side: jl_cols){
        printf("---------------Performing solve %i\n", num_solve);

        std::vector<type_data> solution_col;
        bool converged;
        tie(solution_col, converged) = example_pcg_solver(processor.mat, precond_M, diagonal_entries.data(), is_graph, right_hand_side);
        //example_pcg_solver(processor.mat, precond_M, diagonal_entries.data(), is_graph);

        //printf("trying to write to solution column %d\n", num_solve-1);
        solution.at(num_solve-1) = solution_col;
        num_solve++;
        if (!converged) {
            all_succeeded = false;
        }
    }
    return all_succeeded;
}
//END COPIED CODE

// MODIFIED FROM DRIVER_LOCAL.CPP MAIN FILE
// this code essentially copies the behavior of the main() function in driver_local.cpp, except I hardcoded in values for the arguments for simplicity.
// currently hangs at the preconditioner if num_threads > 1; preconditioner works if num_threads=1 but the solves fail.
void run_solve(std::vector<std::vector<double>> jl_cols, std::vector<std::vector<double>>& solution) {

  constexpr const char *input_filename = "/global/u1/d/dtench/cholesky/Parallel-Randomized-Cholesky/physics/parabolic_fem/parabolic_fem-nnz-sorted.mtx";
  int num_threads = 1; 
  constexpr char *output_filename = "output.txt";
  bool is_graph = 1;

  printf("problem: %s\n", input_filename);
    sparse_matrix_processor<custom_idx, double> processor(input_filename);
    
    factorization_driver<custom_idx, double>(processor, num_threads, output_filename, is_graph, jl_cols, solution);
}

// unflattens a flattened vector into a vec of vecs that can be passed to the solver. used for jl sketch
std::vector<std::vector<double>> unroll_vector(FlattenedVec shared_jl_cols) {

    int n = shared_jl_cols.num_cols;
    int m = shared_jl_cols.num_rows;
    
    std::vector<std::vector<double>> jl_cols(n, std::vector<double>(m, 0.0));
    
    int counter = 0;
    for (double s: shared_jl_cols.vec) {
        int current_column = (int) counter / n;
        int current_row = counter % n;
        jl_cols.at(current_row).at(current_column) = s;
        counter += 1;
    }

    // printf("value 1: %f value 2: %f\n", jl_cols.at(0).at(501), jl_cols.at(1).at(501));
    // printf("value 1: %d value 2: %d\n", jl_cols.at(0).at(501), jl_cols.at(1).at(501));
    // for (int i = 0; i < 10; i++) {
    //     printf("%f\n", jl_cols.at(0).at(i));
    // }    
    // for (int i = 0; i < 10; i++) {
    //     printf("%f\n", jl_cols.at(1).at(i));
    // }

    return jl_cols;
}

//flattens a vector of vectors into a single vector. used to pass the solution back to the rust code.
FlattenedVec flatten_vector(std::vector<std::vector<double>> original) {
    size_t n = original.size();
    size_t m = original.at(0).size();
    rust::cxxbridge1::Vec<double> values = {};
    for (auto col: original) {
        for (auto i: col) {
            values.push_back(i);
        }
    }
    FlattenedVec output = {values, n, m};
    return output;
}

//test function that passes a jl sketch from rust, solves for it on the physics dataset, and sends the solution back to the rust code.
FlattenedVec go(FlattenedVec shared_jl_cols) {
    int n = shared_jl_cols.num_cols;
    int m = shared_jl_cols.num_rows;
    std::vector<std::vector<double>> jl_cols = unroll_vector(shared_jl_cols);
    std::vector<std::vector<double>> solution(n, std::vector<double>(m, 0.0));
    run_solve(jl_cols, solution);
    FlattenedVec flat_solution = flatten_vector(solution);
    return flat_solution;
}

// test function that constructs a sparse matrix object from column pointer, row index, and value vectors sent from the the rust code.
void sprs_test(rust::Vec<size_t> rust_col_ptrs, rust::Vec<size_t> rust_row_indices, rust::Vec<double> rust_values) {
//void sprs_test(rust_col_ptrs: rust::Vec<size_t>, rust_row_indices: rust::Vec<size_t>, rust_values: rust::Vec<double>) {
    std::vector<double> values;
    std::copy(rust_values.begin(), rust_values.end(), std::back_inserter(values));
    std::vector<size_t> col_ptrs;
    std::copy(rust_col_ptrs.begin(), rust_col_ptrs.end(), std::back_inserter(col_ptrs));
    std::vector<size_t> row_indices;
    std::copy(rust_row_indices.begin(), rust_row_indices.end(), std::back_inserter(row_indices));
    std::cout << col_ptrs.size() << std::endl;
    for (auto i: values) {
        std::cout << i << " , ";
    }
    std::cout << std::endl;
    for (auto i: col_ptrs) {
        std::cout << i << ", ";
    }
    std::cout << std::endl;
    for (auto i: row_indices) {
        std::cout << i << ", ";
    }
    std::cout << std::endl;
    size_t num_rows = 3;
    size_t num_cols = 3;
    custom_space::sparse_matrix tester = custom_space::sparse_matrix(num_rows, num_cols, std::move(values), std::move(row_indices), std::move(col_ptrs));
    tester.printCSC();
    printf("%d\n", tester.nonZeros());
}

// test function that constructs a sparse matrix object of the correct data types for running the solver code. BROKEN
void sprs_correctness_test(rust::Vec<custom_idx> rust_col_ptrs, rust::Vec<custom_idx> rust_row_indices, rust::Vec<double> rust_values) {
    std::vector<double> values;
    std::copy(rust_values.begin(), rust_values.end(), std::back_inserter(values));
    std::vector<custom_idx> col_ptrs;
    std::copy(rust_col_ptrs.begin(), rust_col_ptrs.end(), std::back_inserter(col_ptrs));
    std::vector<custom_idx> row_indices;
    std::copy(rust_row_indices.begin(), rust_row_indices.end(), std::back_inserter(row_indices));
    // printf("phys col_ptrs size in c++: %d. first value: %d\n", col_ptrs.size(), col_ptrs.at(0));
    // printf("phys row_indices size in c++: %d. first value: %d\n", row_indices.size(), row_indices.at(0));
    // printf("phys values size in c++: %d. first value: %f\n", values.size(), values.at(0));

    // printf("type of col_ptrs: ");
    // std::cout << typeid(col_ptrs.at(0)).name() << std::endl;
    
    custom_idx num_rows = 525826;
    custom_idx num_cols = 525826;
    std::string name = "placeholder_sparse_matrix_processor_name";
    //sparse_matrix tester = sparse_matrix(name, num_rows, num_cols, std::move(col_ptrs), std::move(row_indices), std::move(values));
    sparse_matrix_processor<custom_idx, double> tester = sparse_matrix_processor(name, num_rows, num_cols, std::move(col_ptrs), std::move(row_indices), std::move(values));
    printf("nonzeros in csc: %d\n", tester.mat.nonZeros());
    //std::cout << "Size of int: " << sizeof(int) * 8 << " bits" << std::endl;
}


// test function for ensuring that flattenedvec data is passed appropriately
FlattenedVec test_roll(FlattenedVec jl_cols) {
    std::vector<std::vector<double>> unrolled = unroll_vector(jl_cols);
    FlattenedVec rerolled = flatten_vector(unrolled);
    return rerolled;
}

// function that runs the solver code on rust-provided laplacian and jl sketch.
FlattenedVec run_solve_lap(FlattenedVec shared_jl_cols, rust::Vec<custom_idx> rust_col_ptrs, \
    rust::Vec<custom_idx> rust_row_indices, rust::Vec<double> rust_values, int num_nodes) {

    //constexpr const char *input_filename = "/global/u1/d/dtench/cholesky/Parallel-Randomized-Cholesky/physics/parabolic_fem/parabolic_fem-nnz-sorted.mtx";
    int num_threads = 32; 
    constexpr char *output_filename = "";
    bool is_graph = 1;

    std::vector<double> values;
    std::copy(rust_values.begin(), rust_values.end(), std::back_inserter(values));
    std::vector<custom_idx> col_ptrs;
    std::copy(rust_col_ptrs.begin(), rust_col_ptrs.end(), std::back_inserter(col_ptrs));
    std::vector<custom_idx> row_indices;
    std::copy(rust_row_indices.begin(), rust_row_indices.end(), std::back_inserter(row_indices));

    custom_idx num_rows = num_nodes;
    custom_idx num_cols = num_nodes;
    //printf("num rows: %d\n", col_ptrs.size()-1);
    std::string input_filename = "placeholder_sparse_matrix_processor_name";
    sparse_matrix_processor<custom_idx, double> processor = sparse_matrix_processor(input_filename, num_rows, num_cols, std::move(col_ptrs), std::move(row_indices), std::move(values));

    
    custom_idx n = shared_jl_cols.num_cols;
    custom_idx m = shared_jl_cols.num_rows;
    std::vector<std::vector<double>> jl_cols = unroll_vector(shared_jl_cols);
    std::vector<std::vector<double>> solution(n, std::vector<double>(m, 0.0));

    printf("problem: %s\n", input_filename.c_str());
    //sparse_matrix_processor<custom_idx, double> processor(input_filename);
    
    factorization_driver<custom_idx, double>(processor, num_threads, output_filename, is_graph, jl_cols, solution);

    FlattenedVec flat_solution = flatten_vector(solution);
    return flat_solution;
}


std::vector<std::vector<double>> read_sketch_from_csv(const std::string& filename) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    bool first = true;
    int row_length = 0;

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> row;
        std::string token;

        while (std::getline(iss, token, ',')) {
            try {
                row.push_back(std::stod(token));
            } catch (const std::invalid_argument& e) {
                throw std::runtime_error("Invalid float value: " + token);
            } catch (const std::out_of_range& e) {
                throw std::runtime_error("Float value out of range: " + token);
            }
        }

        if (first) {
            row_length = row.size();
            first = false;
        }
        else {
            assert(row_length == row.size());
        }

        data.push_back(row);
    }

    file.close();
    printf("jl matrix has %d rows and %d cols\n", data.size(), row_length);
    return data;
}

// ------------------------------------------------------------------
// Helper: split a line on commas (naïve – no quoting support)
// ------------------------------------------------------------------
static std::vector<std::string> split_csv_line(const std::string& line)
{
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ',')) {
        fields.emplace_back(std::move(cell));
    }
    return fields;
}

// ------------------------------------------------------------------
// Helper: convert a string to double
// ------------------------------------------------------------------
static double parse_double(const std::string& s, bool strict = true)
{
    // Trim leading/trailing whitespace (optional but nice)
    const auto first = s.find_first_not_of(" \t\r\n");
    const auto last  = s.find_last_not_of(" \t\r\n");
    const std::string trimmed = (first == std::string::npos) ? "" : s.substr(first, last - first + 1);

    if (trimmed.empty()) {
        // Empty field – treat as NaN in lenient mode, otherwise error
        if (strict) {
            throw std::invalid_argument("Empty field cannot be parsed as double");
        }
        return std::numeric_limits<double>::quiet_NaN();
    }

    try {
        size_t idx = 0;
        double val = std::stod(trimmed, &idx);
        // Ensure the whole token was consumed (e.g. "12abc" is an error)
        if (idx != trimmed.size()) {
            throw std::invalid_argument("Trailing characters after number");
        }
        return val;
    }
    catch (const std::exception&) {
        if (strict) {
            throw; // re‑throw the original conversion error
        }
        // Lenient: return NaN for any unparsable token
        return std::numeric_limits<double>::quiet_NaN();
    }
}

// ------------------------------------------------------------------
// Load CSV into column‑wise storage of doubles
// ------------------------------------------------------------------
std::vector<std::vector<double>> load_csv_columns(const std::string& filename,
                                                             bool strict = true)
{
    std::ifstream infile(filename);
    if (!infile) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    std::vector<std::vector<double>> columns;   // outer vector = columns

    // --------------------------------------------------------------
    // 1️⃣ Read the first line → discover column count and fill first row
    // --------------------------------------------------------------
    if (!std::getline(infile, line)) {
        // Empty file → return empty container
        return columns;
    }

    std::vector<std::string> firstRowStr = split_csv_line(line);
    const std::size_t nCols = firstRowStr.size();

    // Allocate one inner vector per column
    columns.resize(nCols);
    for (std::size_t i = 0; i < nCols; ++i) {
        columns[i].push_back(parse_double(firstRowStr[i], strict));
    }

    // --------------------------------------------------------------
    // 2️⃣ Process the remaining lines, appending to each column vector
    // --------------------------------------------------------------
    std::size_t lineNumber = 1; // we already processed line 1
    while (std::getline(infile, line)) {
        ++lineNumber;
        std::vector<std::string> rowStr = split_csv_line(line);

        if (rowStr.size() != nCols) {
            throw std::runtime_error(
                "Inconsistent column count on line " + std::to_string(lineNumber) +
                ". Expected " + std::to_string(nCols) + " but got " + std::to_string(rowStr.size()));
        }

        for (std::size_t i = 0; i < nCols; ++i) {
            columns[i].push_back(parse_double(rowStr[i], strict));
        }
    }

    return columns;
}

void julia_test_solve(FlattenedVec interop_jl_cols, rust::Vec<custom_idx> rust_col_ptrs, rust::Vec<custom_idx> rust_row_indices, rust::Vec<double> rust_values, int num_nodes) {
  constexpr const char *input_filename = "../tianyu-stream/data/virus_lap_tianyu.mtx";
  std::string sketch_filename = "../tianyu-stream/data/virus_sketch_tianyu.csv";

  int num_threads = 32; 
  constexpr char *output_filename = "";
  bool is_graph = 1;

  std::vector<std::vector<double>> file_jl_cols;

  try {
    file_jl_cols = load_csv_columns(sketch_filename);
  }
  catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }

  int num_rows = file_jl_cols[0].size();
  int num_cols = file_jl_cols.size();
  printf("file jl matrix has %d rows and %d cols\n", num_rows, num_cols);

  std::vector<std::vector<double>> unrolled_interop_jl_cols = unroll_vector(interop_jl_cols);
  printf("interop jl matrix has %d rows and %d cols\n", unrolled_interop_jl_cols[0].size(), unrolled_interop_jl_cols.size());

  for (int i = 0; i < num_cols; i++) {
    for (int j = 0; j < num_rows; j++) {
        assert(file_jl_cols.at(i).at(j) == unrolled_interop_jl_cols.at(i).at(j));
    }
  }
  printf("jl sketch test passed: jl sketch passed through interop is equiv to the one in the file.\n");

// now we test lap for equivalence

  sparse_matrix_processor<custom_idx, double> file_processor(input_filename);
  std::string name = "interop_processor";
  std::vector<double> values;
  std::copy(rust_values.begin(), rust_values.end(), std::back_inserter(values));
  std::vector<custom_idx> col_ptrs;
  std::copy(rust_col_ptrs.begin(), rust_col_ptrs.end(), std::back_inserter(col_ptrs));
  std::vector<custom_idx> row_indices;
  std::copy(rust_row_indices.begin(), rust_row_indices.end(), std::back_inserter(row_indices));
  sparse_matrix_processor<custom_idx, double> interop_processor(name, num_rows, num_rows, std::move(col_ptrs), std::move(row_indices), std::move(values));

  for (int i = 0; i < 5; i++) {
    printf("col %d: file has %d and interop has %d\n", i, file_processor.mat.col_ptrs.at(i), rust_col_ptrs.at(i));
  }
  for (int i = 0; i < file_processor.mat.col_ptrs.size(); i++) {
    assert(file_processor.mat.col_ptrs.at(i) == interop_processor.mat.col_ptrs.at(i));
  }

  for (int i = 0; i < file_processor.mat.row_indices.size(); i++) {
    assert(file_processor.mat.row_indices.at(i) == interop_processor.mat.row_indices.at(i));
    assert(file_processor.mat.values.at(i) == interop_processor.mat.values.at(i));
  }
  printf("lap test passed: laplacian passed through interop is equiv to the one in the file.\n");
}

// reads in jl sketch and lap from file (produced by tianyu julia code) and solves. status: WORKS (but figure out how to check solution for good quality later)
bool file_only_solver_test(std::vector<std::vector<double>> jl_cols) {
  printf("-------------------------------------\n");
  printf("performing file_only_solver_test\n");
  printf("-------------------------------------\n");
  constexpr const char *input_filename = "../tianyu-stream/data/virus_lap_tianyu.mtx";
  int num_threads = 32; 
  constexpr char *output_filename = "output/file_only.txt";
  bool is_graph = 1;

  custom_idx num_cols = jl_cols.size();
  custom_idx num_rows = jl_cols.at(0).size();
  std::vector<std::vector<double>> solution(num_cols, std::vector<double>(num_rows, 0.0));

  printf("problem: %s\n", input_filename);
  sparse_matrix_processor<custom_idx, double> processor(input_filename);
    
  bool result = factorization_driver<custom_idx, double>(processor, num_threads, output_filename, is_graph, jl_cols, solution);
  printf("file_only_solver_test done. if the solves converged, the test passed.\n");
  return result;
}

// this test establishes that the jl sketches from file and interop are the same. status: WORKS
bool jl_file_interop_equiv_test(std::vector<std::vector<double>> file_jl_cols, std::vector<std::vector<double>> interop_jl_cols) {
  printf("-------------------------------------\n");
  printf("performing jl_file_interop_equiv_test\n");
  printf("-------------------------------------\n");
  int num_cols = file_jl_cols.size();
  int num_rows = file_jl_cols.at(0).size();

  printf("verifying the sketch matrices have the same dimensions: ");
  assert(num_cols = interop_jl_cols.size());
  assert(num_rows = interop_jl_cols.at(0).size());
  printf("verified.\n");

  printf("verifying the sketch matrices have the same entries: ");
  for (int i = 0; i < num_cols; i++) {
    for (int j = 0; j < num_rows; j++) {
        assert(file_jl_cols.at(i).at(j) == interop_jl_cols.at(i).at(j));
    }
  }
  printf("verified.\n");
  printf("jl_file_interop_equiv_test passed: jl sketch passed through interop is equivalent to the one in the file.\n");
  return true;
}

// this test tries to solve with jl sketch from interop and lap from direct file read (in tianyu's sparse matrix processor code). status: WORKS
bool jl_interop_lap_file_solver_test(std::vector<std::vector<double>> jl_cols) {
  printf("-------------------------------------\n");
  printf("performing jl_interop_lap_file_solver_test\n");
  printf("-------------------------------------\n");
  constexpr const char *input_filename = "../tianyu-stream/data/virus_lap_tianyu.mtx";
  int num_threads = 32; 
  constexpr char *output_filename = "output/jl_int_lap_file.txt";
  bool is_graph = 1;

  custom_idx num_cols = jl_cols.size();
  custom_idx num_rows = jl_cols.at(0).size();
  std::vector<std::vector<double>> solution(num_cols, std::vector<double>(num_rows, 0.0));

  printf("problem: %s\n", input_filename);
  sparse_matrix_processor<custom_idx, double> processor(input_filename);
    
  bool result = factorization_driver<custom_idx, double>(processor, num_threads, output_filename, is_graph, jl_cols, solution);
  printf("jl_interop_lap_file_solver_test done. if the solves converged, the test passed.\n");
  return result;
}

// tests whether the file and interop laplacians are equivalent. status: WORKS
bool lap_equiv_test(std::vector<custom_idx> interop_col_ptrs, std::vector<custom_idx> interop_row_indices, std::vector<double> interop_values, int num_nodes) {
  printf("-------------------------------------\n");
  printf("performing lap_equiv_test:\n");
  printf("-------------------------------------\n");
  constexpr const char *input_filename = "../tianyu-stream/data/virus_lap_tianyu.mtx";
  sparse_matrix_processor<custom_idx, double> file_processor(input_filename);
  printf("sparse matrix processor from file DONE building.\n");

//   for (int i = 0; i < 5; i++) {
//     printf("col ptrs in col %d: file has %d and interop has %d\n", i, file_processor.mat.col_ptrs.at(i), interop_col_ptrs.at(i));
//     printf("row idxs in col %d: file has %d and interop has %d\n", i, file_processor.mat.row_indices.at(i), interop_row_indices.at(i));
//     printf("values in col %d: file has %f and interop has %f\n", i, file_processor.mat.values.at(i), interop_values.at(i));
//   }

  std::string name = "interop_processor";
  sparse_matrix_processor<custom_idx, double> interop_processor(name, num_nodes, num_nodes, std::move(interop_col_ptrs), std::move(interop_row_indices), std::move(interop_values));
  printf("sparse matrix processor from interop DONE building.\n");


  for (int i = 0; i < file_processor.mat.col_ptrs.size(); i++) {
    assert(file_processor.mat.col_ptrs.at(i) == interop_processor.mat.col_ptrs.at(i));
  }

  for (int i = 0; i < file_processor.mat.row_indices.size(); i++) {
    assert(file_processor.mat.row_indices.at(i) == interop_processor.mat.row_indices.at(i));
  }
  printf("the two laplacians have the same sparsity pattern.\n");

  double allowed_error = 0.000001;
  for (int i = 0; i < file_processor.mat.row_indices.size(); i++) {
    assert(abs(file_processor.mat.values.at(i) - interop_processor.mat.values.at(i)) < allowed_error);
  }
  printf("lap_equi_test passed: laplacian passed through interop is equivalent to the one in the file.\n");
  return true;
}

// this test tries to solve with jl sketch and lap both from interop. status: WORKS
bool interop_only_solver_test(std::vector<std::vector<double>> jl_cols, std::vector<custom_idx> interop_col_ptrs, std::vector<custom_idx> interop_row_indices, std::vector<double> interop_values, int num_nodes) {
  printf("-------------------------------------\n");
  printf("performing interop_only_solver_test\n");
  printf("-------------------------------------\n");
  constexpr const char *input_filename = "../tianyu-stream/data/virus_lap_tianyu.mtx";
  int num_threads = 32; 
  constexpr char *output_filename = "output/int_only.txt";
  bool is_graph = 1;

  custom_idx num_cols = jl_cols.size();
  custom_idx num_rows = jl_cols.at(0).size();
  std::vector<std::vector<double>> solution(num_cols, std::vector<double>(num_rows, 0.0));

  printf("problem: %s\n", input_filename);
  std::string name = "interop_processor";
  sparse_matrix_processor<custom_idx, double> processor(name, num_nodes, num_nodes, std::move(interop_col_ptrs), std::move(interop_row_indices), std::move(interop_values));
   
  bool result = factorization_driver<custom_idx, double>(processor, num_threads, output_filename, is_graph, jl_cols, solution);
  printf("interop_only_solver_test done. if the solves converged, the test passed.\n");
  return result;
}

// this function reads jl sketch and lap info from rust via interop. intended to be used to handle boilerplate unwrapping, 
// then you call the specific test you want from it.
bool test_stager(FlattenedVec interop_jl_cols, rust::Vec<int> rust_col_ptrs, rust::Vec<int> rust_row_indices, rust::Vec<double> rust_values, int num_nodes, int test_selector) {
  constexpr const char *input_filename = "../tianyu-stream/data/virus_lap_tianyu.mtx";
  std::string sketch_filename = "../tianyu-stream/data/virus_sketch_tianyu.csv";

  // stage jl cols from file
  std::vector<std::vector<double>> file_jl_cols;
  try {
    file_jl_cols = load_csv_columns(sketch_filename);
  }
  catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }

  // stage jl cols from interop
  std::vector<std::vector<double>> unrolled_interop_jl_cols = unroll_vector(interop_jl_cols);

  // stage lap from interop (note that lap from file is handled in tianyu's sparse matrix processor code directly)
  std::vector<double> lap_values;
  std::copy(rust_values.begin(), rust_values.end(), std::back_inserter(lap_values));
  std::vector<custom_idx> lap_col_ptrs;
  std::copy(rust_col_ptrs.begin(), rust_col_ptrs.end(), std::back_inserter(lap_col_ptrs));
  std::vector<custom_idx> lap_row_indices;
  std::copy(rust_row_indices.begin(), rust_row_indices.end(), std::back_inserter(lap_row_indices));

  // at this point you have whatever variables you need to pass into various test functions.
  // example calls:
  bool result;
  switch (test_selector) {
    case 1:
        result = file_only_solver_test(file_jl_cols);
        break;
    case 2:
        result = jl_file_interop_equiv_test(file_jl_cols, unrolled_interop_jl_cols);
        break;
    case 3:
        result = jl_interop_lap_file_solver_test(unrolled_interop_jl_cols);
        break;
    case 4:
        result = lap_equiv_test(lap_col_ptrs, lap_row_indices, lap_values, num_nodes);
        break;
    case 5: 
        result = interop_only_solver_test(unrolled_interop_jl_cols, lap_col_ptrs, lap_row_indices, lap_values, num_nodes);
        break;
    default:
        result = false;
        break;
  }
  return result;
}