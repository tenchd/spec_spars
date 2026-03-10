using Laplacians
using MatrixMarket
using SparseArrays
using Plots
using Random
using Metis
using Statistics
using Graphs
using GraphsFlows
using KrylovKit
using DelimitedFiles
#using MKLSparse
#pyplot()
include("solving_with_sketch.jl")
include("leverage_score_approx.jl")


function calculate_cut_weight(as, partition)
    # calculate the cut weight
    colptr = as.colptr
    rowval = as.rowval
    nzval = as.nzval
    total_weight = 0.0
    for c = 1 : length(colptr) - 1
        for r = colptr[c] : colptr[c + 1] - 1
            if partition[c] != partition[rowval[r]]
                total_weight += nzval[r]
            end
            
        end 
    end
  
    return total_weight
end

function remove_diagonal(G)
    I = Vector{Int64}()
    J = Vector{Int64}()
    V = Vector{Float64}()
    for i = 1 : length(G.colptr) - 1
        for j = G.colptr[i] : G.colptr[i + 1] - 1
            if G.rowval[j] != i
                push!(I, G.rowval[j])
                push!(J, i)
                push!(V, G.nzval[j])
            end
        end
    end
    G = sparse(I,J,V)
    return G
end



function Stream_Sparsify(mat::String; eps=1e-1, matrixConcConst=4.0, num_sample=1)
    # read in the matrix and stream it
    G = MatrixMarket.mmread(mat)
    G = SparseMatrixCSC{Float64, Int64}(G)
    G = remove_diagonal(G)
    if !issymmetric(G)
        G = tril(G) + tril(G)'
    end
    #G.nzval[:] .= 1.0 # set all nonzeros to 1
    G.nzval .= abs.(G.nzval)
    
    n = size(G, 1)

    trimmed_list = Vector{Int64}()
    
    desired_capacity = 4000 * n * log(n) / eps^2
    println("desired capacity: ", desired_capacity)
    
    (ai,aj,av) = findnz(triu(G))
    shuffle_idx = randperm(length(ai))
    ai = ai[shuffle_idx]
    aj = aj[shuffle_idx]
    av = av[shuffle_idx]

    total_edge = length(av)
    #cutoff = max(round(Int, log(n) * n), round(Int, total_edge / 2))
    #cutoff = round(Int, total_edge / 2)
    #cutoff = min(cutoff, total_edge)
    cutoff = total_edge
    #matrixConcConst = cutoff / 20 / n / log(n) * eps ^ 2
    println("matrix constant: ", matrixConcConst)
    left_edge_list = Vector{Int64}()
    right_edge_list = Vector{Int64}()
    val = Vector{Float64}()
    decision_vec = zeros(Float64, cutoff)

    println("number of vertices: ", n)
    println("cutoff: ", cutoff)
    println("total edges: ", total_edge)
    iter = 1
    total_time_list = zeros(3)
    overall_time = 0
    for i = 1 : total_edge
        push!(left_edge_list, ai[i])
        push!(right_edge_list, aj[i])
        push!(val, av[i])
        
        if length(left_edge_list) == cutoff
            
            # sort the indices
            comb_list = [[left_edge_list[i], right_edge_list[i]] for i in 1 : cutoff]
            perm1 = sortperm(comb_list, by = x -> (x[2], x[1]))
            left_edge_list = left_edge_list[perm1]
            right_edge_list = right_edge_list[perm1]
            val = val[perm1]


            # create sparse adjacency graph
            as = sparse(left_edge_list, right_edge_list, val, n, n)
            as = as + as'

            t1 = time()
            prs_blocked, partial_time_list = l_sparsify_blocked(as, matrixConcConst=matrixConcConst, ep=eps, first = true)
            overall_time += (time() - t1)
            total_time_list .= total_time_list .+ partial_time_list
            rand!(decision_vec)
            println("min of prs: ", minimum(abs.(prs_blocked)))
            ind = decision_vec .< prs_blocked

            left_edge_list = left_edge_list[ind]
            right_edge_list = right_edge_list[ind]
            val = val[ind]
            prs_blocked = prs_blocked[ind]

            val ./= prs_blocked    
            
            num_trimmed = cutoff - length(val)
            println("iter: ", iter)
            println("element progress index: ", i)
            println("num trimmed: ", num_trimmed)
            push!(trimmed_list, num_trimmed)
            if num_trimmed == 0
                println("ALGORITHM STAGNATED")
            end
            
            iter += 1
        end
    end
    println("\n\n\n\n\n")
    println(trimmed_list)
    println("overall_time: ", overall_time)

    println("length: ", length(val))
    println("factor time: ", total_time_list[1], " sketch time: ", total_time_list[2], " solve and compute time: ", total_time_list[3])
    println(total_time_list)

    # create sparse matrix from the resulting edges
    as = sparse(left_edge_list, right_edge_list, val, n, n)
    as = as + as'


    # write matrix or further processing
    # MatrixMarket.mmwrite("output_sparse_mat/sparse_" * mat, as)


    # compute cut weight
    original_cut_2 = Metis.partition(G, 2)
    original_cut_weight_2 = calculate_cut_weight(G, original_cut_2)
    println("cut weight for 2 partitions (original): ", original_cut_weight_2)

    stream_cut_2 = Metis.partition(as, 2)
    stream_cut_weight_2 = calculate_cut_weight(G, stream_cut_2)
    println("cut weight for 2 partitions (stream): ", stream_cut_weight_2)

    prs_blocked, partial_time_list = l_sparsify_blocked(G, matrixConcConst=matrixConcConst, ep=eps)
    (ai,aj,av) = findnz(triu(G))
    ind = rand(Float64, size(prs_blocked)) .< prs_blocked
    non_stream_G = sparse(ai[ind], aj[ind], av[ind] ./ prs_blocked[ind], n, n)
    non_stream_G = non_stream_G + non_stream_G'
    non_stream_cut_2 = Metis.partition(non_stream_G, 2)
    non_stream_cut_weight_2 = calculate_cut_weight(G, non_stream_cut_2)
    println("cut weight for 2 partitions (non stream): ", non_stream_cut_weight_2)

    #ind = rand(Bool, length(ai)) 
    ind = rand(length(ai)) .<= 0.5
    random_G = sparse(ai[ind], aj[ind], av[ind] / 0.5, n, n)
    random_G = random_G + random_G'
    random_G_cut_2 = Metis.partition(random_G, 2)
    random_G_cut_weight_2 = calculate_cut_weight(G, random_G_cut_2)
    println("cut weight for 2 partitions (random): ", random_G_cut_weight_2)


    # compute quadratic form estimates
    #stream_ratio, non_stream_ratio, random_ratio = compute_quadratic_ratio(G, as, non_stream_G, random_G)
    #println("stream ratio average: ", mean(stream_ratio))
    #println("non stream ratio average: ", mean(non_stream_ratio))
    #println("random ratio average: ", mean(random_ratio))
    
    #println(stream_ratio)
    #println(non_stream_ratio)
    #println(random_ratio)
    L1 = Graph(G)
    L2 = Graph(as)
    L3 = Graph(non_stream_G)
    L4 = Graph(random_G)

    CL1 = connected_components(L1)
    CL2 = connected_components(L2)
    CL3 = connected_components(L3)
    CL4 = connected_components(L4)

    println("number of connected components in original graph: ", length(CL1))
    println("number of connected components in streamed graph: ", length(CL2))
    println("number of connected components in non-streamed graph: ", length(CL3))
    println("number of connected components in random graph: ", length(CL4))

    #=
    f1 = Vector{Float64}()
    f2 = Vector{Float64}()
    f3 = Vector{Float64}()
    f4 = Vector{Float64}()
    DL1 = DiGraph(L1)
    DL2 = DiGraph(L2)
    DL3 = DiGraph(L3)
    DL4 = DiGraph(L4)
    for i = 1 : num_sample
        z1 = time()
        two_points = [-1, -1]
        while two_points[1] == two_points[2]
            two_points = rand(1 : n, 2)
        end
        GC.gc()
    
        a1 = maximum_flow(DL1, two_points[1], two_points[2], G, algorithm=BoykovKolmogorovAlgorithm())
        a2 = maximum_flow(DL2, two_points[1], two_points[2], as, algorithm=BoykovKolmogorovAlgorithm())
        a3 = maximum_flow(DL3, two_points[1], two_points[2], non_stream_G, algorithm=BoykovKolmogorovAlgorithm())
        a4 = maximum_flow(DL4, two_points[1], two_points[2], random_G, algorithm=BoykovKolmogorovAlgorithm())
        
        push!(f1, a1[1])
        push!(f2, a2[1])
        push!(f3, a3[1])
        push!(f4, a4[1])
        println("flow iteration: ", i)
        println("iter time: ", time() - z1)
    end
    println("flow f1: ", f1)
    println("flow f2: ", f2)
    println("flow f3: ", f3)
    println("flow f4: ", f4)
    

    println("average stretch factor maximum flow streaming: ", mean(filter(x -> x != 0, f2 ./ f1)))
    println("average stretch factor maximum flow non-streaming: ", mean(filter(x -> x != 0, f3 ./ f1)))
    println("average stretch factor maximum flow random: ", mean(filter(x -> x != 0, f4 ./ f1)))
    =#

    # rand_idx = randperm(size(G, 1))[1 : num_sample]
    # ecc1 = compute_eccentricity(L1, rand_idx, G)
    # ecc2 = compute_eccentricity(L2, rand_idx, as)
    # ecc3 = compute_eccentricity(L3, rand_idx, non_stream_G)
    # ecc4 = compute_eccentricity(L4, rand_idx, random_G)

    # ecc4_index = findall(x -> (x < 9223372036854775807), ecc4)
    # println(ecc4)
    # println(ecc4_index)

    # println("average eccentricity stretch ratio for streaming: ", mean(ecc2 ./ ecc1))
    # println("average eccentricity stretch ratio for non-streaming: ", mean(ecc3 ./ ecc1))
    # println("average eccentricity stretch ratio for random: ", mean(ecc4[ecc4_index] ./ ecc1[ecc4_index]))



    return L1, L2, L3, L4
end

function compute_quadratic_ratio(original_G, stream_G, non_stream_G, random_G)
    n = size(original_G, 1)
    original_L = lap(original_G)
    stream_L = lap(stream_G)
    non_stream_L = lap(non_stream_G)
    random_L = lap(random_G)
    println("nnz of original: ", nnz(original_G))
    println("nnz of stream: ", nnz(stream_G))
    println("nnz of non stream: ", nnz(non_stream_G))
    println("nnz of random: ", nnz(random_G))

    original_val = Vector{Float64}()
    stream_ratio = Vector{Float64}()
    non_stream_ratio = Vector{Float64}()
    random_ratio = Vector{Float64}()

    # x = rand(n)
    # for i = 1 : 3000
    #     x = original_L * x;
    #     x .= x ./ norm(x)
    # end
    x_set = eigsolve(original_L, min(20, n), :SR)[2]
    for i = 1 : min(20, n)
        
        x = x_set[i]
    
        push!(original_val, x' * original_L * x)
        push!(stream_ratio, (x' * stream_L * x) / original_val[end])
        push!(non_stream_ratio, (x' * non_stream_L * x) / original_val[end])
        push!(random_ratio, (x' * random_L * x) / original_val[end])
    end
    
    println("diag norm diff of stream: ", norm(diag(original_L) - diag(stream_L)))
    println("diag norm diff of non-stream: ", norm(diag(original_L) - diag(non_stream_L)))
    println("diag norm diff of random: ", norm(diag(original_L) - diag(random_L)))
        
   
    println(original_val)
    return stream_ratio, non_stream_ratio, random_ratio
end

function compute_eccentricity(G, samples, G_weight=weights(G))
    println(typeof(G_weight))
    return eccentricity(G, samples, G_weight)

end
println("-------------------------------------------------------------------")
Stream_Sparsify("/global/cfs/cdirs/m1982/david/bulk_to_process/virus/virus.mtx", eps=5e-1)
println("-------------------------------------------------------------------")
#Stream_Sparsify("/global/cfs/cdirs/m1982/david/bulk_to_process/mouse_gene/mouse_gene.mtx", eps=5e-1)
println("-------------------------------------------------------------------")
#Stream_Sparsify("/global/cfs/cdirs/m1982/david/bulk_to_process/human_gene1/human_gene1.mtx", eps=5e-1)
println("-------------------------------------------------------------------")
#Stream_Sparsify("/global/cfs/cdirs/m1982/david/bulk_to_process/human_gene2/human_gene2.mtx", eps=5e-1)
println("-------------------------------------------------------------------")
