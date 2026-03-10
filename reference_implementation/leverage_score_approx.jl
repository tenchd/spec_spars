function l_sparsify(a; ep=0.3, matrixConcConst=4.0, JLfac=4.0)

  if ep > 1
    @warn "Calling sparsify with ep > 1 can produce a disconnected graph."
  end

  @time f = approxchol_lap(a,tol=1e-2,params=ApproxCholParams(:wdeg));

  n = size(a,1)
  k = round(Int, JLfac*log(n)) # number of dims for JL

  U = wtedEdgeVertexMat(a)
  m = size(U,1)
  R = randn(Float64, m,k)


  BLAS.set_num_threads(Threads.nthreads())
  println(size(U))
  println(size(R))
  UR = zeros(n, k)
  @time mul!(UR, U', R)
  BLAS.set_num_threads(1)
  UR .= UR ./ sqrt(n)

  V = zeros(k, n)
  @time solve_with_lap(V, UR, f)

  (ai,aj,av) = findnz(triu(a))
  prs = zeros(size(av))

  @time compute_diff_norm(prs, length(av), V, ai, aj)

  @time @inbounds @tturbo for h in 1:length(av)
      prs[h] = av[h] * (prs[h]^2 / k) * matrixConcConst *log(n) / ep^2
  end

  @time @inbounds for h in 1:length(av)
      prs[h] = min(prs[h], 1)
  end
  return prs

end

#vibe-coded function to sort U by column nonzero index pairs, lexicographically
function sort_columns_by_two_rows(M::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
  ncol = size(M, 2)

  # ----------------------------------------------------------------------
  # 1️⃣  Check that every column contains exactly two entries
  # ----------------------------------------------------------------------
  for j in 1:ncol
      if M.colptr[j+1] - M.colptr[j] != 2
          bad_nonzeros = M.colptr[j+1] - M.colptr[j]
          throw(ArgumentError("Column $j does not contain exactly two non‑zeros. instead, $bad_nonzeros"))
      end
  end

  # ----------------------------------------------------------------------
  # 2️⃣  Extract the two row‑indices for each column (already sorted in CSC)
  # ----------------------------------------------------------------------
  pairs = Vector{Tuple{Ti,Ti}}(undef, ncol)   # (row₁, row₂) for each column
  for j in 1:ncol
      i1 = M.colptr[j]           # first stored entry of column j
      i2 = i1 + 1                # second stored entry (exactly 2 per column)
      r1 = M.rowval[i1]
      r2 = M.rowval[i2]
      # CSC guarantees r1 < r2, but we keep the guard for safety
      if r1 > r2
          r1, r2 = r2, r1
      end
      pairs[j] = (r1, r2)
  end

  # ----------------------------------------------------------------------
  # 3️⃣  Sort columns lexicographically by these pairs
  # ----------------------------------------------------------------------
  perm = sortperm(pairs; stable=true)   # stable → equal pairs keep original order

  # ----------------------------------------------------------------------
  # 4️⃣  Apply permutation
  # ----------------------------------------------------------------------
  Msorted = M[:, perm]                  # cheap column reordering in CSC
  return Msorted
end

function sort_rows_by_two_columns(M::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    nrow, ncol = size(M)

    # ----------------------------------------------------------------------
    # 1️⃣  Allocate temporary storage for the two column indices of each row
    # ----------------------------------------------------------------------
    col1   = Vector{Ti}(undef, nrow)   # first column index of the row
    col2   = Vector{Ti}(undef, nrow)   # second column index of the row
    count  = zeros(Int, nrow)          # how many non‑zeros we have seen per row

    # ----------------------------------------------------------------------
    # 2️⃣  Scan the CSC structure column‑wise and fill `col1/col2`
    # ----------------------------------------------------------------------
    for j in 1:ncol                     # j = column index
        start = M.colptr[j]             # first entry of column j
        stop  = M.colptr[j+1] - 1       # last entry  of column j
        for k in start:stop            # iterate over stored rows of column j
            i = M.rowval[k]             # row index of the current non‑zero
            cnt = count[i] + 1
            count[i] = cnt
            if cnt == 1
                col1[i] = Ti(j)
            elseif cnt == 2
                col2[i] = Ti(j)
            else
                throw(ArgumentError("Row $i contains more than two non‑zeros"))
            end
        end
    end

    # ----------------------------------------------------------------------
    # 3️⃣  Verify the “exactly two per row” invariant and build the tuple list
    # ----------------------------------------------------------------------
    pairs = Vector{Tuple{Ti,Ti}}(undef, nrow)   # (col₁, col₂) for each row
    for i in 1:nrow
        if count[i] != 2
            throw(ArgumentError("Row $i does not contain exactly two non‑zeros"))
        end
        c1, c2 = col1[i], col2[i]
        # ensure the pair is ordered (CSC does *not* guarantee ordering inside rows)
        if c1 > c2
            c1, c2 = c2, c1
        end
        pairs[i] = (c1, c2)
    end

    # ----------------------------------------------------------------------
    # 4️⃣  Lexicographic sort of the row‑pairs → permutation vector
    # ----------------------------------------------------------------------
    perm = sortperm(pairs; alg = Base.Sort.MergeSort)   # stable keeps the original order for ties

    # ----------------------------------------------------------------------
    # 5️⃣  Apply the permutation (cheap row reordering)
    # ----------------------------------------------------------------------
    Msorted = M[perm, :]                  # re‑order rows
    return Msorted
end

function l_sparsify_blocked(a; ep=0.3, matrixConcConst=4.0, JLfac=4.0, first = false)

  if (first)
    println("writing laplacian to file.")
     MatrixMarket.mmwrite("/global/homes/d/dtench/m1982/david/spec_spars_files/julia_output/julia_lap.mtx", a)
     println("done writing laplacian.")
  end
  
  if ep > 1
    @warn "Calling sparsify with ep > 1 can produce a disconnected graph."
  end
  factorize_time_start = time()

  f = approxchol_lap(a,tol=1e-2,params=ApproxCholParams(:wdeg));

  factorize_time_end = time() - factorize_time_start

  n = size(a,1)
  k = round(Int, JLfac*log2(n)) # number of dims for JL
  println("number of nodes is $n, jl factor is $JLfac, log(n) is $(log2(n)), jl dimension is $k, epsilon is $ep")

  U = wtedEdgeVertexMat(a)
  U = sort_rows_by_two_columns(U)
  if (first)
    println("writing evim to file.")
    MatrixMarket.mmwrite("/global/homes/d/dtench/m1982/david/spec_spars_files/julia_output/julia_evim.mtx", U)
    println("done writing evim.")
  end
  m = size(U,1)
  #k = size(U,2)
  println("total edges is $m, total jl columns is $k")
  URt = zeros(k,size(U,2))
  println("size of URt: ", size(URt))

  #println(size(U))
  #println(size(URt))
  #println(size(k))
  sketch_time_start = time()
  multithread_blocked_dense_sparse!(U, URt, k, 50, 15000, first)
  sketch_time_end = time() - sketch_time_start
  #@time advanced_blocked_dense_sparse!(U, URt, k, 50000, 2000)
  UR = URt';
  UR .= UR .* sqrt(3)
  if (first)
    println("writing sketch product to file.")
    writedlm("/global/cfs/cdirs/m1982/david/spec_spars_files/julia_output/julia_sketch_product.csv", UR, ',')
    MatrixMarket.mmwrite("/global/homes/d/dtench/m1982/david/spec_spars_files/julia_output/julia_sketch_product.mtx", sparse(UR))
    println("done writing sketch product.")
  end

  V = zeros(k, n)

  solve_and_compute_start = time()
  solve_with_lap(V, UR, f)

  if (first)
    println("writing solution to file.")
    MatrixMarket.mmwrite("/global/homes/d/dtench/m1982/david/spec_spars_files/julia_output/julia_solution.mtx", sparse(V'))
    println("done writing solution.")
  end

  (ai,aj,av) = findnz(triu(a))
  prs = zeros(size(av))



compute_diff_norm(prs, length(av), V, ai, aj)
if (first)
    println("writing diff norms to file.")
    writedlm("/global/homes/d/dtench/m1982/david/spec_spars_files/julia_output/julia_diff_norms.csv", prs', ',')
    println("done writing diff norms.")
end

diffmean = mean(prs)
diffstd = std(prs)
println("the mean of all of the diff norms is $diffmean and the std dev is $diffstd")
#println("the mean of all of the diff norms is $diffmean")

@inbounds @turbo for h in 1:length(av)
    prs[h] = av[h] * (prs[h]^2 / k) * matrixConcConst *log(n) / ep^2
end

bigterm = matrixConcConst *log(n) / ep^2
println("big term: $bigterm")

@inbounds for h in 1:length(av)
    prs[h] = min(prs[h], 1)
end
if (first)
  println("writing probabilities to file.")
  writedlm("/global/homes/d/dtench/m1982/david/spec_spars_files/julia_output/julia_probs.csv", prs', ',')
  println("done writing probabilities.")
end

probmean = mean(prs)
probstd = std(prs)
println("the mean of all of the probabilities is $probmean and the std dev is $probstd")
#println("the mean of all of the probabilities is $probmean")

solve_and_compute_end = time() - solve_and_compute_start
#println("l1 norm: ", sum(prs))
time_list = Vector{Float64}()
push!(time_list, factorize_time_end)
push!(time_list, sketch_time_end)
push!(time_list, solve_and_compute_end)
return prs, time_list

end



function l_sparsify_blocked_pm1(a; ep=0.3, matrixConcConst=4.0, JLfac=7.0)

    if ep > 1
      @warn "Calling sparsify with ep > 1 can produce a disconnected graph."
    end
  
    @time f = approxchol_lap(a,tol=1e-2,params=ApproxCholParams(:wdeg));
  
    n = size(a,1)
    k = round(Int, JLfac*log(n)) # number of dims for JL

    U = wtedEdgeVertexMat(a)
    m = size(U,1)

    
    URt = zeros(k,size(U,2))

    println(size(U))
    println(size(URt))
    println(size(k))
    @time multithread_blocked_dense_sparse_pm1!(U, URt, k, 50, 15000)
    UR = URt';
  
    UR .= UR ./ sqrt(n)
    V = zeros(k, n)
    @time solve_with_lap(V, UR, f)
  
    (ai,aj,av) = findnz(triu(a))
    prs = zeros(size(av))

  #   @time for h in 1:length(av)
  #       i = ai[h]
  #       j = aj[h]
  #       temp_norm = 0.0
  #       @turbo for loop = 1 : k
  #         temp_norm += (Vt[loop, i] - Vt[loop, j])^2
  #       end
  #       #@views temp_norm = norm(Vt[:, i] - Vt[:, j])
  #       temp_norm = sqrt(temp_norm)
  #       prs[h] = min(1, av[h] * (temp_norm^2 / k) * matrixConcConst *log(n) / ep^2)
  #   end
  
  @time compute_diff_norm(prs, length(av), V, ai, aj)

  @time @inbounds @tturbo for h in 1:length(av)
      prs[h] = av[h] * (prs[h]^2 / k) * matrixConcConst *log(n) / ep^2
  end

  @time @inbounds for h in 1:length(av)
      prs[h] = min(prs[h], 1)
  end


  
  return prs

end

function solve_with_lap(V, UR, f)
  k = size(V, 1)
  n = size(V, 2)
  BLAS.set_num_threads(1)
  Threads.@threads for i in 1 : k
      @views V[i,:] = f(UR[:,i])
  end
  BLAS.set_num_threads(Threads.nthreads())
end


function compute_diff_norm(prs, len, V, ai, aj)
  k = size(V, 1)
  @inbounds Threads.@threads for h = 1 : len
      i = ai[h]
      j = aj[h]
      temp = 0.0
      @inbounds @turbo for loop = 1 : k
          temp += (V[loop, i] - V[loop, j])^2
      end
      prs[h] = sqrt(temp)
  end
end
