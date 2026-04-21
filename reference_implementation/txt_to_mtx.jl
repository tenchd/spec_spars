using SparseArrays
using MatrixMarket

# --------------------------------------------------------------
"""
    edge_list_to_mtx(in_path::AbstractString,
                     out_path::AbstractString;
                     default_weight::Float64 = 1.0,
                     index_base::Int = 0)

Read an edge‑list file whose vertices are numbered starting at `index_base`
(0 for the data you have, 1 for a 1‑based file) and write a Matrix‑Market
(`.mtx`) file.

* **in_path** – first line: `num_nodes num_edges` (the second number is
  ignored).  Remaining lines: `src dst [weight]`.  If *weight* is omitted,
  `default_weight` is used.

* **out_path** – destination file.

* **default_weight** – weight assigned when a line does not contain a
  third column (default = `1.0`).

* **index_base** – smallest vertex identifier in the file (`0` for your
  data).  The function internally shifts the ids to Julia’s 1‑based
  indexing.

The routine collects all unique undirected edges (first weight wins),
creates the sparse matrix **once** with `sparse(I,J,V)`, and writes it
using `MatrixMarket.mmwrite`.  The output is a *real symmetric* matrix
in coordinate format.
"""
function edge_list_to_mtx(in_path::AbstractString,
                          out_path::AbstractString;
                          default_weight::Float64 = 1.0,
                          index_base::Int = 0)

    # --------------------------------------------------------------
    # 1️⃣  Read header, allocate the dictionary that will hold unique edges
    # --------------------------------------------------------------
    open(in_path, "r") do io
        header = readline(io)
        parts  = split(header)

        length(parts) ≥ 2 ||
            throw(ArgumentError("Header line must contain at least two integers"))

        n_nodes = parse(Int, parts[1])          # number of vertices
        # (parts[2] – declared edge count – is ignored)

        # offset to convert from file indexing to Julia 1‑based indexing
        offset = 1 - index_base                  # e.g. 1 - 0 = +1  → 0→1, 1→2, …

        # store the *first* weight seen for every unordered pair (i,j) with i≤j
        edge_weights = Dict{Tuple{Int,Int}, Float64}()

        for raw in eachline(io)
            line = strip(raw)
            isempty(line) && continue            # skip blank lines

            tokens = split(line)
            length(tokens) < 2 &&
                throw(ArgumentError("Edge line needs at least two numbers"))

            u_raw = parse(Int, tokens[1])
            v_raw = parse(Int, tokens[2])

            # optional third token = weight
            w = (length(tokens) ≥ 3) ?
                parse(Float64, tokens[3]) :
                default_weight

            # shift to 1‑based indexing
            u = u_raw + offset
            v = v_raw + offset

            # basic bounds check (helps catch malformed files)
            if !(1 ≤ u ≤ n_nodes) || !(1 ≤ v ≤ n_nodes)
                throw(ArgumentError(
                    "Vertex out of range after conversion: " *
                    "u=$u_raw→$u, v=$v_raw→$v (expected 0:$(n_nodes-1) )"))
            end

            # enforce j ≤ i for a symmetric matrix
            i,j = minmax(u, v)                 # same as: i, j = u ≤ v ? (u, v) : (v, u)

            # keep the first weight encountered for a duplicate edge
            if !haskey(edge_weights, (i, j))
                edge_weights[(i, j)] = w
            end
        end

        # --------------------------------------------------------------
        # 2️⃣  Build the three vectors I, J, V  (no per‑edge matrix updates)
        # --------------------------------------------------------------
        ne = length(edge_weights)                # number of unique undirected edges
        I = Vector{Int}(undef, ne)
        J = Vector{Int}(undef, ne)
        V = Vector{Float64}(undef, ne)

        k = 1
        for ((i, j), w) in edge_weights
            I[k] = i
            J[k] = j
            V[k] = w
            k += 1
        end

        # Construct the sparse matrix *once*
        A = sparse(I, J, V, n_nodes, n_nodes)   # only lower‑triangle entries stored

        # --------------------------------------------------------------
        # 3️⃣  Write the Matrix‑Market file
        # --------------------------------------------------------------
        mkpath(dirname(out_path))                # create parent folder if missing
        mmwrite(out_path, A)
    end

    return nothing
end

# ----------------------------------------------------------------------
# Example usage ---------------------------------------------------------
# ----------------------------------------------------------------------

in_files  = ["/global/homes/d/dtench/m1982/david/dense_streams/sources/kron13_raw.txt",
                "/global/homes/d/dtench/m1982/david/dense_streams/sources/kron14_raw.txt",
                "/global/homes/d/dtench/m1982/david/dense_streams/sources/kron15_raw.txt",
                "/global/homes/d/dtench/m1982/david/dense_streams/sources/kron17_raw.txt",
                "/global/homes/d/dtench/m1982/david/dense_streams/sources/ktree_13_2048_edgelist.txt",
                "/global/homes/d/dtench/m1982/david/dense_streams/sources/ktree_15_8192_edgelist.txt",
                "/global/homes/d/dtench/m1982/david/dense_streams/sources/ktree_16_16384_edgelist.txt",
                "/global/homes/d/dtench/m1982/david/dense_streams/sources/ktree_17_32768_edgelist.txt",
            ]

out_files = ["/global/homes/d/dtench/m1982/david/dense_streams/kron13.mtx",
                "/global/homes/d/dtench/m1982/david/dense_streams/kron14.mtx",
                "/global/homes/d/dtench/m1982/david/dense_streams/kron15.mtx",
                "/global/homes/d/dtench/m1982/david/dense_streams/kron17.mtx",
                "/global/homes/d/dtench/m1982/david/dense_streams/ktree13.mtx",
                "/global/homes/d/dtench/m1982/david/dense_streams/ktree15.mtx",
                "/global/homes/d/dtench/m1982/david/dense_streams/ktree16.mtx",
                "/global/homes/d/dtench/m1982/david/dense_streams/ktree17.mtx",
            ]
for i in 1:8
    start_time = time()
    in_file = in_files[i]
    out_file = out_files[i]
    println("converting $in_file to mtx")
    edge_list_to_mtx(in_file, out_file; default_weight = 1.0, index_base = 0)
    println("Matrix‑Market file written to $out_file")
    elapsed_time = time() - start_time
    println("took $elapsed_time seconds.")
end
