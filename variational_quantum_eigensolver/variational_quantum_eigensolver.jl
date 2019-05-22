include("patch.jl")

# generate a Heisenberg Model Hamiltonian
nbit = 4
h = mat(heisenberg(nbit))
v0 = statevec(zero_state(nbit))
function energy(circuit)
    v = mat(circuit) * v0
    (v'* h * v)[] |> real
end

# Generate a circuit as a wave function ansatz
circuit = random_diff_circuit(nbit, 2, pair_ring(nbit))

using Flux: ADAM, Optimise
function train!(lossfunc, circuit, optimizer; maxiter::Int=200)
    dispatch!(circuit, :random)
    params = parameters(circuit)
    loss_history = Float64[]
    for i = 1:maxiter
        # collect gradients from returned structured data
        grad = collect_gradients(lossfunc'(circuit))
        Optimise.update!(optimizer, params, grad)
        dispatch!(circuit, params)
        eng = lossfunc(circuit)
        push!(loss_history, eng)
        println("Iter $i, Energy (Loss) = $(eng)")
    end
    loss_history
end

using Random
Random.seed!(5)
EG = eigvals(Matrix(h))[1]
println("$nbit site Heisenberg model, exact ground state energy = $EG")
loss_history = train!(energy, circuit, ADAM(0.1); maxiter=200)

using DelimitedFiles
fname = "loss_history_$nbit.dat"
println("Saving training data to file $fname")
writedlm(fname, loss_history)


