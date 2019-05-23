using Yao, PartialP, Flux.Optimise
include("patch.jl")

# make a one qubit GHZ state as learning target
# NOTE: this can be an arbitrary one qubit state
t = ArrayReg(bit"0") + ArrayReg(bit"1")
normalize!(t)

# calculate the fidelity with GHZ state
function fid(xs)
    r = zero_state(1)
    U = mat(chain(Rx(xs[1]), Rz(xs[2])))
    return abs(statevec(t)' * U * statevec(r))
end

# simply tell PartialP to get the gradient and start training
function train()
    opt = ADAM()
    xs = rand(2)
    for _ in 1:1000
        println(fid(xs))
        Optimise.update!(opt, xs, -fid'(xs))
    end
    return xs
end

train()
