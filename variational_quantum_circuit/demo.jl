using Yao, Zygote, Flux.Optimise
include("patch.jl")

t = ArrayReg(bit"0") + ArrayReg(bit"1")
normalize!(t)

function fid(xs)
    r = zero_state(1)
    U = mat(chain(Rx(xs[1]), Rz(xs[2])))
    return abs(statevec(t)' * U * statevec(r))
end

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
