using Zygote, Flux, BenchmarkTools, Plots, Test

function build_model(num_layers::Int, size::Int)
    layers = Any[]
    for layer_idx in 1:num_layers
        push!(layers, LSTM(size, size))
    end
    return mapleaves(Flux.data, Chain(layers...))
end

timings = Dict()
layers = 1:5
seq_lens = 1:20
for num_layers in layers
    timings[num_layers] = Any[]
    for seq_len in seq_lens
        size = 4
        model = build_model(num_layers, size)
        x = randn(size)
        y, back = Zygote.forward(m -> begin
            x_t = x
            for idx in 1:seq_len
                x_t = m(x_t)
            end
            return sum(x_t)
        end, model)
        push!(timings[num_layers], @benchmark $back(1f0))
        @info num_layers, seq_len, size, timings[num_layers][end]
    end
end

using DataFrames, GLM, Statistics
overhead_estimates = Float64[]
for num_layers in layers
    data = DataFrame(X=[minimum(t).time for t in timings[num_layers]], Y=seq_lens)
    ols = lm(@formula(Y ~ X), data)
    push!(overhead_estimates, coeftable(ols).cols[1][1])
end
println("Mean overhead: $(mean(overhead_estimates))ns")

gr()
ENV["GKSwstype"]="100"
p = plot()
for num_layers in layers
    plot!(p, seq_lens, [minimum(t).time for t in timings[num_layers]]; title="LSTM Runtime", ylabel="Runtime (μs)", xlabel="Sequence Length", label="$(num_layers) layers")
end
savefig(p, joinpath(@__DIR__, "lstm_runtime.png"))
