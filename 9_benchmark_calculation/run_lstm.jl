using LinearAlgebra
BLAS.set_num_threads(1)

using Zygote, Flux, BenchmarkTools, Test, DataFrames, GLM, Statistics, StaticArrays
using Plots

to_smatrix(x) = x
function to_smatrix(x::Matrix)
    return SMatrix{size(x)...}(x)
end

function build_model(num_layers::Int, size::Int)
    layers = Any[]
    for layer_idx in 1:num_layers
        push!(layers, LSTM(size, size))
    end
    return mapleaves(to_smatrix, mapleaves(Flux.data, Chain(layers...)))
end

@info "Reporting (num_layers, seq_len, feature_size, batch_size, time)"
function sweep_batchsizes(batch_sizes, num_layers = 1, seq_len = 4, feature_size = 4)
    timings = Float64[]
    for batch_size in batch_sizes
        model = build_model(num_layers, feature_size)
        x = SMatrix{feature_size,batch_size}(randn(feature_size, batch_size))
        test(m, x) = gradient(m, x) do m, x
            x_t = x
            for idx in 1:seq_len
                x_t = m(x_t)
            end
            return sum(x_t)
        end
        push!(timings, minimum(@benchmark $test($model, $x)).time)
        @info num_layers, seq_len, feature_size, batch_size, timings[end]
    end
    return timings
end


batch_sizes = 1:8
layer_sizes = 1:3
timings = Dict()
for num_layers in layer_sizes,
    seq_len in (4,),
    feature_size in (4,)

    timings[(num_layers, seq_len, feature_size, batch_sizes)] = sweep_batchsizes(batch_sizes, num_layers, seq_len, feature_size)
end

# These gotten via count_ops/count_lstm_ops.jl
num_ops = Dict(
    1 => 255,
    2 => 491,
    3 => 727,
    4 => 963,
)

overhead_estimates = Dict()
workload_estimates = Dict()
for k in keys(timings)
    num_layers, seq_len, feature_size, batch_sizes = k
    data = DataFrame(X=batch_sizes, Y=timings[k])
    ols = lm(@formula(Y ~ X), data)
    overhead_estimates[k] = coeftable(ols).cols[1][1]/num_ops[num_layers]
    workload_estimates[k] = coeftable(ols).cols[1][2]
end

gr()
ENV["GKSwstype"]="100"
p = plot()
for key in sort(collect(keys(timings)))
    num_layers, seq_len, feature_size, batch_sizes = key
    plot!(p, batch_sizes, timings[key]; title="LSTM Runtime", ylabel="Absolute Runtime (ns)", xlabel="Batch Size", label="$(num_layers) layers")
    extrapolate(x) = overhead_estimates[key]*num_ops[num_layers] + workload_estimates[key]*x
    x_points = [0, batch_sizes...]
    plot!(p, x_points, extrapolate.(x_points); style=:dashdot, label="$(num_layers) layers extrapolation")
end
savefig(p, joinpath(@__DIR__, "lstm_runtime.pdf"))
