using Zygote, Flux, BenchmarkTools, Plots

function build_model(num_layers::Int, size::Int)
    layers = Any[]
    for layer_idx in 1:num_layers
        push!(layers, Dense(size, size))
        push!(layers, x -> relu.(x))
    end
    return Chain(layers...)
end

sizes = 2 .^ (1:11)
layers = 1:3
timings = Dict()
for num_layers in layers
    timings[num_layers] = Any[]
    for size in sizes
        model = build_model(num_layers, size)
        x = randn(size)
        y, back = Zygote.forward(m -> sum(m(x)), model)
        push!(timings[num_layers], @benchmark $back(1f0))
        @info num_layers, size, timings[num_layers][end]
    end
end

gr()
ENV["GKSwstype"]="100"
p = plot()
for num_layers in layers
    plot!(p, sizes, [minimum(t).time for t in timings[num_layers]]; title="MLP Runtime", ylabel="Runtime (μs)", xlabel="Perceptron Width", label="$(num_layers) layers")
end
savefig(p, joinpath(@__DIR__, "mlp_runtime.png"))
