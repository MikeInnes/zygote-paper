using Zygote, Flux, BenchmarkTools, Plots

function build_model(num_layers::Int, size::Int)
    layers = Any[]
    for layer_idx in 1:num_layers
        push!(layers, Dense(size, size))
        push!(layers, x -> relu.(x))
    end
    return Chain(layers...)
end

timings = Dict()
sizes = 2 .^ (1:11)
for num_layers in 1:3
    timings[num_layers] = Any[]
    for size in sizes
        model = build_model(num_layers, size)
        x = randn(size)
        y, back = Zygote.forward(x -> sum(model(x)), x)
        push!(timings[num_layers], @benchmark $back(1f0))
        @info num_layers, size, timings[num_layers][end]
    end
end

gr()
ENV["GKSwstype"]="100"
p = plot()
for num_layers in 1:3
    plot!(p, sizes, [minimum(t).time for t in timings[num_layers]]; title="MLP Runtime", ylabel="Runtime (μs)", xlabel="Perceptron Width", label="$(num_layers) layers")
end
savefig(p, "mlp_runtime.png")
