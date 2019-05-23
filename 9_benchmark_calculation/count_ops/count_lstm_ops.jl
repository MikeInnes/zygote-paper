# Note, this requires a special branch of Zygote to count ops
using Zygote, Flux

function build_model(num_layers::Int, size::Int)
    layers = Any[]
    for layer_idx in 1:num_layers
        push!(layers, LSTM(size, size))
    end
    return mapleaves(Flux.data, Chain(layers...))
end


for num_layers in 1:4
	model = build_model(num_layers, 4)

	seq_len = 4
	x = randn(4, 4)
	test(m, x) = Zygote.gradient(m, x) do m, x
		x_t = x
		for idx in 1:seq_len
			x_t = m(x_t)
		end
		return sum(x_t)
	end
	test(model, x)
	Zygote.reset_num_ops!()
	test(model, x)
	@info "ops for layers", num_layers, Zygote.get_num_ops()
end
