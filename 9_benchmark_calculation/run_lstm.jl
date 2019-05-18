using Zygote, Flux, BenchmarkTools, Test, DataFrames, GLM, Statistics
using Plots
import Flux: gate, glorot_uniform

struct LSTMCell{A,V}
  Wi::A
  Wh::A
  b::V
  h::V
  c::V
end

function LSTMCell(in::Integer, out::Integer;
                  init = glorot_uniform)
  cell = LSTMCell(init(out*4, in), init(out*4, out), init(out*4),
                  zeros(Float32, out), zeros(Float32, out))
  cell.b[gate(out, 2)] .= 1
  return cell
end

function (m::LSTMCell)(h, c, x)
  b, o = m.b, size(h, 1)
  g = m.Wi*x .+ m.Wh*h .+ b
  input = σ.(gate(g, o, 1))
  forget = σ.(gate(g, o, 2))
  cell = tanh.(gate(g, o, 3))
  output = σ.(gate(g, o, 4))
  c = forget .* c .+ input .* cell
  h′ = output .* tanh.(c)
  return h′, c, h′
end

# m = LSTMCell(4, 4)
# x = rand(Float32, 4)
# @code_typed gradient((m, x) -> sum(m(m.h, m.c, x)[3]), m, x)
#
# y, back = Zygote._forward(Zygote.Context(), (m, x) -> m(m.h, m.c, x)[3], m, x)

function test(m::LSTMCell, x, len)
  h = m.h
  c = m.c
  for i = 1:len
    r = m(h, c, x)
    h = r[1]
    c = r[2]
    x = r[3]
  end
  return sum(x)
end

gtest(m, x, len) = gradient(test, m, x, len)

to_smatrix(x) = x
function to_smatrix(x::Matrix)
    return SMatrix{size(x)...}(x)
end

@info "Reporting (num_layers, seq_len, feature_size, batch_size, time)"
function sweep_batchsizes(batch_sizes, seq_len = 4, feature_size = 4)
    timings = Float64[]
    for batch_size in batch_sizes
        model = LSTMCell(feature_size, feature_size)
        x = randn(feature_size, batch_size)
        push!(timings, minimum(@benchmark $gtest($model, $x, $seq_len)).time/seq_len)
        @info seq_len, feature_size, batch_size, timings[end]
    end
    return timings
end

batch_sizes = [10, 20, 40, 60, 100]
timings = Dict()
for seq_len in (4, 8),
    feature_size in (4, 8)
    timings[(seq_len, feature_size, batch_sizes)] = sweep_batchsizes(batch_sizes, seq_len, feature_size)
end

overhead_estimates = Dict()
for k in keys(timings)
    data = DataFrame(X=batch_sizes, Y=timings[k])
    ols = lm(@formula(Y ~ X), data)
    overhead_estimates[k] = coeftable(ols).cols[1][1]
end
@show overhead_estimates
lstm_ops = 40
samples = values(overhead_estimates)./lstm_ops
println("Mean overhead: $(mean(samples))ns ± $(std(samples))")

#gr()
#ENV["GKSwstype"]="100"
#p = plot()
#for num_layers in layers
#    plot!(p, seq_lens, [minimum(t).time for t in timings[num_layers]]; title="LSTM Runtime", ylabel="Runtime (μs)", xlabel="Sequence Length", label="$(num_layers) layers")
#end
#savefig(p, joinpath(@__DIR__, "lstm_runtime.png"))
