# Define LSTM structure that will hold parameters
# as well as state from previous invocations
mutable struct LSTM{A,V}
	# Recurrency parameters
    Wi::A
    Wh::A
    b::V

	# Output
    h::V
	# Hidden recurrent state
    c::V
end

function LSTM(in::Integer, out::Integer)
    cell = LSTM(
        # Recurrency parameters
        glorot_uniform(Float32, out*4, in),
        glorot_uniform(Float32, out*4, out),
        glorot_uniform(Float32, out*4),

        # Output/state
        zeros(Float32, out),
        zeros(Float32, out),
    )

	# Initialize `forget` gate to unitary feedback
    cell.b[gate(out, 2)] .= 1
    return cell
end

# Helper accessors for gate outputs
gate(h, n) = (1:h) .+ h*(n-1)
gate(x::AbstractVector, h, n) = x[gate(h,n)]
gate(x::AbstractMatrix, h, n) = x[gate(h,n),:]

# The venerable sigmoid function
σ(x::Real) = one(x) / (one(x) + exp(-x))

# Reset LSTM state to initial zero state
function reset!(m::LSTM)
    out = length(m.h)
    m.h = zeros(Float32, out)
    m.c = zeros(Float32, out)
    return m
end

# Define forward pass for LSTM cell
function (m::LSTM)(x)
    # Output size
	o = size(m.h, 1)

    # Internal computation
	g = m.Wi*x .+ m.Wh*m.h .+ m.b
	input = σ.(gate(g, o, 1))
	forget = σ.(gate(g, o, 2))
	cell = tanh.(gate(g, o, 3))
	output = σ.(gate(g, o, 4))

	# Calculate our next state, store that in m.state
	m.c = forget .* m.c .+ input .* cell
	m.h = output .* tanh.(m.c)

    # Return output vector
	return m.h
end

