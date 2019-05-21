struct Dense{S,T}
    W::S
    b::T
end

function Dense(in::Integer, out::Integer)
    return Dense(
        glorot_uniform(Float32, out, in),
        zeros(Float32, out),
    )
end

# Define forward pass for `Dense` layer
function (m::Dense)(x)
    return m.W * x .+ m.b
end
