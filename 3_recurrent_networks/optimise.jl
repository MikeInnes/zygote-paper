"""
    RMSProp(η = 0.001, ρ = 0.9)

[RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
optimiser. Parameters other than learning rate don't need tuning. Often a good
choice for recurrent networks.
"""
mutable struct RMSProp
    eta::Float64
    rho::Float64
    acc::IdDict
end

RMSProp(η = 0.001, ρ = 0.9) = RMSProp(η, ρ, IdDict())

function apply!(o::RMSProp, x, Δ)
    η, ρ = o.eta, o.rho
    acc = get!(o.acc, x, zero(x))::typeof(data(x))
    @. acc = ρ * acc + (1 - ρ) * Δ^2
    @. Δ *= η / (√acc + ϵ)
end



# Recursive zygote update method, this is the general recursion case:
function zyg_update!(opt, model, updates)
	# If this `model` node has no fields, then just return it
    if nfields(model) == 0
        return model
    end

	# If it does have fields, recurse into them:
    for field_idx in 1:nfields(model)
        zyg_update!(opt, getfield(model, field_idx), getfield(updates, field_idx))
    end

    # In the end, return the `model`
    return model
end

# If the `updates` is set to `Nothing`, then just return `model`; this means
# that there were no changes to be applied to this piece of the model.
zyg_update!(opt, model, updates::Nothing) = model

# If `model` is an `AbstractArray` and `updates` is too, then apply our Flux
# optimizer to the incoming gradients and apply them to the model!
function zyg_update!(opt, model::AbstractArray, updates::AbstractArray)
    # Sub off to our RMSProp optimizer
    apply!(opt, model, updates)
    return model .-= updates
end
