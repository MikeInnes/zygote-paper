# Recursive zygote update method, this is the general recursion case:
function update!(opt, model, updates)
	# If this `model` node has no fields, then just return it
    if nfields(model) == 0
        return model
    end

	# If it does have fields, recurse into them:
    for field_idx in 1:nfields(model)
        update!(opt, getfield(model, field_idx), getfield(updates, field_idx))
    end

    # In the end, return the `model`
    return model
end

# If the `updates` is set to `Nothing`, then just return `model`; this means
# that there were no changes to be applied to this piece of the model.
update!(opt, model, updates::Nothing) = model

# If the `updates` are set as a `Ref`, then we need to peel it:
update!(opt, model, updates::Base.RefValue) = update!(opt, model, updates[])

# If `model` is an `AbstractArray` and `updates` is too, then apply our Flux
# optimizer to the incoming gradients and apply them to the model!
function update!(opt, model::AbstractArray, updates::AbstractArray)
    # Sub off to our RMSProp optimizer
    Flux.Optimise.apply!(opt, model, updates)
    return model .-= updates
end
