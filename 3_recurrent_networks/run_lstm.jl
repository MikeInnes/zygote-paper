using Zygote

# Bring in some utilities
include("utils.jl")

# Bring in our LSTM definition
include("lstm.jl")

# Simple Dense layer definition
include("dense.jl")

# Simple softmax definition (and gradient, for numerical stability)
include("softmax.jl")

# Optimizer definition
include("optimise.jl")

# Run the forward pass of our model
function forward(model, x)
    for layer in model
        x = layer(x)
    end
    return x
end

function loss(model, x, y)
    # For each timestep in x, run it through the model, returning
    # all output datapoints in our matrix y_hat
    y_hat = hcat(forward.(Ref(model), [x[t, :] for t in 1:size(x,1)])...)'
    
    # return crossentropy loss
    return -sum(y .* logsoftmax(y_hat)) * 1 // size(y, 2)
end

function train_model!(model, X, num_epochs = 10, seq_len = 50)
    # Run through our entire dataset a few times 
    opt = RMSProp()
    for epoch_idx in 1:num_epochs
        for start_idx in 1:(size(text_encoded, 1)-seq_len-1)
            # We train to predict the next character
            x_batch = X[start_idx   : start_idx + seq_len, :]
            y_batch = X[start_idx+1 : start_idx+1+seq_len, :]

            # Calculate gradients upon the model for this batch of data
            l, back = Zygote.forward(m -> loss(m, x_batch, y_batch), model)
            grads = back(1f0)

            # Update the model, then reset its internal LSTM states
            model = zyg_update!(opt, model, grads)
            for layer in model
                layer isa LSTM && reset!(layer)
            end

            @info(epoch_idx, start_idx, model, l)
        end
        @info("Done with epoch $epoch_idx!")
    end

    return model
end


# Load data, split it into an alphabet, then convert
# the text to a onehot matrix representation.
text = collect(String(read("shakespeare_input.txt")))
alphabet, text_encoded = make_onehot(text)

# Define simple LSTM-based model to map from alphabet back on to alphabet,
# predicting the next letter in a corpus of Shakespeare text.
model = (
    LSTM(length(alphabet), 128),
    LSTM(128, 128),
    Dense(128, length(alphabet)),
)

model = train_model!(model, text_encoded)
