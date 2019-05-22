using Zygote, Flux, NNlib
using Flux: onehot, chunk, batchseq, crossentropy
using Base.Iterators: partition
using StatsBase: wsample

# Optimizer and data loading utilities
include("utils.jl")
include("optimise.jl")

function train_model!(model, Xs, Ys, num_epochs = 10)
    opt = ADAM(0.01) 
    avg_loss = 0.0

    # Run through our entire dataset a few times 
    for epoch_idx in 1:num_epochs
        for (batch_idx, (x_batch, y_batch)) in enumerate(zip(Xs, Ys))
            # Calculate gradients upon the model for this batch of data,
            # summing crossentropy loss across time
            l, back = Zygote.forward(model) do model
				return sum(crossentropy.(model.(x_batch), y_batch))
			end
            grads = back(1f0)[1]

            # Update the model, then reset its internal LSTM states
            model = update!(opt, model, grads)
            Flux.reset!(model)

            avg_loss = avg_loss*0.98 + 0.02*l
            @info(epoch_idx, batch_idx, l, avg_loss)
        end
        @info("Done with epoch $(epoch_idx)!")
    end

    return model
end


# Download dataset if it does not already exist
if !isfile("shakespeare_input.txt")
    download("https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt", "shakespeare_input.txt")
end

# Load data and alphabet
alphabet, Xs, Ys = load_data("shakespeare_input.txt")

# Define simple LSTM-based model to map from alphabet back on to alphabet,
# predicting the next letter in a corpus of Shakespeare text.
model = mapleaves(Flux.data, Chain(
    LSTM(length(alphabet), 128),
    LSTM(128, 128),
    Dense(128, length(alphabet)),
	softmax,
))

model = train_model!(model, Xs, Ys)

println("Behold, I speak:")
println(sample(model, alphabet, 1000))

