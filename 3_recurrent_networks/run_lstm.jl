using PartialP, Flux, NNlib, Random
using Flux: onehot, chunk, batchseq, crossentropy
using Base.Iterators: partition
using StatsBase: wsample

# Optimizer and data loading utilities
include("utils.jl")
include("optimise.jl")

function train_model!(model, Xs, Ys, num_epochs = 100)
    opt = ADAM(0.001) 
    avg_loss = 0.0

    # Run through our entire dataset a few times 
    for epoch_idx in 1:num_epochs
        permutation = shuffle(1:length(Xs))
        for batch_idx in 1:length(permutation)
            x_batch, y_batch = Xs[permutation[batch_idx]], Ys[permutation[batch_idx]]
            # Calculate gradients upon the model for this batch of data,
            # summing crossentropy loss across time
            l, back = PartialP.forward(model) do model
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
        println(sample(model, alphabet, 300))
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
    LSTM(length(alphabet), 512),
    LSTM(512, 512),
    Dense(512, length(alphabet)),
	softmax,
))

model = train_model!(model, Xs, Ys)

println("Behold, I speak:")
println(sample(model, alphabet, 1000))

