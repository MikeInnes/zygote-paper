# Download dataset if it does not already exist
if !isfile("shakespeare_input.txt")
    download("https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt", "shakespeare_input.txt")
end

# Helper function to generate glorot (Xavier) initializations
function glorot_uniform(T, dims...)
	zero_mean_noise = rand(T, dims...) .- 0.5f0
    return zero_mean_noise .* sqrt(24.0f0/sum(dims))
end

# Helper function to convert a sequence of tokens into an
# alphabet and a large onehot-encoded matrix
function make_onehot(text::Vector)
    alphabet = [unique(text)..., '_']
    text_encoded = zeros(length(text), length(alphabet))
    for ch_idx in 1:length(text)
        text_encoded[ch_idx, findfirst(isequal(text[ch_idx]), alphabet)] = 1
    end
    
    return alphabet, text_encoded
end


