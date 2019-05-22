function load_data(filename::String, seq_len = 50, batch_size = 64)
    # Load data, split it into an alphabet, then convert
    # the text to a onehot matrix representation.
    text = collect(String(read(filename)))
    alphabet = [unique(text)..., '_']
    text = map(ch -> onehot(ch, alphabet), text)
    stop = onehot('_', alphabet)

    Xs = collect(partition(batchseq(chunk(text[1:end-1], batch_size), stop), seq_len))
    Ys = collect(partition(batchseq(chunk(text[2:end],   batch_size), stop), seq_len))
    return alphabet, Xs, Ys
end

# Sample randomly from model output to generate sentences
function sample(m, alphabet, len)
    Flux.reset!(m)
    buf = IOBuffer()
    c = rand(alphabet)
    for i = 1:len
        write(buf, c)
        c = wsample(alphabet, m(onehot(c, alphabet)))
    end
    return String(take!(buf))
end


