using Colors, Zygote

target = RGB(1, 0, 0)
colour = RGB(1, 1, 1)

function update_color(c, Δ, η = 0.001)
    return RGB(
        c.r - η*Δ.r,
        c.g - η*Δ.g,
        c.b - η*Δ.b,
    )
end

for idx in 1:51
    global colour, target

    # Calculate gradients
    grads = Zygote.gradient(colour) do y
        colordiff(target, y)
    end
    # Update colour
    colour = update_color(colour, grads[1])
    if idx % 5 == 1
        @info idx, colour
    end
end
