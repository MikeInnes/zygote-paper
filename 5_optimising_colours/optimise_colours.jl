using Colors, Zygote

target = RGB(1, 0, 0)
colour = RGB(1, 1, 1)

grads = Zygote.gradient(colour) do y
    colordiff(target, y)
end
@show grads
