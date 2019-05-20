# Since Zygote is at its early stage
# this patch contains some workaround of known issues
# it should be able to get removed once those issues are fixed

using Zygote, LuxurySparse, Yao, YaoBase, SparseArrays, BitBasis

# LuxurySparse constructor adjoints
Zygote.@adjoint (::Type{T})(perms, vals) where T <: PermMatrix = T(perms, vals), Δ -> nothing
Zygote.@adjoint (::Type{T})() where T <: IMatrix = T(), Δ -> nothing
Zygote.@adjoint Base.:(*)(A::Number, B::PermMatrix) = A * B, Δ->(sum(Δ .* B), A * Δ)
Zygote.@adjoint Base.:(*)(A::PermMatrix, B::Number) = A * B, Δ->(A * Δ, sum(Δ .* B))

Zygote.@adjoint SparseArrays.SparseMatrixCSC(A::PermMatrix) = SparseMatrixCSC(A), Δ->(Δ, )

# BitBasis constructor (no gradient needed)
Zygote.@adjoint BitBasis.onehot(::Type{T}, nbits::Int, x::Integer, nbatch::Int) where T = onehot(T, nbits, x, nbatch), Δ -> nothing

# state(r::ArrayReg)
Zygote.@adjoint function Base.getfield(r::ArrayReg, n::Symbol)
    getfield(r, n), Δ -> (ArrayReg(Δ), nothing)
end

# workaround of upstream
Zygote.@adjoint Base.:(-)(a, b) = a-b, Δ -> (Δ, -Δ)
Zygote.@adjoint Base.:(+)(a, b) = a+b, Δ -> (Δ, Δ)

# require mutation support
Zygote.@adjoint! function copyto!(xs::AbstractVector, ys::Tuple)
    xs_ = copy(xs)
    copyto!(xs, ys), function (dxs)
        copyto!(xs_, xs)
        return (nothing, Tuple(dxs))
    end
end
