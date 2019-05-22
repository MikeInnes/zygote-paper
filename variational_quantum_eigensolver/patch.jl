using Zygote
using Zygote: gradient, @adjoint
using Zygote, LuxurySparse, Yao, YaoBase, SparseArrays, BitBasis
import Zygote: Context

using Yao
using YaoBlocks: ConstGate
import Yao: apply!, ArrayReg, statevec, RotationGate

using LuxurySparse, SparseArrays, LinearAlgebra
using BitBasis: controller, controldo
using TupleTools


Zygote.@adjoint (::Type{T})(perms, vals) where T <: PermMatrix = T(perms, vals), Δ -> nothing
Zygote.@adjoint (::Type{T})() where T <: IMatrix = T(), Δ -> nothing
Zygote.@adjoint Base.:(*)(A::Number, B::PermMatrix) = A * B, Δ->(sum(Δ .* B), A * Δ)
Zygote.@adjoint Base.:(*)(A::PermMatrix, B::Number) = A * B, Δ->(A * Δ, sum(Δ .* B))

Zygote.@adjoint SparseArrays.SparseMatrixCSC(A::PermMatrix) = SparseMatrixCSC(A), Δ->(Δ, )

Zygote.@adjoint BitBasis.onehot(::Type{T}, nbits::Int, x::Integer, nbatch::Int) where T = onehot(T, nbits, x, nbatch), Δ -> nothing

# upstreams
Zygote.@adjoint Base.:(-)(a, b) = a-b, Δ -> (Δ, -Δ)
Zygote.@adjoint Base.:(+)(a, b) = a+b, Δ -> (Δ, Δ)

# require mutate
Zygote.@adjoint! function copyto!(xs::AbstractVector, ys::Tuple)
    xs_ = copy(xs)
    copyto!(xs, ys), function (dxs)
        copyto!(xs_, xs)
        return (nothing, Tuple(dxs))
    end
end

@adjoint function Base.Iterators.Zip(tp)
    Base.Iterators.Zip(tp), adjy-> ((@show adjy;zip(adjy...)),)
end

@adjoint function reduce(func, xs; kwargs...)
    backs = Any[]
    ys = Any[]
    function nfunc(x, x2)
        y, back = forward(func, x, x2)
        push!(backs, back)
        push!(ys, y)
        return y
    end
    reduce(nfunc, xs; kwargs...),
    function (adjy)
        res = Vector{Any}(undef, length(ys))
        for i=length(ys):-1:1
            back, y = backs[i], ys[i]
            adjy, res[i] = back(adjy)
        end
        if !haskey(kwargs, :init)
            insert!(res, 1, adjy)
        end
        return (nothing, res)
    end
end

@adjoint function mapreduce(op, func, xs; kwargs...)
    opbacks = Any[]
    backs = Any[]
    ys = Any[]
    function nop(x)
        y, back = forward(op,x)
        push!(opbacks, back)
        y
    end
    function nfunc(x, x2)
        y, back = forward(func, x, x2)
        push!(backs, back)
        push!(ys, y)
        return y
    end
    mapreduce(nop, nfunc, xs; kwargs...),
    function (adjy)
        offset = haskey(kwargs, :init) ? 0 : 1
        res = Vector{Any}(undef, length(ys)+offset)
        for i=length(ys):-1:1
            opback, back, y = opbacks[i+offset], backs[i], ys[i]
            adjy, adjthis = back(adjy)
            res[i+offset], = opback(adjthis)
        end
        if offset==1
            res[1], = opbacks[1](adjy)
        end
        return (nothing, nothing, res)
    end
end

@adjoint function collect(::Type{T}, source::TS) where {T, TS}
    collect(T, source),
    adjy -> (adjy,)   # adjy -> convert(TS, adjy)
end

@adjoint Iterators.reverse(x::T) where T = Iterators.reverse(x), adjy->(collect(Iterators.reverse(adjy)),)  # convert is better

# data projection
@adjoint function *(sp::Union{SDSparseMatrixCSC, SDDiagonal, SDPermMatrix}, v::AbstractVector)
    sp*v, adjy -> (outer_projection(sp, adjy, v'), sp'*adjy)
end

@adjoint YaoBlocks.decode_sign(args...) = YaoBlocks.decode_sign(args...), adjy->nothing

@adjoint function *(v::LinearAlgebra.Adjoint{T, V}, sp::Union{SDSparseMatrixCSC, SDDiagonal, SDPermMatrix}) where {T, V<:AbstractVector}
    v*sp, adjy -> (adjy*sp', outer_projection(sp, v', adjy))
end

@adjoint function *(v::LinearAlgebra.Adjoint{T, V}, sp::SDDiagonal, v2::AbstractVector) where {T, V<:AbstractVector}
    v*sp, adjy -> (adjy*(sp*v2)', adjy*projection(sp, v', v2'), adjy*(v*sp)')
end

function outer_projection(y::SDSparseMatrixCSC, adjy, v)
    # adjy*v^T
    out = zero(y)
    is, js, vs = findnz(y)
    for (k,(i,j)) in enumerate(zip(is, js))
        @inbounds out.nzval[k] = adjy[i]*v[j]
    end
    out
end

outer_projection(y::SDDiagonal, adjy, v) = Diagonal(adjy.*v)

"""
Project a dense matrix to a sparse matrix
"""
function projection(y::AbstractSparseMatrix, m::AbstractMatrix)
    out = zero(y)
    is, js, vs = findnz(y)
    for (k,(i,j)) in enumerate(zip(is, js))
        @inbounds out.nzval[k] = m[i,j]
    end
    out
end

Base.zero(pm::PermMatrix) = PermMatrix(pm.perm, zero(pm.vals))
projection(y::SDDiagonal, m::AbstractMatrix) = Diagonal(diag(m))
function projection(y::PermMatrix, m::AbstractMatrix)
    res = zero(y)
    for i=1:size(res, 1)
        @inbounds res.vals[i] = m[i,res.perm[i]]
    end
    res
end
projection(x::RT, adjx::Complex) where RT<:Real = RT(real(adjx))


function rotgrad(::Type{T}, rb::RotationGate{N}) where {N, T}
    -sin(rb.theta / 2)/2 * IMatrix{1<<N}() + im/2 * cos(rb.theta / 2) * mat(T, rb.block)
end

@adjoint function mat(::Type{T}, rb::RotationGate{N, RT}) where {T, N, RT}
    mat(T, rb), adjy -> (nothing, projection(rb.theta, sum(adjy .* rotgrad(T, rb))),)
end

@adjoint function mat(::Type{T}, rb::Union{PutBlock{N, C, RT}, RT}) where {T, N, C, RT<:ConstGate.ConstantGate}
    mat(T, rb), adjy -> (nothing, nothing)
end

@adjoint function RotationGate(G, θ)
    RotationGate(G, θ), adjy->(nothing, adjy)
end

@adjoint function PutBlock{N}(block::GT, locs::NTuple{C, Int}) where {N, M, C, GT <: AbstractBlock{M}}
    PutBlock{N}(block, locs), adjy->(adjy.content, adjy.locs)
end

@adjoint function ChainBlock(blocks) where N
    ChainBlock(blocks),
    adjy -> ((@show adjy.blocks; adjy.blocks),)
end

@adjoint function chain(blocks) where N
    chain(blocks),
    adjy -> ((@show adjy.blocks; adjy.blocks),)
end

@adjoint function KronBlock{N, MT}(slots::Vector{Int}, locs::Vector{Int}, blocks::Vector{MT}) where {N, MT<:AbstractBlock}
    KronBlock{N, MT}(slots, locs, blocks),
    adjy -> (adjy.slots, adjy.locs, adjy.blocks)
end

@adjoint function ControlBlock{N, BT, C, M}(ctrl_locs, ctrl_config, block, locs) where {N, C, M, BT<:AbstractBlock}
    ControlBlock{N, BT, C, M}(ctrl_locs, ctrl_config, block, locs),
    adjy->(adjy.ctrl_locs, adjy.ctrl_config, adjy.content, adjy.locs)
end

@adjoint function YaoBlocks.cunmat(nbit::Int, cbits::NTuple{C, Int}, cvals::NTuple{C, Int}, U0::AbstractMatrix, locs::NTuple{M, Int}) where {C, M}
    y = YaoBlocks.cunmat(nbit, cbits, cvals, U0, locs)
    y, adjy-> (nothing, nothing, nothing, adjcunmat(y, adjy, nbit, cbits, cvals, U0, locs), nothing)
end

@inline function adjsetcol!(csc::SparseMatrixCSC, icol::Int, rowval::AbstractVector, nzval::SubArray)
    @inbounds begin
        S = csc.colptr[icol]
        E = csc.colptr[icol+1]-1
        nzval .+= view(csc.nzval, S:E)
    end
    csc
end

@inline function adjunij!(mat::SparseMatrixCSC, locs, U::Matrix)
    for j = 1:size(U, 2)
        @inbounds adjsetcol!(mat, locs[j], locs, view(U,:,j))
    end
    return U
end

@inline function adjunij!(mat::SparseMatrixCSC, locs, U::SparseMatrixCSC)
    for j = 1:size(U, 2)
        S = U.colptr[j]
        E = U.colptr[j+1]-1
        @inbounds adjsetcol!(mat, locs[j], locs, view(U.nzval,S:E))
    end
    return U
end

@inline function adjunij!(mat::SDDiagonal, locs, U::Diagonal)
    @inbounds U.diag .+= mat.diag[locs]
    return U
end

@inline function adjunij!(mat::SDPermMatrix, locs, U::PermMatrix)
    @inbounds U.vals .+= mat.vals[locs]
    return U
end

function adjcunmat(y::AbstractMatrix, adjy::AbstractMatrix, nbit::Int, cbits::NTuple{C, Int}, cvals::NTuple{C, Int}, U0::AbstractMatrix{T}, locs::NTuple{M, Int}) where {C, M, T}
    U, ic, locs_raw = YaoBlocks.reorder_unitary(nbit, cbits, cvals, U0, locs)
    adjy = _render_adjy(adjy, y)
    adjU = _render_adjU(U)

    ctest = controller(cbits, cvals)

    controldo(ic) do i
        adjunij!(adjy, locs_raw+i, adjU)
    end

    adjU = all(TupleTools.diff(locs).>0) ? adjU : YaoBase.reorder(adjU, collect(locs)|>sortperm|>sortperm)
    adjU
end
_render_adjy(adjy, y) = projection(y, adjy)

_render_adjU(U0::AbstractMatrix{T}) where T = zeros(T, size(U0)...)
_render_adjU(U0::SDSparseMatrixCSC{T}) where T = SparseMatrixCSC(size(U0)..., dynamicize(U0.colptr), dynamicize(U0.rowval), zeros(T, U0.nzval|>length))
_render_adjU(U0::SDDiagonal{T}) where T = Diagonal(zeros(T, size(U0, 1)))
_render_adjU(U0::SDPermMatrix{T}) where T = PermMatrix(U0.perm, zeros(T, length(U0.vals)))

function collect_gradients(st, out=Any[])
    for blk in st
        collect_gradients(blk, out)
    end
    out
end

collect_gradients(st::Number, out=any[]) = push!(out, st)
collect_gradients(st::Nothing, out=any[]) = out


############# Build a circuit (Copied from `QuAlgorithmZoo`) ###############
"""
    pair_ring(n::Int) -> Vector

Pair ring.
"""
pair_ring(n::Int) = [i=>mod(i, n)+1 for i=1:n]

"""
    cnot_entangler([n::Int, ] pairs::Vector{Pair}) = ChainBlock

Arbitrary rotation unit, support lazy construction.
"""
cnot_entangler(n::Int, pairs) = chain(n, control(n, [ctrl], target=>X) for (ctrl, target) in pairs)
cnot_entangler(pairs) = n->cnot_entangler(n, pairs)

"""
    rotor(nbit::Int, ibit::Int, noleading::Bool=false, notrailing::Bool=false) -> ChainBlock{nbit, ComplexF64}

Arbitrary rotation unit (put in `nbit` space), set parameters notrailing, noleading true to remove trailing and leading Z gates.
"""
function rotor(nbit::Int, ibit::Int, noleading::Bool=false, notrailing::Bool=false)
    rt = chain(nbit, [put(nbit, ibit=>Rz(0.0)), put(nbit, ibit=>Rx(0.0)), put(nbit, ibit=>Rz(0.0))])
    noleading && popfirst!(rt)
    notrailing && pop!(rt)
    rt
end

rotorset(nbit::Int, noleading::Bool=false, notrailing::Bool=false) = chain(nbit, [rotor(nbit, j, noleading, notrailing) for j=1:nbit])

"""
    random_diff_circuit(nbit, nlayer, pairs; do_cache=false)

A kind of widely used differentiable quantum circuit, angles in the circuit is randomely initialized.

ref:
    1. Kandala, A., Mezzacapo, A., Temme, K., Takita, M., Chow, J. M., & Gambetta, J. M. (2017).
       Hardware-efficient Quantum Optimizer for Small Molecules and Quantum Magnets. Nature Publishing Group, 549(7671), 242–246.
       https://doi.org/10.1038/nature23879.
"""
function random_diff_circuit(nbit, nlayer, pairs; do_cache=false)
    circuit = chain(nbit)

    ent = cnot_entangler(pairs)
    if do_cache
        ent = ent |> cache
    end
    for i = 1:(nlayer + 1)
        i!=1 && push!(circuit, ent)
        push!(circuit, rotorset(nbit, i==1, i==nlayer+1))
    end
    circuit
end

"""
    heisenberg(nbit::Int; periodic::Bool=true)

heisenberg hamiltonian, for its ground state, refer `PRB 48, 6141`.
"""
function heisenberg(nbit::Int; periodic::Bool=true)
    sx = i->put(nbit, i=>X)
    sy = i->put(nbit, i=>Y)
    sz = i->put(nbit, i=>Z)
    mapreduce(i->(j=i%nbit+1; sx(i)*sx(j)+sy(i)*sy(j)+sz(i)*sz(j)), +, 1:(periodic ? nbit : nbit-1))
end



