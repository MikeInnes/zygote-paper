using Zygote, StaticArrays, InteractiveUtils

# Naive Matrix Multiply

function mul(a::AbstractMatrix, b::AbstractMatrix)
  [sum(a[i, k]*b[k,j] for k = 1:size(a, 2))
   for i = 1:size(a, 1), j = 1:size(b, 2)]
end

a = [1 2; 3 4]

mul(a, a)

# Static Matrix Multiple

_sum(xs) = reduce((a, b) -> :($a+$b), xs)

@generated function mul(a::SMatrix{M,N}, b::SMatrix{N,O}) where {M,N,O}
  :(SMatrix{M,O}(
      $([_sum(:(a[$i, $k]*b[$k,$j]) for k = 1:size(a, 2))
         for i = 1:size(a, 1), j = 1:size(b, 2)]...),))
end

@generated function mul(a::SMatrix{M,N}, b::SMatrix{N,O}) where {M,N,O}
  quote
    Base.@_inline_meta
    SMatrix{M,O}(
      $([_sum(:(a[$i, $k]*b[$k,$j]) for k = 1:size(a, 2))
        for i = 1:size(a, 1), j = 1:size(b, 2)]...),)
  end
end

a = @SArray [1 2; 3 4]

mul(a, a)

sqmul(a) = sum(mul(a, a))

@code_llvm sqmul(a)

# See the code output

function _mul(a::SMatrix{M,N}, b::SMatrix{N,O}) where {M,N,O}
  :(SMatrix{M,O}(
      $([_sum(:(a[$i, $k]*b[$k,$j]) for k = 1:size(a, 2))
         for i = 1:size(a, 1), j = 1:size(b, 2)]...),))
end

_mul(a, a)
