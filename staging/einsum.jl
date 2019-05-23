using PartialP, MacroTools, InteractiveUtils

# This script shows off a simple `einsum` macro, similar to that in numpy,
# PyTorch or TensorFlow.

include("_einsum_impl.jl")

# This differs from the numpy notation (though it doesn't have to; one could
# equally implement `@einsum("ik,kj->ij", a, a)`). The syntax is motivated by
# the analogy to a lambda `c = (i, j) -> a[i*k]*a[k*j]`.

# Here's an einsum expression that just calculates `a*a`.

a = [1 2; 3 4]

b = @einsum [i,j] -> a[i,k]*a[k,j]

b == a*a

# Sum across the first and second dimension.
@einsum [i] -> a[k,i]
@einsum [i] -> a[i,k]

# Outer product
v = [1, 2, 3]
@einsum [i,j] -> v[i]*v[j]

# We can see what actually runs using MacroTools' `@expand`. There is _no_
# runtime overhead to using `einsum`.

@expand @einsum [i,j] -> a[i,k]*a[k,j] # a * a
@expand @einsum [i] -> a[i] # a

# This is unlike numpy or PyTorch where the expression must be interpreted
# each time it runs, addding overhead.

mul(x, y) = @einsum [i,j] -> x[i,k]*y[k,j]
# Exactly equivalent to `mul(x, y) = x*y`

# This applies for the purposes of gradients, too:
gradient(x -> sum(mul(x, x)), a)

mul5(x) = mul(x, 5)

# In some cases we can fully statically resolve the answer.
@code_llvm mul5'(5)
