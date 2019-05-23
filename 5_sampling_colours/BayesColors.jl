using Colors, Distributions, AdvancedHMC, PartialP

#############
# Utilities #
#############

# Prior
struct UniformRGB end
Distributions.rand(::UniformRGB) = RGB(rand(), rand(), rand())
Distributions.logpdf(::UniformRGB, ::RGB{T}) where {T<:AbstractFloat} = zero(T)

struct BetaRGB
    α
    β
end
Distributions.rand(b::BetaRGB) =
    RGB(map(rand, [Beta(b.α[i], b.β[i]) for i = 1:3])...)
function Distributions.logpdf(b::BetaRGB, x::RGB{T}) where {T<:AbstractFloat}
    betas = [Beta(b.α[i], b.β[i]) for i = 1:3]
    xs = [x.r, x.g, x.b]
    return sum(map(i -> logpdf(betas[i], xs[i]), 1:3))
end

# Likelihood
struct NormalRGB{T}
    μ::RGB{T}
    σ::T
end
Distributions.logpdf(d::NormalRGB{T}, x::RGB{T}) where {T<:AbstractFloat} =
    logpdf(Normal(0, d.σ), colordiff(d.μ, x)^2)
# Data
D = 3   # dimension of parameter space
data = [RGB(0.5, 0.0, 0.0), RGB(0.6, 0.0, 0.0), RGB(0.4, 0.0, 0.0)]

#############
# Modelling #
#############

# Log-joint and it's gradient
function logπ(θ::Vector{Float64}) :: Float64
    c = RGB(θ...)
    logprior = logpdf(UniformRGB(), c)
    loglikelihood = sum([logpdf(NormalRGB(t, 5.0), c) for t in data])
    return logprior + loglikelihood
end
∂logπ∂θ(θ::Vector{Float64}) :: Vector{Float64} = collect(gradient(logπ, θ)[1])

# Sampling parameter settings
n_adapts = 1_000
n_samples = 1_000

# Initial points
θ_init = rand(D)

# Define metric space, Hamiltonian and sampling method
metric = UnitEuclideanMetric(D)
h = Hamiltonian(metric, logπ, ∂logπ∂θ)
prop = NUTS(Leapfrog(find_good_eps(h, θ_init)), 5, 1000.0)
adaptor = StanNUTSAdaptor(n_adapts, Preconditioner(metric), NesterovDualAveraging(0.8, prop.integrator.ϵ))

# Sampling
samples = sample(h, prop, θ_init, n_samples, adaptor, n_adapts)

colors = map(s -> RGB(s...), samples)
mean_color = RGB(mean(samples)...)
