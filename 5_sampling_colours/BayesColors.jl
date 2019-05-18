using Colors, Distributions, Bijectors, Zygote, StatsPlots, MCMCChains
using StatsFuns: logsumexp, logit, logistic
using Revise, AdvancedHMC

########
# Data #
########
const DATA = [RGB(0.5, 0.0, 0.0), RGB(0.6, 0.0, 0.0), RGB(0.4, 0.0, 0.0)]
const DATA_MIXTURE = vcat(DATA, [RGB(0.5, 0.0, 1.0), RGB(0.6, 0.0, 1.0), RGB(0.4, 0.0, 1.0)])

#######################
# Color distributions #
#######################

struct UniformRGB end
Distributions.rand(::UniformRGB) = RGB{Float64}(rand(3)...)
Distributions.logpdf(::UniformRGB, ::RGB{Float64}) = zero(Float64)

# struct BetaRGB
#     α
#     β
# end
# Distributions.rand(b::BetaRGB) =
#     RGB(map(rand, [Beta(b.α[i], b.β[i]) for i = 1:3])...)
# function Distributions.logpdf(b::BetaRGB, x::RGB{T}) where {T<:AbstractFloat}
#     betas = [Beta(b.α[i], b.β[i]) for i = 1:3]
#     xs = [x.r, x.g, x.b]
#     return sum(map(i -> logpdf(betas[i], xs[i]), 1:3))
# end

function Bijectors.logpdf_with_trans(d, c::RGB{Float64})
    θ = [c.r, c.g, c.b]
    return logpdf(d, c) + sum(log.(θ .* (one(Float64) .- θ)))
end

struct NormalRGB
    μ::RGB{Float64}
    σ::Float64
end
Distributions.logpdf(d::NormalRGB, x::RGB{Float64}) = logpdf(Normal(zero(Float64), d.σ), colordiff(d.μ, x)^2)

###################
# Sampling helper #
###################

function sample_colors(θ_init, logjoint, grad; stepsize=1e-3, n_samples=2_000, n_adapts=1_000)
    # Dimension
    dim = length(θ_init)

    # Define metric space, Hamiltonian and sampling method
    metric = DiagEuclideanMetric(dim)
    h = Hamiltonian(metric, logjoint, grad)
    # prop = NUTS(Leapfrog(find_good_eps(h, θ_init)), 5, 1000.0)
    prop = StaticTrajectory(Leapfrog(stepsize), 20)
    # adaptor = StanNUTSAdaptor(n_adapts, Preconditioner(metric), NesterovDualAveraging(0.8, prop.integrator.ϵ))

    # Sampling
    # samples = sample(h, prop, θ_init, n_samples, adaptor, n_adapts; progress=true)
    samples = sample(h, prop, θ_init, n_samples; progress=true)
    return samples
end

############################
# Model 1: single Gaussian #
############################
# Log-joint and it's gradient
function logπ(θ::Vector{Float64}) :: Float64
    c = RGB{Float64}(θ...)
    logprior = logpdf(UniformRGB(), c)
    loglikelihood = sum([logpdf(NormalRGB(t, 5one(Float64)), c) for t in  DATA])
    return logprior + loglikelihood
end
# @code_warntype logπ(rand(3))
∂logπ∂θ(θ::Vector{Float64}) :: Vector{Float64} = vec(collect(gradient(logπ, θ)[1]))
# @code_warntype ∂logπ∂θ(rand(6))
samples = sample_colors(rand(3), logπ, ∂logπ∂θ)
plot(
    Chains(
        reshape(hcat(samples...)', (2_000, 3, 1)),
        ["c.r", "c.g", "c.b"]
    )
)

colors = map(s -> RGB(s...), samples)
mean_samples = mean(samples)
mean_color = RGB(mean_samples...)

###################################
# Model 1: mixture of 2 Gaussians #
###################################
# Log-joint and it's gradient
logπ_mixture(θlink::Vector{Float64}) = logπ_mixture(θlink[1:3], θlink[4:6])
# @Mike: I cannot define `logπ_mixture(θ::Vector{Float64})` directly
#        and construct colors by `c1 = RGB(θ[1:3]...)`, which gives me
#        error when calling `gradient`.
function logπ_mixture(θ1link::Vector{Float64}, θ2link::Vector{Float64}) :: Float64
    p = ones(Float64) / 2
    σ = 20one(Float64)
    c1 = RGB{Float64}(logistic.(θ1link)...)
    c2 = RGB{Float64}(logistic.(θ2link)...)
    logprior = logpdf_with_trans(UniformRGB(), c1) + logpdf_with_trans(UniformRGB(), c2)
    loglikelihood = sum(
        [logsumexp([logpdf(NormalRGB(t, σ), c1) - log(2one(Float64)),
                    logpdf(NormalRGB(t, σ), c2) - log(2one(Float64))]) for t in DATA_MIXTURE]
    )
    return logprior + loglikelihood
end
# @code_warntype logπ(rand(6))
∂logπ∂θ_mixture(θlink::Vector{Float64}) :: Vector{Float64} = ∂logπ∂θ_mixture(θlink[1:3], θlink[4:6])
∂logπ∂θ_mixture(θ1link::Vector{Float64}, θ2link::Vector{Float64}) :: Vector{Float64} =
    vcat(map(collect, gradient(logπ_mixture, θ1link, θ2link))...)
# @code_warntype ∂logπ∂θ_mixture(rand(6))
sampleslink_mixture = sample_colors(randn(6), logπ_mixture, ∂logπ∂θ_mixture; stepsize=1e-2)
samples_mixture = map(s -> logistic.(s), sampleslink_mixture)
plot(
    Chains(
        reshape(hcat(samples_mixture...)', (2_000, 6, 1)),
        ["c1.r", "c1.g", "c1.b", "c2.r", "c2.g", "c2.b"]
    )
)

colors_mixture = map(s -> [RGB(s[1:3]...), RGB(s[4:6]...)], samples_mixture)
map(cs->cs[1], colors_mixture)
map(cs->cs[2], colors_mixture)
mean_samples_mixture = mean(samples_mixture)
mean_color = [RGB(mean_samples_mixture[1:3]...), RGB(mean_samples_mixture[4:6]...)]
