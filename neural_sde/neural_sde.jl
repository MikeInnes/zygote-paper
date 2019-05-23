using Flux, DiffEqFlux, StochasticDiffEq, Plots, DiffEqMonteCarlo

u0 = Float32[2.; 0.]
datasize = 30
tspan = (0.0f0,1.0f0)

function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end
t = range(tspan[1],tspan[2],length=datasize)
mp = Float32[0.2,0.2]
function true_noise_func(du,u,p,t)
    du .= mp.*u
end
prob = SDEProblem(trueODEfunc,true_noise_func,u0,tspan)

# Take a typical sample from the mean
monte_prob = MonteCarloProblem(prob)
monte_sol = solve(monte_prob,SOSRI(),num_monte = 100)
monte_sum = MonteCarloSummary(monte_sol)
sde_data = Array(timeseries_point_mean(monte_sol,t))

dudt = Chain(x -> x.^3,
             Dense(2,50,tanh),
             Dense(50,2))
ps = Flux.params(dudt)
n_sde = x->neural_dmsde(dudt,x,mp,tspan,SOSRI(),saveat=t,reltol=1e-1,abstol=1e-1)

pred = n_sde(u0) # Get the prediction using the correct initial condition

dudt_(u,p,t) = Flux.data(dudt(u))
g(u,p,t) = mp.*u
nprob = SDEProblem(dudt_,g,u0,(0.0f0,1.2f0),nothing)

monte_nprob = MonteCarloProblem(nprob)
monte_nsol = solve(monte_nprob,SOSRI(),num_monte = 100)
monte_nsum = MonteCarloSummary(monte_nsol)
#plot(monte_nsol,color=1,alpha=0.3)
p1 = plot(monte_nsum, title = "Neural SDE: Before Training")
scatter!(p1,t,sde_data',lw=3)

scatter(t,sde_data[1,:],label="data")
scatter!(t,Flux.data(pred[1,:]),label="prediction")

function predict_n_sde()
  n_sde(u0)
end
loss_n_sde1() = sum(abs2,sde_data .- predict_n_sde())
loss_n_sde10() = sum([sum(abs2,sde_data .- predict_n_sde()) for i in 1:10])
Flux.back!(loss_n_sde1())

data = Iterators.repeated((), 10)
opt = ADAM(0.025)
cb = function () #callback function to observe training
  sample = predict_n_sde()
  # loss against current data
  display(sum(abs2,sde_data .- sample))
  # plot current prediction against data
  cur_pred = Flux.data(sample)
  pl = scatter(t,sde_data[1,:],label="data")
  scatter!(pl,t,cur_pred[1,:],label="prediction")
  display(plot(pl))
end

# Display the SDE with the initial parameter values.
cb()

Flux.train!(loss_n_sde1 , ps, Iterators.repeated((), 100), opt, cb = cb)
Flux.train!(loss_n_sde10, ps, Iterators.repeated((), 20), opt, cb = cb)

dudt_(u,p,t) = Flux.data(dudt(u))
g(u,p,t) = mp.*u
nprob = SDEProblem(dudt_,g,u0,(0.0f0,1.2f0),nothing)

monte_nprob = MonteCarloProblem(nprob)
monte_nsol = solve(monte_nprob,SOSRI(),num_monte = 100)
monte_nsum = MonteCarloSummary(monte_nsol)
#plot(monte_nsol,color=1,alpha=0.3)
p2 = plot(monte_nsum, title = "Neural SDE: After Training", xlabel="Time")
scatter!(p2,t,sde_data',lw=3,label=["x" "y" "z" "y"])

plot(p1,p2,layout=(2,1))

savefig("neural_sde.pdf")
savefig("neural_sde.png")
