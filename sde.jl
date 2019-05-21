using StochasticDiffEq, Flux, DiffEqFlux, DiffEqOperators, CuArrays, LinearAlgebra
ts = 0.0:0.1:10.0
data = [sin.(0.0:0.1:10.0) .+ rand.() cos.(0.0:0.1:10.0) .+ rand.()]'
x = data[:,1]

using Plots
plot(ts,data')

tspan = Float32.((0.0f0,10.0f0))
dudt = Chain(Dense(2,50,tanh),Dense(50,2))
p = DiffEqFlux.destructure(dudt)
dudt_(u::TrackedArray,p,t) =  DiffEqFlux.restructure(dudt,p)(u)
dudt_(u::AbstractArray,p,t) = Flux.data(
                                  DiffEqFlux.restructure(dudt,Flux.data(p))(u))
g(u,p,t) = 3.0 .* u
ff = SDEFunction(dudt_,g)
prob = SDEProblem(ff,g,x,tspan,p)
sol = solve(prob,SOSRI(),saveat=ts)
loss_reduction(sol) = [sum(abs2, norm(sol[i].-data[i]) for i in length(sol))]
diffeq_fd(p,loss_reduction,1,prob,SOSRI();u0=x,saveat=ts)

function loss_fd()
    diffeq_fd(p,loss_reduction,1,prob,SOSRI();u0=x,saveat=ts)[1]
end

opt = ADAM(0.1)
cb = function () #callback function to observe training
  res = diffeq_fd(p,loss_reduction,1,prob,SOSRI();u0=x,saveat=ts)
  display(res)
  # using `remake` to re-create our `prob` with current parameters `p`
  display(plot(solve(remake(prob,p=Flux.data(p)),SOSRI(),saveat=ts),ylim=(-3,3)))
  plot!(ts,data')
end

# Display the ODE with the initial parameter values.
cb()
ps = Flux.Params([p])
Flux.train!(loss_fd, ps, Iterators.repeated((), 100), opt, cb = cb)
