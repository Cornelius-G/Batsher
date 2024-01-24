using Pkg
# DEPOT_PATH[1]="/net/e4-nfs-home.e4.physik.tu-dortmund.de/home/slacagnina/.juliakiss/"
# Pkg.activate("/net/e4-nfs-home.e4.physik.tu-dortmund.de/home/slacagnina/.juliakiss/environments/kiss")
# "julia.environmentPath": "/net/e4-nfs-home.e4.physik.tu-dortmund.de/home/slacagnina/kiss",
using BAT
using Distributions
using IntervalSets
using DensityInterface
using InverseFunctions
using ValueShapes
ENV["JULIA_DEBUG"] = "BAT"

cmd = `./build/test OUTPUT=0`
nthreads = Threads.nthreads()
nthreads=1

#process = open(cmd, "w+") #open a process, with read and write access
processes = [open(cmd, "w+") for i in 1:nthreads]#open a process, with read and write access

pid = getpid(processes[1])

prior = BAT.NamedTupleDist(
    a = fill(0..1, 5),
)

noshape = inverse(varshape(prior))


function myf(params)
    i = Threads.threadid()
    process = processes[i]
    float_array = noshape(params)
    input_string = join([string(x) for x in float_array], " ")
    #println("input: ", input_string)
    println(process, input_string)
    #flush(process)  # Ensure data is sent immediately

    available_data = readavailable(process)
    #println("available_data: ", available_data)
    result = if !isempty(available_data)
        #output = String(readavailable(process))
        output = String(available_data)
        #println(output)
        result = parse(Float64, output)
        #println("result = ", result)
        result
    else
        #sleep(0.1)
        println("I've been sleeping")
        0  # Sleep briefly to avoid busy-waiting
    end

    # output = String(readavailable(process))
    # result = parse(Float64, output)
    # println("result: ", result)
    # return result
end 

myf(rand(prior))
params = rand(prior)

likelihood = logfuncdensity(params -> begin
    log(myf(params))
end)

posterior = PosteriorMeasure(likelihood, prior);

logdensityof(posterior)(rand(prior))
#using BenchmarkTools
#@btime logdensityof(posterior)(rand(posterior.prior))


#posterior = BAT.example_posterior()

include("ram_sampler_v3.jl")


using Optim, AutoDiffOperators, FiniteDifferences
set_batcontext(ad = ADModule(:FiniteDifferences))
bat_findmode(posterior, OptimAlg(optalg = Optim.LBFGS()))

samples, chains = bat_sample(posterior, SobolSampler(nsamples=10^3, trafo=NoDensityTransform()))
samples, chains = bat_sample(posterior, SobolSampler(nsamples=10^3))
samples = bat_sample(posterior, MCMCSampling(mcalg = MetropolisHastings(), nsteps = 10^3, nchains = 4, trafo=DoNotTransform())).result

import NestedSamplers
samples = bat_sample(posterior, EllipsoidalNestedSampling(max_ncalls=5e5,))

samples = bat_sample(posterior, RAMSampler(nchains=2))

close.(processes)


using Plots
p = plot(samples.result)


n_samples = length(samples.result)
samples_no_weights = bat_sample(samples.result, RandResampling(nsamples=n_samples))

processes = [open(cmd, "w+") for i in 1:nthreads]

myf.(samples_no_weights.result.v)

close.(processes)

# Send a number to the executable
# number_to_send = "0.7 0.5"
# println(process, number_to_send)
# flush(process)  # Ensure data is sent immediately
# output = String(readavailable(process))
# result = parse(Float64, output)
# close(process)
