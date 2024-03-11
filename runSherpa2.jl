using BAT
using AdvancedHMC
using Dates
using Distributions
using IntervalSets
using DensityInterface
using InverseFunctions
using ValueShapes
using ThreadPinning

ENV["JULIA_DEBUG"] = "BAT"
#ENV["GKSwstype"] = "100" 

pinthreads(:cores)

cmd = `./build/test julia nohepmc OUTPUT=0`
cmd2 = `./build/test julia OUTPUT=0`

@show nthreads = Threads.nthreads()
#nthreads=1

processes = [open(cmd, "w+") for i in 1:nthreads]#open a process, with read and write access

# get pid of the processes to track in taskmanager
pid = getpid.(processes)


prior = BAT.NamedTupleDist(
    a = fill(0..1, 10),
)

noshape = inverse(varshape(prior))


function myf(params)
    i = Threads.threadid()
    process = processes[i]
    float_array = noshape(params)
    input_string = join([string(x) for x in float_array], " ")
    #println("input: ", input_string)
    #a = readavailable(process)
    #println("a: ", a)
    println(process, input_string)
    flush(process)  # Ensure data is sent immediately

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


include("ram_sampler_v3.jl")

t1 = Dates.now() 

@info "Starting sampling at "*Dates.format(t1, "dd-mm-Y HH:MM:SS.ss")

#samples = bat_sample(posterior, MCMCSampling(mcalg = MetropolisHastings(), nsteps = 6*10^6, nchains = 4, trafo = DoNotTransform()))#, convergence = BAT.AssumeConvergence()))

samples = bat_sample(posterior, RAMSampler(nchains=6, nsteps=2*10^6, nburnin=8*10^5, trafo=DoNotTransform()))

#samples = bat_sample(posterior, PriorImportanceSampler(nsamples=15*10^6))
#using Optim, AutoDiffOperators, FiniteDifferences
#set_batcontext(ad = ADModule(:FiniteDifferences))
#samples = bat_sample(posterior, MCMCSampling(mcalg = HamiltonianMC(), trafo=PriorToUniform()))

close.(processes)


@info "Start plotting"
using Plots
p1 = plot(samples.result, dpi=600);
p2 = plot(samples.result, vsel=vec(5:10), dpi=600);

savefig(p1, "samples1.png")
savefig(p2, "samples2.png") 

include("BAT_utils.jl")
p3 = logposteriorplot(samples, dpi=600);
savefig(p3, "logposterior.png")

# Get samples with all weights =1:
n_samples = 500_000#length(samples.result)
@info "Start resampling"
samples_no_weights = bat_sample(samples.result, RandResampling(nsamples=n_samples))

@info "Start generating hepmc events"
# run unweighted samples through likelihood again to get hepmc file with events:
processes = [open(cmd2, "w+") for i in 1:nthreads]
myf.(samples_no_weights.result.v)
close.(processes)
t2 = Dates.now()




function ms_to_hms_str(milliseconds)
    total_seconds = div(milliseconds, 1000)
    hours, remainder = divrem(total_seconds, 3600)
    minutes, seconds = divrem(remainder, 60)
    return string(hours, " hour(s), ", minutes, " minute(s), ", seconds, " second(s)")
end

@info "Finished sampling at " * Dates.format(t2, "dd-mm-Y HH:MM:SS.ss") * " after "*ms_to_hms_str(Dates.value(t2 - t1))


#@info "Started python plotting"
#run(`python Batsher/plots.py`)