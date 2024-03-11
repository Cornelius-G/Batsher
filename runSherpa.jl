using BAT
using Distributions
using IntervalSets
using DensityInterface
using InverseFunctions
using ValueShapes
ENV["JULIA_DEBUG"] = "BAT"

cmd = `./build/test julia OUTPUT=0`
nthreads = Threads.nthreads()
nthreads=1

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
#using BenchmarkTools
#@btime logdensityof(posterior)(rand(posterior.prior))


#posterior = BAT.example_posterior()

include("ram_sampler_v3.jl")


using Optim, AutoDiffOperators, FiniteDifferences
set_batcontext(ad = ADModule(:FiniteDifferences))


samples, chains = bat_sample(posterior, SobolSampler(nsamples=10^3, trafo=NoDensityTransform()))
samples, chains = bat_sample(posterior, SobolSampler(nsamples=10^3))
samples = bat_sample(posterior, MCMCSampling(mcalg = MetropolisHastings(), nsteps = 10^3, nchains = 4, trafo=DoNotTransform())).result

import NestedSamplers
samples = bat_sample(posterior, EllipsoidalNestedSampling(max_ncalls=5e5,))


nchains = 2
g_mode = []
while length(g_mode)< nchains
    try 
        gm = bat_findmode(posterior, OptimAlg(optalg = Optim.LBFGS())).result
        push!(g_mode, gm)
    catch e
        nothing
    end
end


gm = bat_findmode(posterior, OptimAlg(optalg = Optim.LBFGS())).result        


samples = bat_sample(posterior, RAMSampler(nchains=nchains, nsteps=6*10^5, nburnin=2*10^5, trafo=DoNotTransform(), x0=fill(gm.a, nchains)))

samples = bat_sample(posterior, RAMSampler(nchains=3, nsteps=6*10^5, nburnin=3*10^5, trafo=DoNotTransform()))

samples = bat_sample(posterior, PriorImportanceSampler(nsamples=5*10^5))

close.(processes)


using Plots
p = plot(samples.result)
p = plot(samples.result, vsel=vec(5:10))

plot(samples.result, 10, size=(1000, 800), bins=2000, xlims=(0, 0.013))

# Get samples with all weights =1:
n_samples = 500_000#length(samples.result)
samples_no_weights = bat_sample(samples.result, RandResampling(nsamples=n_samples))


p = plot(samples_no_weights.result)

samples_no_weights.result

# run unweighted samples through likelihood again to get hepmc file with events:
processes = [open(cmd, "w+") for i in 1:nthreads]
myf.(samples_no_weights.result.v)

close.(processes)


p = plot(layout = (5, 2), size=(1200, 800), left_margin=15Plots.px)
for i in 1:10
    p = plot!(samples.result, i, bins=100, subplot=i, legend=false, marginalmode=false)
end
p

savefig(p, "output/samples_ram.pdf")


include("BAT_utils.jl")
logposteriorplot(samples)

r = rand(prior)
string(r.a[1])
bitstring(r.a[1])

myf(r)
# #----- Sample flat ------------------------------
for i in 1:400_000
    r = rand(prior)
    y = myf(r)
    #println(y)
end

r1 = (a = [0.11661,0.440481,0.946096,0.347573,0.251482,0.786994,0.16368,0.397118, 0.917472, 0.564124 ], )
r2 = (a = [0.879875,0.528624,0.787269,0.793085,0.91823,0.714834,0.255809,0.812159, 0.535425, 0.085799 ], )

myf(r1)
myf(r2)


#--- Read in Sherpa samples from hypercube -------------
# Function to read the file and parse numbers, returning a Vector{Vector{Real}}
function read_numbers_from_file(filename::String) :: Vector{Vector{Real}}
    # Initialize an empty array to hold arrays of Real numbers from each line
    numbers = Vector{Vector{Real}}()

    # Open the file for reading
    open(filename, "r") do file
        # Iterate over each line in the file
        for line in eachline(file)
            # Split the line into substrings based on space, parse each to Float64, and then explicitly convert to Real
            num_array = parse.(Float64, split(line)) |> x -> Vector{Real}(x)
            # Append the array of Real numbers to the 'numbers' array
            push!(numbers, num_array)
        end
    end

    return numbers
end

# Function to read numbers from file into a Vector{Float64}
function read_numbers_into_vector(filename::String) :: Vector{Float64}
    numbers = Float64[]  # Initialize an empty Vector{Float64}

    open(filename, "r") do file
        for line in eachline(file)
            # Parse each line as Float64 and append to the numbers vector
            push!(numbers, parse(Float64, line))
        end
    end

    return numbers
end


function read_samples(file_samples, file_cs)
    numbers = read_numbers_from_file(file_samples)

    # Convert each sub-array of Real to a specific type if necessary, e.g., Float64
    PV = [Vector{Float64}(n) for n in numbers]

    # Length of your data
    len = length(numbers)
    # Create dummy data for TV, WV, RV, QV
    TV = fill(0.0, len)  # Assuming T is Float64 for simplicity
    WV = fill(1.0, len)  # Assuming weights are Float64 or change to appropriate Real type
    RV = fill(nothing, len)  # Adjust type as necessary for R
    QV = fill(nothing, len)  # Adjust type as necessary for Q

    cs_me = read_numbers_into_vector(file_cs)
    logd = log.(cs_me)

    samples = DensitySampleVector((PV, logd, cs_me, RV, QV))
end

# Example usage
samples_sherpa = read_samples("output/samples.txt", "output/me.txt")
samples_bat = read_samples("output/samples_bat.txt", "output/me_bat.txt")


# Creating a DensitySampleVector
# Note: Adjust types in the ArrayOfSimilarArrays and other vectors as necessary

plot(samples_sherpa)
plot(samples_sherpa, vsel=vec(5:10))

plot(samples_bat)
plot(samples_bat, vsel=vec(5:10))


i = 10
s_bat = [inner[i] for inner in samples_bat.v]
s_sherpa = [inner[i] for inner in samples_sherpa.v]

#p = plot(s_bat, weights=samples_bat.weight, st=:stephist, label="BAT")
#p = plot!(s_sherpa, weights = samples_sherpa.weight, st=:stephist, label="Sherpa")

p = plot(s_bat, st=:stephist, label="BAT")
p = plot!(s_sherpa, st=:stephist, label="Sherpa")


# Send a number to the executable
# number_to_send = "0.7 0.5"
# println(process, number_to_send)
# flush(process)  # Ensure data is sent immediately
# output = String(readavailable(process))
# result = parse(Float64, output)
# close(process)
