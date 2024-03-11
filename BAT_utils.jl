function filter_by_chainID(samples::DensitySampleVector, chainids)
    has_chainid(c...) = x -> x.info.chainid in reduce(vcat, c)
    return filter(has_chainid(chainids), samples)
end

function filter_by_chainID(generator::BAT.MCMCSampleGenerator, chainids)
    has_chainid(c...) = x -> x.info.id in reduce(vcat, c)
    filtered_chains = filter(has_chainid(chainids), generator.chains)
    return BAT.MCMCSampleGenerator(copy(filtered_chains))
end


function filter_by_mean(samples::DensitySampleVector, condition)
    samples_by_chain = [filter_by_chainID(samples, cid) for cid in _unique_chainids(samples)]
    
    means = [mean(x.logd) for x in samples_by_chain]
    idxs = findall(condition, means)
    cids = _unique_chainids(samples)[idxs]

    return filter_by_chainID(samples, cids)
end

function filter_by_mean(samples::DensitySampleVector, generator::BAT.MCMCSampleGenerator, condition)
    samples_by_chain = [filter_by_chainID(samples, cid) for cid in _unique_chainids(samples)]
    
    means = [mean(x.logd) for x in samples_by_chain]
    idxs = findall(condition, means)
    cids = _unique_chainids(samples)[idxs]

    return (filter_by_chainID(samples, cids), filter_by_chainID(generator, cids))
end

@userplot LogPosteriorPlot

function _unique_chainids(x)
    if isa(x, DensitySampleVector)
        return unique(getproperty.(x.info,:chainid))
    elseif isa(x, DensitySample)
        return unique(x.info.chainid)
    else
        return unique(getproperty.(x.result.info,:chainid))
    end
end

@recipe function f(
    lpp::LogPosteriorPlot; 
    chainids=_unique_chainids(lpp.args[1]),
)
    samples = isa(lpp.args[1], DensitySampleVector) ? lpp.args[1] : lpp.args[1].result

    for cid in chainids
        suc = filter_by_chainID(samples, cid)

        @series begin
            seriestype --> :line
            label --> "Chain ID: $cid"
            color --> Int(cid)
            xguide --> "Steps"
            yguide --> "log(posterior)"
            legend --> :outerright
            suc.logd
        end
    end
end

#=
# Usage:

logposteriorplot(samples) # plot all chainids
logposteriorplot(samples, chainids=(27, 80)) # use Tuple
logposteriorplot(samples, chainids=80) # or use Array
samples2 = filter_by_chainID(samples.result, [27, 80])
generator2 = filter_by_chainID(samples.generator, [27, 80])
logposteriorplot(samples2)
samples3 = filter_by_mean(samples.result, x-> -190> x>-228)
is_converged = BAT.check_convergence(BrooksGelmanConvergence(), samples_filtered)
=#