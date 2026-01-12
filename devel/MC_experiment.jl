using correlated
using Plots
using Statistics

include("0_GameSetup.jl")
include("CorrBasedOptimizer.jl")
include("SearchNash.jl")
include("NashBasedOptimizer.jl")

# Parameter setting
r = 1 # Number of runways
n = 2 # Number of agents
dF = 5 # Delay factor
eF = dF*100 # Hazard factor
sF = dF # All stop factor
λ = ones(n,1)
# λ = rand(l+d,1) * 3
ACEiters = 50
MCiters = 3
printPrecision = 2

rnumSample = 3
nnumSample = 3

global MNETr = Vector{Any}(undef,MCiters)
global ACETr = Vector{Any}(undef,MCiters)
global CETr = Vector{Any}(undef,MCiters)

global MNEStr = Vector{Any}(undef,MCiters)
global ACEStr = Vector{Any}(undef,MCiters)
global CEStr = Vector{Any}(undef,MCiters)

global MNEScorer = Vector{Any}(undef,MCiters)
global ACEScorer = Vector{Any}(undef,MCiters)
global CEScorer = Vector{Any}(undef,MCiters)

global ResultRecord = Array{Any}(undef,rnumSample,nnumSample,12)
global ResultRecordStd = Array{Any}(undef,rnumSample,nnumSample,9)

# Dummy
println("Dummy test")
SearchNash(1,2,λ,false)
SearchCorr(1,2,λ)

println("========= Test begins!! ==========")

# global λ_seed = abs.(randn(5) * 3 .+ 6)
global λ_seed = ones(10)

for i in 1:rnumSample # runway num
    for j in 1:nnumSample # player num -1
        for tt in 1:MCiters
            global r = i
            global n = j+1
            global λ = λ_seed[1:n]
            println("................")
            global MNETr[tt] = @elapsed global MNEout = SearchNash(r,n,λ,false)
            global ACETr[tt] = @elapsed global ACEout = NashBasedOptimizer(r,n,λ,ACEiters)
            global CETr[tt] = @elapsed global CEout = SearchCorr(r,n,λ)

            global MNEStr[tt] = MNEout.solverTime
            global ACEStr[tt] = ACEout.solverTime
            global CEStr[tt] = CEout.solverTime

            global MNEScorer[tt] =MNEout.score
            global ACEScorer[tt] =ACEout.score
            global CEScorer[tt] =CEout.score
        end
        localResult = [MNETr';ACETr';CETr';MNEStr';ACEStr';CEStr';MNEScorer';ACEScorer';CEScorer']
        localResultAvg = mean(localResult,dims=2)
        localResultStd = std(localResult,dims=2)
        println("TTIME ::: ",round.(localResultAvg[1:3]*1000,digits=printPrecision))
        println("STIME ::: ",round.(localResultAvg[4:6]*1000,digits=printPrecision))
        println("SCORE ::: ",round.(localResultAvg[7:end],digits=printPrecision))
        println("TTIMESTD ::: ",round.(localResultStd[1:3]*1000,digits=printPrecision))

        global ResultRecord[i,j,:] = [localResultAvg;MNEout.varsize;ACEout.varsize;CEout.varsize]
        global ResultRecordStd[i,j,:] = localResultStd
    end
end

ResultRecord[:,:,1:6] = ResultRecord[:,:,1:6] * 1000
ResultRecordStd[:,:,1:6] = ResultRecordStd[:,:,1:6] * 1000

totalSampleNum = rnumSample * nnumSample

p1 = plot()
plot!(p1,1:totalSampleNum,ResultRecord[:,:,1]'[:],label="MNE",yaxis=:log)
plot!(p1,1:totalSampleNum,ResultRecord[:,:,2]'[:],label="ACE",yaxis=:log)
plot!(p1,1:totalSampleNum,ResultRecord[:,:,3]'[:],label="CE",yaxis=:log)
title!(p1,"Total computation time")
xlabel!(p1,"Scenario number")
ylabel!(p1,"Computation time [ms]")

p2 = plot()
plot!(p2,1:totalSampleNum,ResultRecord[:,:,4]'[:],label="MNE",yaxis=:log)
plot!(p2,1:totalSampleNum,ResultRecord[:,:,5]'[:],label="ACE",yaxis=:log)
plot!(p2,1:totalSampleNum,ResultRecord[:,:,6]'[:],label="CE",yaxis=:log)
title!(p2,"Solver time")
xlabel!(p2,"Scenario number")
ylabel!(p2,"Computation time [ms]")

p3 = plot()
plot!(p3,1:totalSampleNum,ResultRecord[:,:,7]'[:],label="MNE")
plot!(p3,1:totalSampleNum,ResultRecord[:,:,8]'[:],label="ACE")
plot!(p3,1:totalSampleNum,ResultRecord[:,:,9]'[:],label="CE")
title!(p3,"Cost")
xlabel!(p3,"Scenario number")
ylabel!(p3,"Cost")

display(p1)
display(p2)
display(p3)

savefig(p1,"test1");
savefig(p2,"test2");
savefig(p3,"test3");
