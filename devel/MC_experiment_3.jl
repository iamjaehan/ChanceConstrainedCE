# 2830898829&Courtesy&&&USR&45321&5_1_2021&1000&PATH&GEN&31_12_2025&0_0_0&6000&0_0
using correlated
using Plots
using Statistics
using MAT

include("0_GameSetup.jl")
include("CorrBasedOptimizer.jl")
include("SearchNash.jl")
include("NashBasedOptimizer.jl")
include("SearchNashBrute.jl")
include("BruteNashBasedOptimizer.jl")

# Parameter setting
r = 1 # Number of runways
n = 2 # Number of agents
m = 2^r
dF = 5 # Delay factor
eF = dF*100 # Hazard factor
sF = dF # All stop factor
λ = ones(n,1)
# λ = rand(l+d,1) * 3
ACEiters = 100
MCiters = 31
printPrecision = 2
Δ = 5

rnumSample = 3
nnumSample = 6
c1Num = 6
c2Num = 3
c3Num = 2

global λ_seed = [5.743634505273297
4.786110901049589
3.457826833269652
3.854818609208209
4.814772972788456
4.546629163427745
4.394415300232811
5.49925856558372
3.0008753071514014
4.719508107688774]

global totalTime = Vector{Any}(undef,MCiters)
global solverTime = Vector{Any}(undef,MCiters)
global fairScore = Vector{Any}(undef,MCiters)
global giniScore = Vector{Any}(undef,MCiters)
global avgDelayScore = Vector{Any}(undef,MCiters)

global NashTime = Array{Any}(undef,rnumSample,nnumSample,6)
global NashSTime = Array{Any}(undef,rnumSample,nnumSample,6)
global NashFairScore = Array{Any}(undef,rnumSample,nnumSample,6)
global NashGiniScore = Array{Any}(undef,rnumSample,nnumSample,6)
global NashAvgDelayScore = Array{Any}(undef,rnumSample,nnumSample,6)
global AceTime = Array{Any}(undef,rnumSample,nnumSample,6)
global AceSTime = Array{Any}(undef,rnumSample,nnumSample,6)
global AceFairScore = Array{Any}(undef,rnumSample,nnumSample,6)
global AceGiniScore = Array{Any}(undef,rnumSample,nnumSample,6)
global AceAvgDelayScore = Array{Any}(undef,rnumSample,nnumSample,6)

global NashPrimal = Array{Any}(undef,rnumSample,nnumSample)
global AcePrimal = Array{Any}(undef,rnumSample,nnumSample)
global CList = Array{Any}(undef,rnumSample,nnumSample)
global actionListSet = Vector{Any}(undef,rnumSample)

# Dummy
println("Dummy test")
SearchNash(1,2,λ,false,Δ)
SearchCorr(1,2,λ,Δ)
# NashBasedOptimizer(1,2,λ,Δ,3)
BruteNashBasedOptimizer(1,2,λ,Δ)

println("========= Test begins!! ==========")

for i in 1:rnumSample # runway num
    for j in 1:nnumSample # player num -1
        global r = i
        global n = j+1
        global m = 2^r
        global Δ = 0.1*n
        global λ = λ_seed[1:n]
        println("................")

        # MNE
        for tt in 1:MCiters
            global totalTime[tt] = @elapsed global out = SearchNash(r,n,λ,true,Δ)
            #global totalTime[tt] = @elapsed global out = NashBasedOptimizer(r,n,λ,Δ,ACEiters)
            global solverTime[tt] = out.solverTime
            global fairScore[tt] = out.fairScore
            global giniScore[tt] = out.giniScore
            global avgDelayScore[tt] = out.avgDelayScore
        end
        global NashTime[i,j,:] = [quantile(totalTime[2:end],[0,0.25,0.5,0.75,1]);std(totalTime[2:end])]
        global NashSTime[i,j,:] = [quantile(solverTime[2:end],[0,0.25,0.5,0.75,1]);std(solverTime[2:end])]
        global NashFairScore[i,j,:] = [quantile(fairScore[2:end],[0,0.25,0.5,0.75,1]);std(fairScore[2:end])]
        global NashGiniScore[i,j,:] = [quantile(giniScore[2:end],[0,0.25,0.5,0.75,1]);std(giniScore[2:end])]
        global NashAvgDelayScore[i,j,:] = [quantile(avgDelayScore[2:end],[0,0.25,0.5,0.75,1]);std(avgDelayScore[2:end])]
        global NashPrimal[i,j] = out.primals

        # ACE
        for tt in 1:MCiters
            global totalTime[tt] = @elapsed global out = NashBasedOptimizer(r,n,λ,Δ,ACEiters)
            # global totalTime[tt] = @elapsed global out = BruteNashBasedOptimizer(r,n,λ,Δ)
            global solverTime[tt] = out.solverTime
            global fairScore[tt] = out.fairScore
            global giniScore[tt] = out.giniScore
            global avgDelayScore[tt] = out.avgDelayScore
        end
        global AceTime[i,j,:] = [quantile(totalTime[2:end],[0,0.25,0.5,0.75,1]);std(totalTime[2:end])]
        global AceSTime[i,j,:] = [quantile(solverTime[2:end],[0,0.25,0.5,0.75,1]);std(solverTime[2:end])]
        global AceFairScore[i,j,:] = [quantile(fairScore[2:end],[0,0.25,0.5,0.75,1]);std(fairScore[2:end])]
        global AceGiniScore[i,j,:] = [quantile(giniScore[2:end],[0,0.25,0.5,0.75,1]);std(giniScore[2:end])]
        global AceAvgDelayScore[i,j,:] = [quantile(avgDelayScore[2:end],[0,0.25,0.5,0.75,1]);std(avgDelayScore[2:end])]
        global AcePrimal[i,j] = reshape(out.primalJoint,ntuple(i->m,n))

        global CList[i,j] = SetC(r,n,λ)
    end
end

global CorrTime1 = Array{Any}(undef,c1Num,6)
global CorrSTime1 = Array{Any}(undef,c1Num,6)
global CorrFairScore1 = Array{Any}(undef,c1Num,6)
global CorrGiniScore1 = Array{Any}(undef,c1Num,6)
global CorrAvgDelayScore1 = Array{Any}(undef,c1Num,6)
global CorrTime2 = Array{Any}(undef,c2Num,6)
global CorrSTime2 = Array{Any}(undef,c2Num,6)
global CorrFairScore2 = Array{Any}(undef,c2Num,6)
global CorrGiniScore2 = Array{Any}(undef,c2Num,6)
global CorrAvgDelayScore2 = Array{Any}(undef,c2Num,6)
global CorrTime3 = Array{Any}(undef,c3Num,6)
global CorrSTime3 = Array{Any}(undef,c3Num,6)
global CorrFairScore3 = Array{Any}(undef,c3Num,6)
global CorrGiniScore3 = Array{Any}(undef,c3Num,6)
global CorrAvgDelayScore3 = Array{Any}(undef,c3Num,6)

global CorrPrimal1 = Array{Any}(undef,c1Num)
global CorrPrimal2 = Array{Any}(undef,c2Num)
global CorrPrimal3 = Array{Any}(undef,c3Num)

for j in 1:size(CorrTime1)[1]
    global r = 1
    global n = j+1
    global m = 2^r
    global λ = λ_seed[1:n]
    println("................")

    # CE
    for tt in 1:MCiters
        global totalTime[tt] = @elapsed global out = SearchCorr(r,n,λ,Δ)
        global solverTime[tt] = out.solverTime
        global fairScore[tt] = out.fairScore
        global giniScore[tt] = out.giniScore
        global avgDelayScore[tt] = out.avgDelayScore
    end
    global CorrTime1[j,:] = [quantile(totalTime[2:end],[0,0.25,0.5,0.75,1]);std(totalTime[2:end])]
    global CorrSTime1[j,:] = [quantile(solverTime[2:end],[0,0.25,0.5,0.75,1]);std(solverTime[2:end])]
    global CorrFairScore1[j,:] = [quantile(fairScore[2:end],[0,0.25,0.5,0.75,1]);std(fairScore[2:end])]
    global CorrGiniScore1[j,:] = [quantile(giniScore[2:end],[0,0.25,0.5,0.75,1]);std(giniScore[2:end])]
    global CorrAvgDelayScore1[j,:] = [quantile(avgDelayScore[2:end],[0,0.25,0.5,0.75,1]);std(avgDelayScore[2:end])]
    global CorrPrimal1[j] = reshape(out.primals[1:m^n],ntuple(i->m,n))
end
for j in 1:size(CorrTime2)[1]
    global r = 2
    global n = j+1
    global m = 2^r
    global λ = λ_seed[1:n]
    println("................")

    # CE
    for tt in 1:MCiters
        global totalTime[tt] = @elapsed global out = SearchCorr(r,n,λ,Δ)
        global solverTime[tt] = out.solverTime
        global fairScore[tt] = out.fairScore
        global giniScore[tt] = out.giniScore
        global avgDelayScore[tt] = out.avgDelayScore
    end
    global CorrTime2[j,:] = [quantile(totalTime[2:end],[0,0.25,0.5,0.75,1]);std(totalTime[2:end])]
    global CorrSTime2[j,:] = [quantile(solverTime[2:end],[0,0.25,0.5,0.75,1]);std(solverTime[2:end])]
    global CorrFairScore2[j,:] = [quantile(fairScore[2:end],[0,0.25,0.5,0.75,1]);std(fairScore[2:end])]
    global CorrGiniScore2[j,:] = [quantile(giniScore[2:end],[0,0.25,0.5,0.75,1]);std(giniScore[2:end])]
    global CorrAvgDelayScore2[j,:] = [quantile(avgDelayScore[2:end],[0,0.25,0.5,0.75,1]);std(avgDelayScore[2:end])]
    global CorrPrimal2[j] = reshape(out.primals[1:m^n],ntuple(i->m,n))
end
for j in 1:size(CorrTime3)[1]
    global r = 3
    global n = j+1
    global m = 2^r
    global λ = λ_seed[1:n]
    println("................")

    # CE
    for tt in 1:MCiters
        global totalTime[tt] = @elapsed global out = SearchCorr(r,n,λ,Δ)
        global solverTime[tt] = out.solverTime
        global fairScore[tt] = out.fairScore
        global giniScore[tt] = out.giniScore
        global avgDelayScore[tt] = out.avgDelayScore
    end
    global CorrTime3[j,:] = [quantile(totalTime[2:end],[0,0.25,0.5,0.75,1]);std(totalTime[2:end])]
    global CorrSTime3[j,:] = [quantile(solverTime[2:end],[0,0.25,0.5,0.75,1]);std(solverTime[2:end])]
    global CorrFairScore3[j,:] = [quantile(fairScore[2:end],[0,0.25,0.5,0.75,1]);std(fairScore[2:end])]
    global CorrGiniScore3[j,:] = [quantile(giniScore[2:end],[0,0.25,0.5,0.75,1]);std(giniScore[2:end])]
    global CorrAvgDelayScore3[j,:] = [quantile(avgDelayScore[2:end],[0,0.25,0.5,0.75,1]);std(avgDelayScore[2:end])]
    global CorrPrimal3[j] = reshape(out.primals[1:m^n],ntuple(i->m,n))
end

# Save result in mat
matwrite("MC_result.mat",Dict(
    "NashTime" => NashTime,
    "AceTime" => AceTime,
    "NashSTime" => NashSTime,
    "AceSTime" => AceSTime,
    "NashFairScore" => NashFairScore,
    "NashGiniScore" => NashGiniScore,
    "NashAvgDelayScore" => NashAvgDelayScore,
    "AceFairScore" => AceFairScore,
    "AceGiniScore" => AceGiniScore,
    "AceAvgDelayScore" => AceAvgDelayScore,
    "CorrTime1" => CorrTime1,
    "CorrTime2" => CorrTime2,
    "CorrTime3" => CorrTime3,
    "CorrSTime1" => CorrSTime1,
    "CorrSTime2" => CorrSTime2,
    "CorrSTime3" => CorrSTime3,
    "CorrFairScore1" => CorrFairScore1,
    "CorrFairScore2" => CorrFairScore2,
    "CorrFairScore3" => CorrFairScore3,
    "CorrGiniScore1" => CorrGiniScore1,
    "CorrGiniScore2" => CorrGiniScore2,
    "CorrGiniScore3" => CorrGiniScore3,
    "CorrAvgDelayScore1" => CorrAvgDelayScore1,
    "CorrAvgDelayScore2" => CorrAvgDelayScore2,
    "CorrAvgDelayScore3" => CorrAvgDelayScore3,
    "NashPrimal" => NashPrimal,
    "AcePrimal" => AcePrimal,
    "CorrPrimal1" => CorrPrimal1,
    "CorrPrimal2" => CorrPrimal2,
    "CorrPrimal3" => CorrPrimal3,
    "Lambda" => λ_seed,
    "C" => Array.(CList)
); version="v7.4")
