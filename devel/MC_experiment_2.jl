# 1259252040&Courtesy&&&USR&GEN2035&5_1_2026&1000&PATH&GEN&31_12_2035&0_0_0&6000&0_0
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
MCiters = 11
printPrecision = 2
Δ = 5

rnumSample = 3
nnumSample = 5
c1Num = 5
c2Num = 4
c3Num = 2
# global λ_seed = abs.(rand(10) * 3 .+ 3)
# global λ_seed = ones(10) * 3
# global λ_seed = [3.0,4.0,2.0,3.5,2.2,2.0,1.5,2.0,2.0]
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

global NashTime = Array{Any}(undef,2,rnumSample,nnumSample)
global NashSTime = Array{Any}(undef,2,rnumSample,nnumSample)
global NashFairScore = Array{Any}(undef,2,rnumSample,nnumSample)
global NashGiniScore = Array{Any}(undef,2,rnumSample,nnumSample)
global NashAvgDelayScore = Array{Any}(undef,2,rnumSample,nnumSample)
global AceTime = Array{Any}(undef,2,rnumSample,nnumSample)
global AceSTime = Array{Any}(undef,2,rnumSample,nnumSample)
global AceFairScore = Array{Any}(undef,2,rnumSample,nnumSample)
global AceGiniScore = Array{Any}(undef,2,rnumSample,nnumSample)
global AceAvgDelayScore = Array{Any}(undef,2,rnumSample,nnumSample)

# global BestNashTime = Array{Any}(undef,rnumSample,nnumSample)
# global BestNashSTime = Array{Any}(undef,rnumSample,nnumSample)
# global BestNashFairScore = Array{Any}(undef,rnumSample,nnumSample)
# global BestNashGiniScore = Array{Any}(undef,rnumSample,nnumSample)
# global BestNashAvgDelayScore = Array{Any}(undef,rnumSample,nnumSample)

global NashPrimal = Array{Any}(undef,rnumSample,nnumSample)
global AcePrimal = Array{Any}(undef,rnumSample,nnumSample)
global CList = Array{Any}(undef,rnumSample,nnumSample)
global actionListSet = Vector{Any}(undef,rnumSample)

# Dummy
println("Dummy test")
SearchNash(1,2,λ,false,Δ)
SearchCorr(1,2,λ,Δ)
NashBasedOptimizer(1,2,λ,Δ,3)
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
        println("Testing for m=$m and n=$n case.")
        # MNE
        for tt in 1:MCiters
            # global totalTime[tt] = @elapsed global out = SearchNash(r,n,λ,true,Δ)
            global totalTime[tt] = @elapsed global out = NashBasedOptimizer(r,n,λ,Δ,ACEiters)
            global solverTime[tt] = out.solverTime
            global fairScore[tt] = out.fairScore
            global giniScore[tt] = out.giniScore
            global avgDelayScore[tt] = out.avgDelayScore
        end
        global NashTime[:,i,j] = [mean(totalTime[2:end]);std(totalTime[2:end])]*1000
        global NashSTime[:,i,j] = [mean(solverTime[2:end]);std(solverTime[2:end])]*1000
        global NashFairScore[:,i,j] = [mean(fairScore[2:end]);std(fairScore[2:end])/sqrt(MCiters-1)]
        global NashGiniScore[:,i,j] = [mean(giniScore[2:end]);std(giniScore[2:end])/sqrt(MCiters-1)]
        global NashAvgDelayScore[:,i,j] = [mean(avgDelayScore[2:end]);std(avgDelayScore[2:end])/sqrt(MCiters-1)]
        global NashPrimal[i,j] = out.primals

        # global BestNashAvgDelayScore[i,j] = [minimum(avgDelayScore[2:end])]
        # global BestNashFairScore[i,j] = [minimum(fairScore[2:end])]
        # global BestNashGiniScore[i,j] = [minimum(giniScore[2:end])]
        # global BestNashSTime[i,j] = [minimum(solverTime[2:end])]
        # global BestNashTime[i,j] = [minimum(totalTime[2:end])]

        # ACE
        for tt in 1:MCiters
            # global totalTime[tt] = @elapsed global out = NashBasedOptimizer(r,n,λ,Δ,Δ)
            global totalTime[tt] = @elapsed global out = BruteNashBasedOptimizer(r,n,λ,Δ)
            global solverTime[tt] = out.solverTime
            global fairScore[tt] = out.fairScore
            global giniScore[tt] = out.giniScore
            global avgDelayScore[tt] = out.avgDelayScore
        end
        global AceTime[:,i,j] = [mean(totalTime[2:end]);std(totalTime[2:end])]*1000
        global AceSTime[:,i,j] = [mean(solverTime[2:end]);std(solverTime[2:end])]*1000
        global AceFairScore[:,i,j] = [mean(fairScore[2:end]);std(fairScore[2:end])/sqrt(MCiters-1)]
        global AceGiniScore[:,i,j] = [mean(giniScore[2:end]);std(giniScore[2:end])/sqrt(MCiters-1)]
        global AceAvgDelayScore[:,i,j] = [mean(avgDelayScore[2:end]);std(avgDelayScore[2:end])/sqrt(MCiters-1)]
        global AcePrimal[i,j] = reshape(out.primalJoint,ntuple(i->m,n))

        global CList[i,j] = SetC(r,n,λ)
    end
end

global CorrTime1 = Array{Any}(undef,2,c1Num)
global CorrSTime1 = Array{Any}(undef,2,c1Num)
global CorrFairScore1 = Array{Any}(undef,2,c1Num)
global CorrGiniScore1 = Array{Any}(undef,2,c1Num)
global CorrAvgDelayScore1 = Array{Any}(undef,2,c1Num)
global CorrTime2 = Array{Any}(undef,2,c2Num)
global CorrSTime2 = Array{Any}(undef,2,c2Num)
global CorrFairScore2 = Array{Any}(undef,2,c2Num)
global CorrGiniScore2 = Array{Any}(undef,2,c2Num)
global CorrAvgDelayScore2 = Array{Any}(undef,2,c2Num)
global CorrTime3 = Array{Any}(undef,2,c3Num)
global CorrSTime3 = Array{Any}(undef,2,c3Num)
global CorrFairScore3 = Array{Any}(undef,2,c3Num)
global CorrGiniScore3 = Array{Any}(undef,2,c3Num)
global CorrAvgDelayScore3 = Array{Any}(undef,2,c3Num)

global CorrPrimal1 = Array{Any}(undef,c1Num)
global CorrPrimal2 = Array{Any}(undef,c2Num)
global CorrPrimal3 = Array{Any}(undef,c3Num)

for j in 1:size(CorrTime1)[2]
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
    global CorrTime1[:,j] = [mean(totalTime[2:end]);std(totalTime[2:end])]*1000
    global CorrSTime1[:,j] = [mean(solverTime[2:end]);std(solverTime[2:end])]*1000
    global CorrFairScore1[:,j] = [mean(fairScore[2:end]);std(fairScore[2:end])/sqrt(MCiters-1)]
    global CorrGiniScore1[:,j] = [mean(giniScore[2:end]);std(giniScore[2:end])/sqrt(MCiters-1)]
    global CorrAvgDelayScore1[:,j] = [mean(avgDelayScore[2:end]);std(avgDelayScore[2:end])/sqrt(MCiters-1)]
    global CorrPrimal1[j] = reshape(out.primals[1:m^n],ntuple(i->m,n))
end
for j in 1:size(CorrTime2)[2]
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
    global CorrTime2[:,j] = [mean(totalTime[2:end]);std(totalTime[2:end])]*1000
    global CorrSTime2[:,j] = [mean(solverTime[2:end]);std(solverTime[2:end])]*1000
    global CorrFairScore2[:,j] = [mean(fairScore[2:end]);std(fairScore[2:end])/sqrt(MCiters-1)]
    global CorrGiniScore2[:,j] = [mean(giniScore[2:end]);std(giniScore[2:end])/sqrt(MCiters-1)]
    global CorrAvgDelayScore2[:,j] = [mean(avgDelayScore[2:end]);std(avgDelayScore[2:end])/sqrt(MCiters-1)]
    global CorrPrimal2[j] = reshape(out.primals[1:m^n],ntuple(i->m,n))
end
for j in 1:size(CorrTime3)[2]
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
    global CorrTime3[:,j] = [mean(totalTime[2:end]);std(totalTime[2:end])]*1000
    global CorrSTime3[:,j] = [mean(solverTime[2:end]);std(solverTime[2:end])]*1000
    global CorrFairScore3[:,j] = [mean(fairScore[2:end]);std(fairScore[2:end])/sqrt(MCiters-1)]
    global CorrGiniScore3[:,j] = [mean(giniScore[2:end]);std(giniScore[2:end])/sqrt(MCiters-1)]
    global CorrAvgDelayScore3[:,j] = [mean(avgDelayScore[2:end]);std(avgDelayScore[2:end])/sqrt(MCiters-1)]
    global CorrPrimal3[j] = reshape(out.primals[1:m^n],ntuple(i->m,n))
end

for i in 1:rnumSample
    primeNum = "[1,0]"
    numUnit = (primeNum*",")^(i-1)*primeNum
    numSum = "vec(collect(Iterators.product("*numUnit*")))"
    actionListSet[i] = collect.(eval(Meta.parse(numSum)))
end

# Save result in mat
matwrite("MC_result_"*string(Int64.(round.(rand(1)*100)))*".mat",Dict(
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
    "C" => Array.(CList),
    "actionSet" => actionListSet
    # "BestNashAvgDelayScore" => BestNashAvgDelayScore,
    # "BestNashFairScore" => BestNashFairScore,
    # "BestNashGiniScore" => BestNashGiniScore,
    # "BestNashSTime" => BestNashSTime,
    # "BestNashTime" => BestNashTime
); version="v7.4")