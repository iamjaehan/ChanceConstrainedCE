module correlated

# The module re-includes several devel/*.jl files a second time via
# init.jl -> exp1.jl/exp2.jl (each of which also does `using correlated`
# and its own top-level `include(...)` of files already pulled in above).
# That duplicate-definition pattern is fine for interactive/script use but
# is a hard error under Julia's module precompilation, so opt out of it.
__precompile__(false)

using Symbolics: Symbolics, @variables
using ParametricMCPs: ParametricMCPs, ParametricMCP
using BlockArrays: BlockArray, Block, mortar, blocks
using LinearAlgebra: norm_sqr

include("parametric_game.jl")
export ParametricOptimizationProblem, solve, total_dim
include("parametric_optimization_problem.jl")
export ParametricGame
include("../devel/0_GameSetup.jl")
export SetC, generateAseq, BuildCorrConstraintMap, GetActiveConstraints, TopKActiveCEByMu, CEKeyList, GetActiveCEConstraints, CorrPacker, TopKCEByMuPSigma, TopKCEByMuSigma, TopKCEByMu, RandomKCE, PrintCEList, GetCEConstraintInfo, TopKCEBySigma
include("../devel/SearchNash.jl")
export SearchNash, PrepNash, SolveNash
include("../devel/NashBasedOptimizer.jl")
export NashBasedOptimizer
include("../devel/CorrBasedOptimizer.jl")
export SearchCorr
include("../devel/SearchNashBrute.jl")
export SearchNashBrute
include("../devel/BruteNashBasedOptimizer.jl")
export BruteNashBasedOptimizer, SolveNashBrute
include("../devel/ReducedActionSet.jl")
export BuildRunwayActionSet, ExclusiveJointSet, IsCCPNEProfile, CheckExclusiveSetCCPNE,
       CalcIndividualJReduced, CalcJReduced, EvalFairnessReduced, EvalGiniReduced,
       CalcHReducedSet, ReducedPacker, SearchCorrReduced
include("../devel/CorrLP.jl")
export SearchCorrLP, SearchCorrReducedLP, BruteNashBasedOptimizerLP
include("../devel/VertiportGame.jl")
export VertiportCost, OccupancyVec, NvVector, IsCCPNEProfileVertiport, FindAllCCPNEVertiport,
       CheckExclusiveSetCCPNEVertiport, SearchCorrLPVertiport, SearchCorrReducedLPVertiport,
       BruteNashBasedOptimizerLPVertiport, HullLPOverCandidateSet,
       HullLPDirect, CertificateBound
include("../devel/VertiportGameHet.jl")
export HetYieldWeights, VertiportCostHet, IsCCPNEProfileHet, FindAllCCPNEHet,
       CheckSetCCPNEHet, SearchCorrLPHet, SearchCorrReducedLPHet,
       HullLPHetOverX, HullLPHetOverF
include("../VQSim/actions.jl")
include("../VQSim/costs.jl")
include("../VQSim/schedule.jl")
include("../VQSim/state.jl")
export SimParams, init_state
include("../devel/MC_ccce.jl")
export run_mc_ccce_experiment, sample_fixed_sigma, CEViolationWithNoise, save_mc_rows_mat, SearchNashBruteWrapper
include("../devel/MC_ccce_alpha.jl")
export run_mc_alpha_experiment, save_mc_alpha_rows_mat

include("../VQSim/VQ_SearchCorrTensor.jl")
export build_epoch_game_tensors, CalcH_Tensor, SearchCorrTensor, sample_k, realized_choice_conditional_BR
include("../VQSim/VQ_BruteRRCETensor.jl")
export find_pne_set, solve_rrce_over_pne, pne_to_distribution, sanity_check_mapping, summarize_regrets, max_regret_per_k
include("../VQSim/VQ_Fcfs.jl")
export select_fcfs
include("init.jl")

end # module correlated
