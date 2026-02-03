module correlated

using Symbolics: Symbolics, @variables
using ParametricMCPs: ParametricMCPs, ParametricMCP
using BlockArrays: BlockArray, Block, mortar, blocks
using LinearAlgebra: norm_sqr

include("parametric_game.jl")
export ParametricOptimizationProblem, solve, total_dim
include("parametric_optimization_problem.jl")
export ParametricGame
include("../devel/0_GameSetup.jl")
export SetC, generateAseq
include("../devel/SearchNash.jl")
export SearchNash, PrepNash, SolveNash
include("../devel/NashBasedOptimizer.jl")
export NashBasedOptimizer
include("../devel/CorrBasedOptimizer.jl")
export SearchCorr
include("../devel/SearchNashBrute.jl")
include("../devel/BruteNashBasedOptimizer.jl")
export BruteNashBasedOptimizer
include("../VQSim/actions.jl")
include("../VQSim/costs.jl")
include("../VQSim/schedule.jl")
include("../VQSim/state.jl")

include("../VQSim/VQ_SearchCorrTensor.jl")
export build_epoch_game_tensors, CalcH_Tensor, SearchCorrTensor, sample_k, realized_choice_conditional_BR
include("../VQSim/VQ_BruteRRCETensor.jl")
export find_pne_set, solve_rrce_over_pne, pne_to_distribution, sanity_check_mapping, summarize_regrets, max_regret_per_k
include("../VQSim/VQ_Fcfs.jl")
export select_fcfs

end # module correlated
