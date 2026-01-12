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
export SetC
include("../devel/SearchNash.jl")
export SearchNash, PrepNash, SolveNash
include("../devel/NashBasedOptimizer.jl")
export NashBasedOptimizer
include("../devel/CorrBasedOptimizer.jl")
export SearchCorr
include("../devel/SearchNashBrute.jl")
include("../devel/BruteNashBasedOptimizer.jl")
export BruteNashBasedOptimizer

end # module correlated
