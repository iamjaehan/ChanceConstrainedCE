module VQActions

export enumerate_actions_subset, enumerate_joint_actions

"""
Given eligible flight indices for one airline, return a list of actions.
Each action is a Vector{Int} of pushed flight indices (subset).
"""
function enumerate_actions_subset(flight_idxs::Vector{Int}; max_subset_size::Int=10)
    if length(flight_idxs) > max_subset_size
        error("Too many eligible flights for subset actions. Use k-release later.")
    end
    actions = Vector{Vector{Int}}()
    n = length(flight_idxs)
    # enumerate all subsets
    for mask in 0:(2^n - 1)
        a = Int[]
        for j in 1:n
            if (mask >> (j-1)) & 1 == 1
                push!(a, flight_idxs[j])
            end
        end
        push!(actions, a)
    end
    return actions
end

"""
actions_by_player: Vector of action-lists, each action-list is Vector{Vector{Int}}
Return joint_actions as Vector{Vector{Int}} where each is concatenated pushed flights.
Also return joint_action_tuples as Vector{Vector{Int}} indices of per-player action choice.
"""
function enumerate_joint_actions(actions_by_player::Vector{Vector{Vector{Int}}})
    n = length(actions_by_player) # n = How many airlines involved?
    # recursive cartesian product over action indices
    joint_pushed = Vector{Vector{Int}}()
    joint_choice = Vector{Vector{Int}}()

    function rec_build(i::Int, choice::Vector{Int}, pushed::Vector{Int})
        if i > n
            push!(joint_choice, copy(choice))
            push!(joint_pushed, copy(pushed))
            return
        end
        for (ai, act) in enumerate(actions_by_player[i])
            push!(choice, ai)
            append!(pushed, act)
            rec_build(i+1, choice, pushed)
            # rollback
            resize!(pushed, length(pushed) - length(act))
            pop!(choice)
        end
    end

    rec_build(1, Int[], Int[])
    return joint_pushed, joint_choice
end

end # module
