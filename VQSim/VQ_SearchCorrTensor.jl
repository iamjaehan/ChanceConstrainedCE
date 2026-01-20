"""
Solve (chance-constrained) correlated equilibrium with general joint-action costs.

Inputs
  J_air[i][k]   : nominal airline i cost at joint action k
  J_coord[k]    : coordinator cost at joint action k (objective)
  action_sizes  : |X_i| for active players i=1..n
  zalpha        : quantile factor (given)
  sigma_vec[i]  : noise std for airline i
  rho           : L2 regularizer on z (>=0)

Returns
  z             : distribution over joint actions (length L)
  info          : diagnostics
"""
function SearchCorrTensor(J_air, J_coord, action_sizes; zalpha, sigma_vec, rho=0.0)
    # build constraints:
    #   (i) sum(z)=1
    #   (ii) z_k >= 0
    #   (iii) CC-CE constraints for all i, ai, ai'
    # objective:
    #   dot(z, J_coord) + rho*dot(z,z)
end
