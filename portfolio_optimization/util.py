import numpy as np

def portfolio_performance(assetReturn, assetCov, portfolioWeights, riskFreeRate=None):
	portfolioReturn = np.dot(portfolioWeights, assetReturn)
	portfolioVolatility = np.sqrt(np.dot(np.dot(portfolioWeights, assetCov), portfolioWeights))
	if riskFreeRate is None:
		return portfolioReturn, portfolioVolatility
	else:
		portfolioReturn += (1-sum(portfolioWeights))*riskFreeRate
		return portfolioReturn, portfolioVolatility

def create_constraints(assets, constraints):
	N = len(assets)
	h = np.hstack(([-constraints[asset][0] for asset in assets if asset in constraints.keys()],
				  [constraints[asset][1] for asset in assets if asset in constraints.keys()]))
	G = np.identity(N)[[assets.index(asset) for asset in assets if asset in constraints.keys()],:]
	G = np.vstack([G, G])
	return G, h