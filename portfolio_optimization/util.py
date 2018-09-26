import numpy as np

def portfolio_performance(assetReturn, assetCov, portfolioWeights, riskFreeRate=None):
	portfolioReturn = np.dot(portfolioWeights, assetReturn)
	portfolioVolatility = np.sqrt(np.dot(np.dot(portfolioWeights, assetCov), portfolioWeights))
	if riskFreeRate is None:
		return portfolioReturn, portfolioVolatility
	else:
		portfolioReturn += (1-sum(portfolioWeights))*riskFreeRate
		return portfolioReturn, portfolioVolatility