from .base import AbstractPortfolioAllocation
import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from .util import portfolio_performance, create_constraints
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

class MarkowitzPortfolioAllocation(AbstractPortfolioAllocation):
	def __init__(self, expectedReturn, assetsCovariance, assetNames,
				 constraints=None, riskFreeAsset=None, 
				 riskAversion=None, targetReturn=None,
				 targetVol=None):
		self.__expectedReturn = expectedReturn
		self.__assetsCovariance = assetsCovariance
		if constraints is not None:
			self.__constraints = {}
			self.__constraints['G'], self.__constraints['h'] = create_constraints(assetNames, constraints)
		else:
			self.__constraints = None
		self.__riskFreeAsset = riskFreeAsset
		self.__riskAversion = riskAversion
		self.__targetReturn = targetReturn
		self.__targetVol = targetVol
		self.__numOfAssets = len(expectedReturn)
		self.__assetNames = assetNames

		## 
		self.__tangency_portfolio = None
		self.__min_variance_portfolio = None
		self.__efficient_frontier = None

		## configuration for the QP solver
		solvers.options['show_progress'] = False
		solvers.options['feastol'] = 1e-12
		solvers.options['abstol'] = 1e-12

		## useful constants
		self.__invAssetCovariance = np.linalg.inv(assetsCovariance)
		self.__A = np.dot(np.dot(self.__expectedReturn, self.__invAssetCovariance), self.__expectedReturn)
		self.__B = np.dot(np.dot(self.__expectedReturn, self.__invAssetCovariance), np.ones(self.__numOfAssets))
		self.__C = np.dot(np.dot(np.ones(self.__numOfAssets), self.__invAssetCovariance), np.ones(self.__numOfAssets))

	def __portfolio_weights_helper(self, weights):
		return pd.DataFrame(weights, index=self.__assetNames, columns=['weights'])

	def get_min_variance_portfolio(self):
		if self.__min_variance_portfolio is None:
			self.__min_variance_portfolio = self.__portfolio_weights_helper(1./self.__C*np.dot(self.__invAssetCovariance, np.ones(self.__numOfAssets)))
		return self.__min_variance_portfolio

	def get_tangency_portfolio(self):
		if self.__tangency_portfolio is None:
			self.__tangency_portfolio = self.__portfolio_weights_helper(1./(self.__B - self.__riskFreeAsset*self.__C)*np.dot(self.__invAssetCovariance, self.__expectedReturn-self.__riskFreeAsset))
		return self.__tangency_portfolio

	def __unconstrainted_weights(self):
		if self.__riskAversion is not None:
			if self.__riskFreeAsset is not None:
				return 1./self.__riskAversion*np.dot(self.__invAssetCovariance, self.__expectedReturn-self.__riskFreeAsset)
			else:
				return 1./self.__riskAversion*np.dot(self.__invAssetCovariance, self.__expectedReturn-self.__B/self.__C*np.ones(self.__numOfAssets))
		else:
			Q = matrix(self.__assetsCovariance)
			p = matrix(np.zeros((self.__numOfAssets, 1)))
			if self.__riskFreeAsset is not None:
				G = matrix(-self.__expectedReturn-self.__riskFreeAsset, (1, self.__numOfAssets))
				h = matrix(-self.__targetReturn+self.__riskFreeAsset)
				A = None
			else:
				G = matrix(-self.__expectedReturn, (1, self.__numOfAssets))
				h = matrix(-self.__targetReturn)
				A = matrix(np.ones(self.__numOfAssets), (1, self.__numOfAssets))
				b = matrix(1.)
			if A is None:
				sol = solvers.qp(Q, p, G=G, h=h, kktsolver='ldl') #, solver='mosek', kktsolver='ldl')
			else:
				sol = solvers.qp(Q, p, G=G, h=h, A=A, b=b, kktsolver='ldl')
			if sol['status'] != 'optimal':
				print("Convergence problem")
			return np.array(sol['x']).reshape(self.__numOfAssets)

	def __constrainted_weights(self):
		G = matrix(self.__constraints['G'])
		h = matrix(self.__constraints['h'])
		if self.__riskAversion is not None:
			Q = self.__riskAversion*matrix(self.__assetsCovariance)
			if self.__riskFreeAsset is not None:
				p = -matrix(self.__expectedReturn-self.__riskFreeAsset)
				A = None
			else:
				p = -matrix(self.__expectedReturn)
				A = matrix(np.ones(self.__numOfAssets), (1, self.__numOfAssets))
				b = matrix(1.)
		else:
			Q = matrix(self.__assetsCovariance)
			p = matrix(np.zeros((self.__numOfAssets, 1)))
			if self.__riskFreeAsset is not None:
				G = matrix(np.append(self.__constraints['G'], -(self.__expectedReturn-self.__riskFreeAsset).reshape(1, self.__numOfAssets)).reshape(self.__constraints['G'].shape[0]+1, self.__constraints['G'].shape[1]))
				h = matrix(np.append(self.__constraints['h'], -self.__targetReturn+self.__riskFreeAsset))
				A = None
			else:
				G = matrix(np.append(self.__constraints['G'], -(self.__expectedReturn).reshape(1, self.__numOfAssets)).reshape(self.__constraints['G'].shape[0]+1, self.__constraints['G'].shape[1]))
				h = matrix(np.append(self.__constraints['h'], -self.__targetReturn))				
				A = matrix((np.ones(self.__numOfAssets)).reshape(1, self.__numOfAssets))
				b = matrix(1.)
		if A is None:
			sol = solvers.qp(Q, p, G=G, h=h, kktsolver='ldl') #, solver='mosek', kktsolver='ldl')
		else:
			sol = solvers.qp(Q, p, G=G, h=h, A=A, b=b, kktsolver='ldl') #, solver='mosek', kktsolver='ldl')
		if sol['status'] != 'optimal':
			print("Convergence problem")
		return np.array(sol['x']).reshape(self.__numOfAssets)

	def calculate_weights(self):
		if self.__constraints is None:
			return self.__portfolio_weights_helper(self.__unconstrainted_weights())
		else:
			return self.__portfolio_weights_helper(self.__constrainted_weights())

	def get_portfolio_performance(self, weights):
		return portfolio_performance(self.__expectedReturn, self.__assetsCovariance, weights, riskFreeRate=self.__riskFreeAsset)

class MarkowitzPortfolio:
	def __init__(self, assetReturns, assetCovariance, assetNames, riskFreeRate=None):
		self.assetReturns = assetReturns
		self.assetCovariance = assetCovariance
		self.assetNames = list(assetNames)
		self.riskFreeRate = riskFreeRate
		self.numOfAssets = len(assetNames)
		## set up a basic markowitz portfolio
		self.portfolio = MarkowitzPortfolioAllocation(self.assetReturns, self.assetCovariance, self.assetNames, riskFreeAsset=self.riskFreeRate)


	def get_portfolio_performance(self):
		return self.portfolio.get_portfolio_performance(self.current_allocation['weights'])

	def get_min_variance_portfolio(self):
		self.current_allocation = self.portfolio.get_min_variance_portfolio()
		return self.current_allocation

	def get_tangency_portfolio(self):
		self.current_allocation = self.portfolio.get_tangency_portfolio()
		return self.current_allocation

	def get_allocations(self, configuration):
		"""
		configuration should be a dictionary of 
		{
			'constraints':,
			'riskAversion':,
			'targetReturn':,
			'targetVol':
 		}
		"""
		self.portfolio = MarkowitzPortfolioAllocation(self.assetReturns, self.assetCovariance, self.assetNames, riskFreeAsset=self.riskFreeRate, **configuration)
		self.current_allocation = self.portfolio.calculate_weights()
		return self.current_allocation

	def display_efficient_frontier(self, assetsAnnotation=False, specialPortfolioAnnotation=False, addTangencyLine=False, figsize=(10, 7), upper_bound=0.34, step=50, path=None):
		## naive portfolio
		portfolio = MarkowitzPortfolioAllocation(self.assetReturns, self.assetCovariance, self.assetNames, riskFreeAsset=self.riskFreeRate)
		if self.riskFreeRate is not None:
			tangency_portfolio = portfolio.get_tangency_portfolio()
			rp, sdp = portfolio.get_portfolio_performance(tangency_portfolio['weights'])

		min_variance_portfolio = portfolio.get_min_variance_portfolio()
		rp_min, sdp_min = portfolio.get_portfolio_performance(min_variance_portfolio['weights'])
		
		fig, ax = plt.subplots(figsize=figsize)

		if assetsAnnotation:
			for i, txt in enumerate(self.assetNames):
				ax.annotate(txt, (np.sqrt(self.assetCovariance[i, i]),self.assetReturns[i]), xytext=(10,0), textcoords='offset points')
		if specialPortfolioAnnotation:
			if self.riskFreeRate is not None:
				ax.scatter(sdp,rp,marker='*',color='r',s=50, label='Maximum Sharpe ratio portfolio')
				ax.scatter(0, self.riskFreeRate, marker='*',color='b',s=50, label='Risk-free rate')
			ax.scatter(sdp_min,rp_min,marker='*',color='g',s=50, label='Minimum volatility portfolio')

		targets = np.linspace(rp_min, upper_bound, step)
		target_portfolio_vols = []
		for target in targets:
			target_portfolio = MarkowitzPortfolioAllocation(self.assetReturns, self.assetCovariance, self.assetNames, targetReturn=target)
			target_portfolio_weights = target_portfolio.calculate_weights()
			_, target_portfolio_vol = target_portfolio.get_portfolio_performance(target_portfolio_weights['weights'])
			target_portfolio_vols.append(target_portfolio_vol)
		ax.plot(target_portfolio_vols, targets, linestyle='-', color='black', label='efficient frontier', linewidth=.5)
		lower_targets = targets - rp_min*np.ones(step)
		lower_targets = rp_min*np.ones(step) - lower_targets
		ax.plot(target_portfolio_vols, lower_targets, linestyle='-.', color='black', label='lower frontier', linewidth=.5)
		ax.set_title('Portfolio Optimization')
		ax.set_xlabel('$\sigma(r_p)$')
		ax.set_ylabel('$E[r_p]$')
		ax.set_xlim(xmin = -0.01)
		if self.riskFreeRate is not None:
			ymin = min(self.riskFreeRate, min(self.assetReturns), rp_min)*0.8
		else:
			ymin = min(min(self.assetReturns), rp_min)*0.8
		if addTangencyLine:
			y = np.linspace(self.riskFreeRate, upper_bound, step)
			ax.plot(sdp/(rp-self.riskFreeRate)*(y-self.riskFreeRate), y, linestyle='-', color='blue', label='capital market line', linewidth=.5)
		ax.set_ylim(ymin = ymin)
		ax.legend(labelspacing=0.8)
		if path is not None:
			plt.savefig(path)

class BlackLittermanPortfolio(MarkowitzPortfolio):
	def __get_variance(self, mean, confidence_level, plusminus):
		val = norm.isf((1. - confidence_level*0.01)/2.)
		return (mean * plusminus * 0.01 / val)**2

	def __create_view(self, views):
		Q = []
		Omega = []
		P = []
		for view in views.keys(): 
			Q.append(views[view]['scale'])
			Omega.append(self.__get_variance(views[view]['scale'], views[view]['confidence'], views[view]['plusminus%']))
			p = np.zeros(self.numOfAssets)
			if views[view]['type'] == 'absolute':
				p[self.assetNames.index(view)] = 1
			else:
				for idx, v in enumerate(view.split('|')):
					p[self.assetNames.index(v)] = views[view]['weights'][idx]
			P.append(p)
		return np.array(Q), np.stack(P), np.diag(Omega)

	def update_views(self, views, tau):
		"""
		views:  a dictionary of news on each asset and its associate confidence
			for example: 
				{'MKT': {'type': 'absolute', 'scale': 0.1, 'confidence': 0.01},
				 'SML|MKT' : {'type': 'relative', 'scale': -0.1, 'confidence': 0.01, 'weights': [1, -1]}}
		tau: scalor representing the precision of the prior
		"""
		## create cache for the equilibrium view
		try:
			if self.equilibriumAssetReturns is None:
				self.equilibriumAssetReturns = self.assetReturns
		except:
			self.equilibriumAssetReturns = self.assetReturns

		invAssetCovariance = np.linalg.inv(self.assetCovariance)
		Q, P, Omega = self.__create_view(views)
		pop = np.dot(np.dot(P.T, np.linalg.inv(Omega)), P)
		v1 = np.linalg.inv(invAssetCovariance + tau*pop)
		v2 = np.dot(v1, tau*np.dot(P.T, np.linalg.inv(Omega)))
		v1 = np.dot(v1, invAssetCovariance)
		self.assetReturns = np.dot(v1, self.equilibriumAssetReturns) + np.dot(v2, Q)
		## reset portfolio
		self.portfolio = MarkowitzPortfolioAllocation(self.assetReturns, self.assetCovariance, self.assetNames, riskFreeAsset=self.riskFreeRate)
		
	def reset(self):
		"""
		remove views 
		"""
		try:
			self.assetReturns = self.equilibriumAssetReturns
			self.equilibriumAssetReturns = None
			## reset portfolio
			self.portfolio = MarkowitzPortfolioAllocation(self.assetReturns, self.assetCovariance, self.assetNames, riskFreeAsset=self.riskFreeRate)
		except:
			print("No view present, so do not need to reset")

