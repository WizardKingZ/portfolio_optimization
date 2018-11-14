import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

class StyleAnalysis:
	def __init__(self, fund_rts, factors, factor_names):
		self.__fund_rts = fund_rts
		self.__factors = factors
		self.__factor_names = factor_names
		self.__num_of_factors = len(factor_names)
		## configuration for the QP solver
		solvers.options['show_progress'] = False
		solvers.options['feastol'] = 1e-12
		solvers.options['abstol'] = 1e-12

	def analyze(self):
		Q = matrix(2*np.dot(self.__factors.T, self.__factors))
		p = matrix(-2*np.dot(self.__factors.T, self.__fund_rts))

		G = matrix(-np.identity(self.__num_of_factors))
		h = matrix(np.zeros(self.__num_of_factors), (self.__num_of_factors, 1))
		A = matrix(np.ones(self.__num_of_factors), (1, self.__num_of_factors))
		b = matrix(1.)

		sol = np.array(solvers.qp(Q, p, G=G, h=h, A=A, b=b, kktsolver='ldl')['x']).flatten()
		return pd.Series(data=sol, index=self.__factor_names)

