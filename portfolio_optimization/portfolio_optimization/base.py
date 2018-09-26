from abc import ABCMeta, abstractmethod

class AbstractPortfolioAllocation(object):
	"""
	"""

	__metaclass__ = ABCMeta

	@abstractmethod
	def calculate_weights(self):
		"""
		Provides the mechanisms to calculate the list of signals, suggested positional size, the strength of the signal
		"""
		raise NotImplementedError("Should implement calculate_weights()")