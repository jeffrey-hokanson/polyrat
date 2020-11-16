import numpy as np

from polyrat import *


def test_skfit_stabilized():
	X = np.random.randn(50,1) #+ 1j*np.random.randn(100,1)
	#y = np.random.randn(100,) + 1j*np.random.randn(100,)
	y = np.abs(X).flatten()

	skfit_stabilized(X, y, 4, 4, norm = np.inf)
