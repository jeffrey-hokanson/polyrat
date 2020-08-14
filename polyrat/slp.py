""" Sequential linear programing approach for minimax optimization

"""

import numpy as np
from iterprinter import IterationPrinter


def slq(obj, x0, jac, constraints):
	r"""

	Solves 
	
	min_x  \| res(x) \|_\infty

	where

		|res(x)[j]| \approx | res(x)[j] + jac(x)[j] | 

	"""

	if verbose:
		iterprinter = IterationPrinter(it = '4d')
		iterprinter.print_header(it = 'iter')

	

