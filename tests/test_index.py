import numpy as np
import pytest
from polyrat import total_degree_index, max_degree_index
from itertools import product


def total_degree_index_slow(dim, degree):
	indices = max_degree_index([degree for i in range(dim)])
	I = np.argwhere(np.sum(indices, axis = 1) <= degree).flatten()
	return indices[I]


@pytest.mark.parametrize("dim", [1,3,5])
@pytest.mark.parametrize("degree", [0,1,2,5])
def test_total_degree_index(dim, degree):
	true_indices = total_degree_index_slow(dim, degree)
	
	indices = total_degree_index(dim, degree)
	assert len(indices) == len(true_indices), "Should be the same length"

	for k in range(len(indices)):
		res = (true_indices == indices[k]).all(axis = 1).nonzero()
		assert len(res[0]) == 1, "Should only be one match"


@pytest.mark.parameterize("degree", [[0],[5],[1,4],[10,3],[4,5,6,]])
def test_max_degree_index(degree):
	indices = max_degree_index(degree)

	for idx in product(*[range(d+1) for d in degree]):
		res = (indices == idx).all(axis = 1).nonzero()
		assert len(res[0]) == 1, "Should only be one match"
		 


if __name__ == '__main__':
#	test_total_degree_index(dim=3, degree=5)
	test_max_degree_index([1,2,3])
