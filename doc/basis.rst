================
Polynomial Bases
================

The foundation of this library is a set of *polynomial basis* classes.
Each class defines a set of functions :math:`\lbrace \phi_j \rbrace_j`
that form a basis for the desired polynomial space;
i.e., a mononomial basis for polynomials of degree 3 of one variable 
has basis functions

.. math::
	\phi_0(x) = 1, \quad \phi_1(x) = x, \quad \phi_2(x) = x^2, \quad \phi_3(x) = x^3. 

Here we consider two types of polynomial bases:
*total degree* polynomial bases
and *maximum degree* polynomial bases:

.. math::
	\mathcal P_{m}^{\text{tot}}
		& := \text{Span} \left\lbrace f:
			f(\mathbf x) = \prod_{i=1}^d x_i^{\alpha_i}
			\right\rbrace_{|\boldsymbol \alpha| \le m}, \quad
		& \text{where} & \quad |\boldsymbol \alpha| = \sum_{i=1}^d \alpha_i; \\
	\label{eq:max}
	\mathcal P_{\mathbf m}^{\text{max}}
		& := 
			\text{Span} \left \lbrace f:
				f(\mathbf x) = \prod_{i=1}^d x_i^{\alpha_i}
			\right\rbrace_{ \boldsymbol \alpha \le \mathbf m},
		& \text{where} & \quad \boldsymbol \alpha \le \mathbf m \Leftrightarrow
			\alpha_i \le m_i.



.. note::
	Although these polynomial bases ideally represent the same quantities,
	depending on the input coordinates their numerical conditioning can be very different.
	The most robust approach, although expensive, is to use :class:`~polyrat.ArnoldiPolynomialBasis`.
	If performance is desired, choose a tensor product polynomial basis 
	for which the input coordinates (after scaling) approximately sample the measure under which the polynomials
	are orthogonal.
	As a rule of thumb:
	
	* if :math:`x_i` is uniformly distributed on :math:`[a,b]` use :class:`~polyrat.LegendrePolynomialBasis`;
	* if :math:`x_i` is normally distributed with mean :math:`\mu` and standard deviation :math:`\sigma` use :class:`~polyrat.HermitePolynomialBasis`;
	* if :math:`\mathbf x_i \in \mathbb{C}^d` for :math:`d\ge 2` and degree greater than 10, use :class:`~polyrat.ArnoldiPolynomialBasis`.
		


Most bases are initialized using a set of input coordinates :math:`\mathbf x_i`
and a degree.
We require the input coordinates to determine a scaling to make the 
basis well conditioned;
i.e., for the Legendre polynomial basis
we perform an affine transformation (which does not alter polynomial basis)
such that these input coordinates are in the :math:`[-1,1]^m` hypercube.



.. autoclass:: polyrat.PolynomialBasis
   :members:
   
   .. automethod:: __init__
   




Tensor Product Bases
====================

Several polynomial bases are constructed
from univariate polynomial bases defined in Numpy
:func:`~numpy:numpy.polynomial`.
These bases are each constructed by subclassing 
:class:`polyrat.TensorProdutPolynomialBasis`
which performs two tasks:

* multiplying univariate polynomials in each coordinate to construct the desired multivariate basis and
* using an affine transformation in each coordinate to improve conditioning.

Here we provide an overview of the internals of this class to show
how new bases could be implemented in the same way.

.. autoclass:: polyrat.TensorProductPolynomialBasis

   .. automethod:: _vander

   .. automethod:: _der
   
   .. automethod:: roots

   .. automethod:: _set_scale

   .. automethod:: _scale

   .. automethod:: _inv_scale
 

Monomial Basis
--------------

.. autoclass:: polyrat.MonomialPolynomialBasis

Legendre Basis
--------------

.. autoclass:: polyrat.LegendrePolynomialBasis

Chebyshev Basis
---------------

.. autoclass:: polyrat.ChebyshevPolynomialBasis

Hermite Basis
-------------

.. autoclass:: polyrat.HermitePolynomialBasis

Laguerre Basis
--------------

.. autoclass:: polyrat.LaguerrePolynomialBasis


Arnoldi Polynomial Basis
========================
.. autoclass:: polyrat.ArnoldiPolynomialBasis


For low-level access, the following functions are available. 

.. autofunction:: polyrat.vandermonde_arnoldi_CGS 

.. autofunction:: polyrat.vandermonde_arnoldi_eval

.. autofunction:: polyrat.vandermonde_arnoldi_eval_der


Lagrange Polynomial Basis
=========================

.. autoclass:: polyrat.LagrangePolynomialBasis

.. autofunction:: polyrat.lagrange_roots
