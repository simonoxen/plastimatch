Mutual information
------------------

Plastimatch uses a histogram-based mutual information method.
Let :math:`\mu` be the transformation parameters, such as B-spline 
coefficients.  The joint histogram :math:`p` keeps track of the 
joint probability, and is indexed by :math:`i` and :math:`k`.  
Marginal probabilities are computed for both the reference (fixed) image 
(stored in :math:`p_R`), and the test (moving) image 
(stored in :math:`p_T`).
The mutual information is given as:

.. math::

   s = \sum_{i,k} p(i,k|\mu) \ln{ \frac{p(i,k|\mu)}{p_T(i|\mu) p_R(k)} }
