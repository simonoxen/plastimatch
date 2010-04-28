Mutual information
------------------

Plastimatch uses a histogram-based mutual information method.
Let :math:`\mu` be the transformation parameters, such as B-spline 
coefficients.  The joint histogram :math:`p` keeps track of the 
joint probability, and is indexed by :math:`i` and :math:`k`.  
Marginal probabilities are computed for both the fixed (reference) image 
(stored in :math:`p_F`), and the moving (test) image 
(stored in :math:`p_M`).
The mutual information equation is given as:

.. math::

   s = \sum_{i,k} p(i,k|\mu) \ln{ \frac{p(i,k|\mu)}{p_M(i|\mu) p_F(k)} }.

To clarify the previous equation, :math:`p(i,k|\mu)` is a two-dimensional 
histogram, and :math:`p_M(i|\mu)` and :math:`p_F(k)` are the marginal 
histograms.  
The marginal histogram :math:`p_F(k)` is traditionally written 
considered independent of :math:`\mu`, but in fact gets recomputed 
each iteration as :math:`\mu` is optimized.  This is because the 
degree of overlap between the two images changes for each iteration.  
Pixels in the fixed image 
which fall outside of the moving image are not included in the calculation 
of :math:`p_F(k)`.

Updates to the joint histogram are performed by exact binning of the 
fixed intensity value and fractional linear interpolation of the 
moving intensity value.  This concept is best illustrated by example.  
Suppose we have a matching voxel with intensity 5 in the fixed image and 
intensity 7 in the moving image, and the histogram bins have width 4.  
The fixed image intensity 5 is binned to the range (4,8), and the joint 
histogram is updated in this bin.  The moving image intensity 7, however,
is matched to bins (4,8) and (8,12).  Linear 
interpolation is used to proportionally update both of the moving bins.

.. image:: ../figures/mutual_information_1.png
   :width: 55 %

Correspondences are formed using the partial value interpolation method.  
Consider a location 
:math:`f` in the fixed image, with intensity :math:`f_v`.  
The matching location :math:`m` in the moving image is not aligned 
with the moving image grid, but rather lies between voxels.  
In the 2D example below, location :math:`m` lies between \
pixels with intensities a, b, c, and d.

.. image:: ../figures/mutual_information_2.png
   :width: 28 %

Instead of performing linear interpolation of the intensities 
and forming a single correspondence, four partial correspondences are made
with the four closest voxels.  Fractional updates are made to 
the joint histogram for each of these correspondences according 
to the table below:

+---------------------------------------------+-------------------------------+
|Intensity pair (fixed value, moving value)   |Update fraction                |
|                                             |                               |
+=============================================+===============================+
|(f_v,a)                                      |:math:`(1-d_x)(1-d_y)`         |
+---------------------------------------------+-------------------------------+
|(f_v,b)                                      |:math:`d_x(1-d_y)`             |
+---------------------------------------------+-------------------------------+
|(f_v,c)                                      |:math:`(1-d_x)d_y`             |
+---------------------------------------------+-------------------------------+
|(f_v,d)                                      |:math:`d_x d_y`                |
+---------------------------------------------+-------------------------------+

Recalling that two joint histogram entries are updated for each 
intensity pair, we see that up to 8 entries will be updated in our 2D example.
