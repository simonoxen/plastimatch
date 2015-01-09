.. _filters:

Filters
=======

The plastimatch filter command is a convenient method for performing 
common filters on images.  This documentation is intended to explain 
in more detail the mathematics and implementation of these filters.

Gabor
-----
The Gabor filter is a complex filter, and has real and imaginary 
components.  Plastimatch implements only a 
uniform and symmetric form of the Gabor kernel, 
which is described as:

.. math::
   S \cdot \mathrm{exp} \left( -\frac{|{\bf x}|^2}{\sigma^2} \right) 
   \cdot \mathrm{exp} \left( j \, \omega {\bf k}^T {\bf x} \right)

Where **x** is the voxel position (relative to the center of the kernel), 
*:sigma:* is the filter width, and *S* is an arbitrary scaling factor.
The vector **k** is the direction of modulation, and the frequency of 
modulation is given by *:omega:*.  Note that 
the vector **k** is internally normalized to be a unit vector 
by plastimatch, so that larger **k** vectors do not increase 
the modulation.




Gauss
-----
The cannonical form for a Gaussian kernel is given as

.. math::
   \frac{1}{\sqrt{(2\pi)^2|\Sigma|}} 
   \cdot \mathrm{exp} \left( -\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu) \right)

Note: because plastimatch uses a fixed width convolution kernel, the 
above expression may not sum to unity.  Therefore, an empirical 
factor S is used instead.

.. math::
   S \cdot \mathrm{exp} \left( -\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu) \right)



Kernel
------
To be written.

