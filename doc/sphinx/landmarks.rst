Landmark-based image registration
=================================

The landmark_warp executable performs registration by matching fiducials on reference and test images. The general format of the command line is::

  landmark_warp [options]

The list of possible options can be seen by typing:: 

  landmark_warp --help

The command line usage is given as follows::

  Usage: landmark_warp [options]
  Required:
      -f=filename	Fixed fiducials
      -m=filename	Moving fidicials
      -I=filename	Input image to warp
      -O=filename	Output warped image
  Optional:
      -x=filename	Input landmark xform
      -V=filename	Output vector field
      -F=filename	Fixed image (match output size to this image)
      -a=ENUM		Algorithm type={tps,gauss,wendland}
      -r=FLOAT		Radius of radial basis function
      -Y=FLOAT		Young modulus
      -d=FLOAT		Value to set for pixels with unknown value

Options "-a", "-r", "-Y", "-d" are set by default to::

      -a=gauss		Gaussian RBFs with infinite support
      -r=50.0		Gaussian width 50 cm
      -Y=0.0		No regularization of vector field
      -d=-1000		Air

You may want to choose different algorithm::

      -a=tps		Thin-plate splines (for global registration)
      -a=wendland	Wendland RBFs with compact support (for local registartion)

In the case of Wendland RBFs "-r" option sets the radius of support.

Regularization of vector field is available for "gauss"  and "wendland" algorithms. To regularize the output vector field increase "-Y" to '0.1' and up with increment '0.1'.
	




