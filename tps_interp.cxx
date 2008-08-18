/*===========================================================
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
===========================================================*/

#include "tps_interp.h"
#include "itk_warp.h"

template<class T>
void do_tps(TPS_parms* parms, typename itk::Image<T,3>::Pointer img_in, T default_val)
{
    typedef typename itk::Image<T,3> ImgType;
    typedef double CoordinateRepType;
    typedef typename itk::ThinPlateSplineKernelTransform<CoordinateRepType,3> TransformType;
    typedef typename itk::Point<CoordinateRepType,3 > PointType;
    typedef typename std::vector<PointType > PointArrayType;
    typedef typename TransformType::PointSetType PointSetType;
    typedef typename PointSetType::Pointer PointSetPointer;
    typedef typename PointSetType::PointIdentifier PointIdType;


    PointType p1;
    PointType p2;
	int dim[3];
    float offset[3];
    float spacing[3];
    char line[BUFLEN];
    Xform xform_tmp, xform;
    FILE* reference;
    FILE* target;

    get_image_header(dim, offset, spacing, img_in);

    typename PointSetType::Pointer sourceLandMarks = PointSetType::New();
    typename PointSetType::Pointer targetLandMarks = PointSetType::New();

    typename PointSetType::PointsContainer::Pointer sourceLandMarkContainer = sourceLandMarks->GetPoints();
    typename PointSetType::PointsContainer::Pointer targetLandMarkContainer = targetLandMarks->GetPoints();

    PointIdType id = itk::NumericTraits< PointIdType >::Zero;
    PointIdType id2 = itk::NumericTraits< PointIdType >::Zero;

    reference=fopen(parms->reference,"r");
    target=fopen(parms->target,"r");

    if(!reference || !target){
	fprintf(stderr, "An error occurred while opening the landmark files!");
	exit(-1);
    }

    while(fgets(line, BUFLEN,reference)){
		if(sscanf(line,"%lf %lf %lf",&p1[0],&p1[1],&p1[2])==3){
			sourceLandMarkContainer->InsertElement( id++, p1 );
			printf("reference Landmark: %f %f %f\n",p1[0],p1[1],p1[2]);
		}else{
			printf("Error! can't read the reference landmarks file");
			exit(-1);
		}
    }
    id = itk::NumericTraits< PointIdType >::Zero;
    while(fgets(line, BUFLEN,target)){
		if(sscanf(line,"%lf %lf %lf",&p2[0],&p2[1],&p2[2])==3){
			targetLandMarkContainer->InsertElement( id2++, p2 );
			printf("target Landmark: %f %f %f \n",p2[0],p2[1],p2[2]);
		}else{
			printf("Error! can't read the target landmarks file");
			exit(-1);
		}
    }

    fclose(reference);
    fclose(target);

	
    typename TransformType::Pointer tps = TransformType::New();
    tps->SetSourceLandmarks(sourceLandMarks);
    tps->SetTargetLandmarks(targetLandMarks);
    tps->ComputeWMatrix();

    xform.set_itk_tps(tps);
    xform_to_itk_vf (&xform_tmp,&xform, dim, offset, spacing);

    typename DeformationFieldType::Pointer vf = DeformationFieldType::New();
    vf = xform_tmp.get_itk_vf();

    printf ("Warping...\n");
	
    typename ImgType::Pointer im_warped = itk_warp_image (img_in, vf, 1, default_val);

    printf ("Saving...\n");
    save_image (im_warped, parms->warped);
    save_image(vf, parms->vf);
}

/* Explicit instantiations */
/* RMK: Visual studio 2005 without service pack requires <float> specifier
   on the explicit extantiations.  The current hypothesis is that this 
   is because the template is nested. */
template void do_tps<unsigned char>(TPS_parms* parms,itk::Image<unsigned char,3>::Pointer img_in, unsigned char);
template void do_tps<float>(TPS_parms* parms,itk::Image<float,3>::Pointer img_in, float);
template void do_tps<short>(TPS_parms* parms,itk::Image<short,3>::Pointer img_in, short);
