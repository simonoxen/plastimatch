/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"

#include "vf_jacobian.h"

Jacobian::Jacobian ()
{
    vf = 0;
    vfjacstats_fn = " ";
}
void 
Jacobian::set_output_vfstats_name (Pstring vfjacstats){
  this->vfjacstats_fn=vfjacstats;
}
void 
Jacobian::set_input_vf(DeformationFieldType::Pointer vf){
    this->vf = vf;
}

void 
Jacobian::write_output_statistics(Jacobian_stats *JacoStats)
{
  FILE *fid;
  
  fid=fopen(JacoStats->outputstats_fn.c_str(),"w");
  
  if (fid != NULL)
  {
    fprintf(fid,"Min Jacobian: %.6f\n",JacoStats->min);
    fprintf(fid,"Max Jacobian: %.6f\n",JacoStats->max);
    fclose(fid);
  }
}

FloatImageType::Pointer
Jacobian::make_jacobian ()
{
    DeformationFieldType::Pointer deffield; 
    deffield= this->vf;
    
    JacobianFilterType::Pointer jacobianFilter = JacobianFilterType::New();
    jacobianFilter->SetInput( deffield );
    jacobianFilter->SetUseImageSpacing( true );
    jacobianFilter->Update();
    
    typedef itk::MinimumMaximumImageCalculator<FloatImageType> MinMaxFilterType;
    MinMaxFilterType::Pointer minmaxfilter = MinMaxFilterType::New();
    
    FloatImageType::Pointer outimg =jacobianFilter->GetOutput();
    
    try
      {
	minmaxfilter->SetImage(jacobianFilter->GetOutput());
      }
    catch( itk::ExceptionObject& err )
      {
      std::cout << "Unexpected error." << std::endl;
      std::cout << err << std::endl;
      exit( EXIT_FAILURE );
      }
    minmaxfilter->Compute();
    
    std::cout<<"Minimum of the determinant of the Jacobian of the warp: " <<minmaxfilter->GetMinimum()<<std::endl;
    std::cout<<"Maximum of the determinant of the Jacobian of the warp: " <<minmaxfilter->GetMaximum()<<std::endl;
    
    Jacobian_stats JacoStats;
    JacoStats.min = minmaxfilter->GetMinimum();
    JacoStats.max = minmaxfilter->GetMaximum();
    JacoStats.outputstats_fn = this->vfjacstats_fn;
    if (this->vfjacstats_fn.not_empty())
      this->write_output_statistics(&JacoStats);
    
    return outimg;

}
