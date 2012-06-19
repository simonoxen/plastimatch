/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _mabs_vote_h_
#define _mabs_vote_h_

#include "sys/plm_path.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCastImageFilter.h"
#include "itk_image_type.h"

const double PI = 3.141592653589793238;
const unsigned int Dimension = 3;

typedef double      PixelType;
typedef itk::Image< PixelType, Dimension > ImageType;
typedef itk::ImageFileReader< ImageType > ReaderType;

typedef unsigned short      OutputPixelType;
typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
typedef itk::CastImageFilter< ImageType, OutputImageType > CastFilterType;
typedef itk::ImageFileWriter< OutputImageType >  WriterType;

class Mabs_subject_manager;

class PLMSEGMENT_API Mabs_vote {
public:
    Mabs_vote ();
    ~Mabs_vote ();

    void vote_contribution (Plm_image& target_image_plm,
                            Plm_image& atlas_image_plm,
                            Plm_image& atlas_structure_plm,
                            ImageType::Pointer like0,
                            ImageType::Pointer like1);
    bool vote (const Mabs_parms& parms);
    bool vote_old (const Mabs_parms& parms);

private:
    void hello_world ();
    int write_to_file (    const ImageType::Pointer image_data,
                           const std::string out_file
);
    
public:
    char target_fn[_MAX_PATH];
    char output_fn[_MAX_PATH];
    Mabs_subject_manager* sman;
};

#endif /* #ifndef _mabs_vote_h_ */
