/* -----------------------------------------------------------------------
 *    See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
 *       ----------------------------------------------------------------------- */
#include "plmsegment_config.h"

#include "itkImage.h"
#include <itkSTAPLEImageFilter.h>

#include "mabs_staple.h"

Mabs_staple::Mabs_staple() {

    this->foreground_val = 1;

}


Mabs_staple::~Mabs_staple() {

    this->structures.clear();

}


void
Mabs_staple::add_input_structure(Plm_image::Pointer& structure) {

    this->structures.push_back(structure);

}


void
Mabs_staple::run() {

    typedef unsigned char PixelComponentType;
    const unsigned int Dimension = 3;
    typedef itk::Image< PixelComponentType, Dimension > ImageType;
    typedef itk::STAPLEImageFilter< ImageType, ImageType > StapleType;
    StapleType::Pointer staple = StapleType::New();

    std::list<Plm_image::Pointer>::iterator stru_it;
    int i;
    for (stru_it = this->structures.begin(), i=0;
         stru_it != this->structures.end(); stru_it++, i++) {

        staple->SetInput(i, (*stru_it)->itk_uchar());
    }

    staple->SetForegroundValue(this->foreground_val);
    staple->Update();

    this->output_img = Plm_image::New(staple->GetOutput());

}

