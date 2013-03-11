/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtds_h_
#define _rtds_h_

#include "plmutil_config.h"
#include <vector>
#include "itk_image_type.h"
#include "plm_image_type.h"

class Metadata;
class Plm_image;
class Rtds_private;
class Rtss;
class Slice_index;
class Volume;
class Xio_ct_transform;

/*! \brief 
 * The Rtds class encapsulates the concept of a radiotherapy planning 
 * data set, including image, structure set, and dose.
 */
class PLMUTIL_API Rtds {
public:
    Rtds_private *d_ptr;
public:
    Plm_image *m_img;                  /* CT image */
    Rtss *m_rtss;                      /* RT structure set */

public:
    Rtds ();
    ~Rtds ();

    void load_dicom_dir (const char *dicom_dir);
    void load_dicom (const char *dicom_dir); 
    void load_dicom_dose (const char *dicom_path);
    void load_dicom_rtss (const char *dicom_path);
    void load_xio (const char *xio_dir, Slice_index *rdd);
    void load_ss_img (const char *ss_img, const char *ss_list);
    void load_dose_img (const char *dose_img);
    void load_dose_xio (const char *dose_xio);
    void load_dose_astroid (const char *dose_astroid);
    void load_dose_mc (const char *dose_mc);
    void load_rdd (const char *image_directory);
    void load_dcmtk (const char *dicom_dir); 
    void load_gdcm (const char *dicom_dir); 

    void save_dicom (const char *output_dir);
    void save_dicom_dose (const char *output_dir);

    void save_dose (const char* fname);
    void save_dose (const char* fname, Plm_image_type image_type);

    void set_user_metadata (std::vector<std::string>& metadata);

    void set_dose (Plm_image *pli);
    void set_dose (FloatImageType::Pointer itk_dose);
    void set_dose (Volume *vol);

    const std::string& get_xio_dose_filename () const;
    Xio_ct_transform* get_xio_ct_transform ();

    Metadata* get_metadata ();

    Slice_index* get_slice_index ();

    Volume* get_image_volume_short ();
    Volume* get_image_volume_float ();

    bool has_dose ();
    Plm_image* get_dose_plm_image ();
    Volume* get_dose_volume_float ();

protected:
    void save_dcmtk (const char *dicom_dir);
    void save_dcmtk_dose (const char *dicom_dir);
    void save_gdcm (const char *dicom_dir);
    void convert_ss_img_to_cxt ();
};

#endif
