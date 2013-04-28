/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rt_study_h_
#define _rt_study_h_

#include "plmutil_config.h"
#include <vector>
#include "itk_image_type.h"
#include "plm_image.h"
#include "plm_image_type.h"
#include "segmentation.h"

class Metadata;
class Plm_image;
class Rt_study_private;
class Slice_index;
class Volume;
class Xio_ct_transform;

/*! \brief 
 * The Rt_study class encapsulates the concept of a radiotherapy planning 
 * data set, including image, structure set, and dose.
 */
class PLMUTIL_API Rt_study {
public:
    SMART_POINTER_SUPPORT (Rt_study);
public:
    Rt_study_private *d_ptr;

public:
    Rt_study ();
    ~Rt_study ();

    void load_dicom_dir (const char *dicom_dir);
    void load_dicom (const char *dicom_dir); 
    void load_dicom_dose (const char *dicom_path);
    void load_dicom_rtss (const char *dicom_path);
    void load_image (const char *fn);
    void load_image (const std::string& fn);
    void load_xio (const char *xio_dir, Slice_index *rdd);
    void load_ss_img (const char *ss_img, const char *ss_list);
    void load_dose_img (const char *dose_img);
    void load_dose_xio (const char *dose_xio);
    void load_dose_astroid (const char *dose_astroid);
    void load_dose_mc (const char *dose_mc);
    void load_rdd (const char *image_directory);
    void load_dcmtk (const char *dicom_dir); 
    void load_gdcm (const char *dicom_dir); 

    void load_cxt (const char *input_fn, Slice_index *rdd);
    void load_prefix (const char *input_fn);

    void save_dicom (const char *output_dir);
    void save_dicom_dose (const char *output_dir);

    void save_dose (const char* fname);
    void save_dose (const char* fname, Plm_image_type image_type);

    void set_user_metadata (std::vector<std::string>& metadata);

    bool have_image ();
    void set_image (Plm_image* pli);
    void set_image (Plm_image::Pointer pli);
    Plm_image::Pointer get_image ();

    bool have_dose ();
    void set_dose (Plm_image *pli);
    void set_dose (FloatImageType::Pointer itk_dose);
    void set_dose (Volume *vol);

    bool have_rtss ();
    Segmentation::Pointer get_rtss ();
    void set_rtss (Segmentation::Pointer rtss);

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