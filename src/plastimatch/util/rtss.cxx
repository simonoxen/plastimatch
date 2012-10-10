/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"

#if GDCM_VERSION_1
#include "gdcm1_dose.h"
#endif
#include "cxt_extract.h"
#include "cxt_io.h"
#include "dir_list.h"
#include "file_util.h"
#include "itk_image_save.h"
#include "itk_image_type.h"
#include "logfile.h"
#include "plm_image.h"
#include "plm_warp.h"
#include "pointset.h"
#include "print_and_exit.h"
#include "pstring.h"
#include "rasterizer.h"
#include "rtds.h"
#include "rtss.h"
#include "rtss_structure.h"
#include "rtss_structure_set.h"
#include "ss_list_io.h"
#include "ss_img_extract.h"
#include "warp_parms.h"
#include "xio_structures.h"

static void
compose_prefix_fn (
    Pstring *fn, 
    const Pstring &output_prefix, 
    const Pstring &structure_name,
    const char* extension
)
{
    fn->format ("%s/%s.%s", 
        (const char*) output_prefix, 
        (const char*) structure_name, 
        extension);
}

Rtss::Rtss (Rtds *rtds) {
    m_cxt = 0;
    m_ss_img = 0;
    m_labelmap = 0;
    if (rtds) {
        m_meta.set_parent (&rtds->m_meta);
    }
}

Rtss::~Rtss () {
    clear ();
}

void
Rtss::clear () {
    if (this->m_cxt) {
        delete this->m_cxt;
        this->m_cxt = 0;
    }
    if (this->m_ss_img) {
        delete this->m_ss_img;
        this->m_ss_img = 0;
    }
    if (this->m_labelmap) {
        delete this->m_labelmap;
        this->m_labelmap = 0;
    }
}

void
Rtss::load (const char *ss_img, const char *ss_list)
{
    /* Load ss_img */
    if (m_ss_img) {
        delete this->m_ss_img;
    }
    if (ss_img && file_exists (ss_img)) {
        this->m_ss_img = plm_image_load_native (ss_img);
    }

    /* Load ss_list */
    if (this->m_cxt) {
        delete this->m_cxt;
    }
    if (ss_list && file_exists (ss_list)) {
        printf ("Trying to load ss_list: %s\n", ss_list);
        this->m_cxt = ss_list_load (0, ss_list);
    }
}

void
Rtss::load_prefix (const Pstring &prefix_dir)
{
    /* Clear out any existing structures */
    this->clear ();

    /* Load the list of files in the directory */
    Dir_list dl;
    dl.load (prefix_dir.c_str());

    /* Make a quick pass through the directory to find the number of 
       files.  This is used to size the ss_img. */
    int max_structures = 0;
    for (int i = 0; i < dl.num_entries; i++) {
        /* Look at filename, make sure it is an mha file */
        const char *entry = dl.entries[i];
	if (!extension_is (entry, ".mha")) {
            continue;
        }
        max_structures++;
    }
    int out_vec_len = 1 + (max_structures - 1) / 8;
    if (out_vec_len < 2) out_vec_len = 2;

    /* Make a second pass that actually loads the files */
    bool first = true;
    int bit = 0;
    UCharVecImageType::Pointer ss_img;
    Plm_image_header ss_img_pih;
    for (int i = 0; i < dl.num_entries; i++) {
        /* Look at filename, make sure it is an mha file */
        const char *entry = dl.entries[i];
	if (!extension_is (entry, ".mha")) {
            continue;
        }

        /* Grab the structure name from the filename */
        char *structure_name = strdup (entry);
        strip_extension (structure_name);
        lprintf ("Loading structure: %s\n", structure_name);

        /* Load the file */
        Pstring input_fn;
        input_fn.format ("%s/%s", prefix_dir.c_str(), entry);
        Plm_image img (input_fn.c_str(), PLM_IMG_TYPE_ITK_UCHAR);
        Plm_image_header pih (img);

        if (first) {
            /* Create ss_image with same resolution as first image */
            this->m_ss_img = new Plm_image;
            ss_img = UCharVecImageType::New ();
            itk_image_set_header (ss_img, pih);
            ss_img->SetVectorLength (out_vec_len);
            ss_img->Allocate ();
            this->m_ss_img->set_itk (ss_img);
            Plm_image_header::clone (&ss_img_pih, &pih);

            /* Create ss_list to hold strucure names */
            this->m_cxt = new Rtss_structure_set;
            this->m_cxt->set_geometry (this->m_ss_img);

            first = false;
        } else {
            if (!Plm_image_header::compare (&pih, &ss_img_pih)) {
                print_and_exit ("Image size mismatch when loading prefix_dir");
            }
        }

        /* Add name to ss_list */
        Rtss_structure* rts = this->m_cxt->add_structure (
            structure_name, "", 
            this->m_cxt->num_structures + 1);
        rts->bit = bit;
        free (structure_name);

        /* GCS FIX: This code is replicated in ss_img_extract */
        unsigned int uchar_no = bit / 8;
        unsigned int bit_no = bit % 8;
        unsigned char bit_mask = 1 << bit_no;
        if (uchar_no > ss_img->GetVectorLength()) {
            print_and_exit ("Error.  Ss_img vector is too small.");
        }

        /* Set up iterators for looping through images */
        typedef itk::ImageRegionConstIterator< UCharImageType > 
            UCharIteratorType;
        typedef itk::ImageRegionIterator< UCharVecImageType > 
            UCharVecIteratorType;
        UCharImageType::Pointer uchar_img = img.itk_uchar();
        UCharIteratorType uchar_img_it (uchar_img, 
            uchar_img->GetLargestPossibleRegion());
        UCharVecIteratorType ss_img_it (ss_img, 
            ss_img->GetLargestPossibleRegion());

        /* Loop through voxels, or'ing them into ss_img */
        /* GCS FIX: This is inefficient, due to undesirable construct 
           and destruct of itk::VariableLengthVector of each pixel */
        for (
            uchar_img_it.GoToBegin(), ss_img_it.GoToBegin();
            !uchar_img_it.IsAtEnd();
            ++uchar_img_it, ++ss_img_it
        ) {
            unsigned char u = uchar_img_it.Get ();
            if (!u) continue;

            itk::VariableLengthVector<unsigned char> v 
                = ss_img_it.Get ();
            v[uchar_no] |= bit_mask;
            ss_img_it.Set (v);
        }            
        bit++;
    }
}

void
Rtss::load_cxt (const Pstring &input_fn, Slice_index *rdd)
{
    this->m_cxt = new Rtss_structure_set;
    cxt_load (this->m_cxt, &this->m_meta, rdd, (const char*) input_fn);
}

void
Rtss::load_gdcm_rtss (const char *input_fn, Slice_index *rdd)
{
#if GDCM_VERSION_1
    this->m_cxt = new Rtss_structure_set;
    gdcm_rtss_load (this, rdd, &this->m_meta, input_fn);
#endif
}

void
Rtss::load_xio (const Xio_studyset& studyset)
{
    this->m_cxt = new Rtss_structure_set;
    printf ("calling xio_structures_load\n");
    xio_structures_load (this->m_cxt, studyset);
}

size_t
Rtss::get_num_structures ()
{
    if (m_cxt) {
        return m_cxt->num_structures;
    }
    return 0;
}

std::string
Rtss::get_structure_name (size_t index)
{
    if (m_cxt) {
        return m_cxt->get_structure_name (index);
    }
    return 0;
}

UCharImageType::Pointer
Rtss::get_structure_image (int index)
{
    if (!m_ss_img) {
        print_and_exit (
            "Error extracting unknown structure image (no ssi %d)\n", index);
    }

    if (!m_cxt) {
        print_and_exit (
            "Error extracting unknown structure image (no cxt %d)\n", index);
    }

    Rtss_structure *curr_structure = m_cxt->slist[index];
    int bit = curr_structure->bit;

    if (bit == -1) {
        print_and_exit (
            "Error extracting unknown structure image (no bit %d)\n", index);
    }
    UCharImageType::Pointer prefix_img 
        = ss_img_extract_bit (m_ss_img, bit);

    return prefix_img;
}

void
Rtss::save_colormap (const Pstring &colormap_fn)
{
    ss_list_save_colormap (this->m_cxt, (const char*) colormap_fn);
}

void
Rtss::save_cxt (
    Slice_index *rdd, 
    const Pstring &cxt_fn, 
    bool prune_empty
)
{
    cxt_save (this->m_cxt, &this->m_meta, rdd, (const char*) cxt_fn, 
        prune_empty);
}

void
Rtss::save_gdcm_rtss (
    const char *output_dir, 
    Slice_index *rdd
)
{
    char fn[_MAX_PATH];

    /* Perform destructive keyholization of the cxt.  This is necessary 
       because DICOM-RT requires that structures with holes be defined 
       using a single structure */
    this->m_cxt->keyholize ();

    /* Some systems (GE ADW) do not allow special characters in 
       structure names.  */
    this->m_cxt->adjust_structure_names ();

    if (rdd) {
        this->apply_dicom_dir (rdd);
    }

    snprintf (fn, _MAX_PATH, "%s/%s", output_dir, "ss.dcm");

#if GDCM_VERSION_1
    gdcm_rtss_save (this, rdd, fn);
#else
    /* GDCM 2 not implemented -- you're out of luck. */
#endif
}

void
Rtss::save_fcsv (
    const Rtss_structure *curr_structure, 
    const Pstring& fn
)
{
    Labeled_pointset pointset;

    for (size_t j = 0; j < curr_structure->num_contours; j++) {
        Rtss_polyline *curr_polyline = curr_structure->pslist[j];
        for (int k = 0; k < curr_polyline->num_vertices; k++) {
            pointset.insert_lps ("", curr_polyline->x[k],
                curr_polyline->y[k], curr_polyline->z[k]);
        }
    }

    pointset.save_fcsv ((const char*) fn);
}

void
Rtss::save_prefix_fcsv (const Pstring &output_prefix)
{
    if (!this->m_cxt) {
        print_and_exit (
            "Error: save_prefix_fcsv() tried to save a RTSS without a CXT\n");
    }

    for (size_t i = 0; i < m_cxt->num_structures; i++)
    {
        Pstring fn;
        Rtss_structure *curr_structure = m_cxt->slist[i];

        compose_prefix_fn (&fn, output_prefix, curr_structure->name, "fcsv");
        save_fcsv (curr_structure, fn);
    }
}

void
Rtss::save_ss_image (const Pstring &ss_img_fn)
{
    if (!this->m_ss_img) {
        print_and_exit (
            "Error: save_ss_image() tried to write a non-existant file");
    }
    if (this->m_ss_img->m_type == PLM_IMG_TYPE_GPUIT_UCHAR_VEC
        || this->m_ss_img->m_type == PLM_IMG_TYPE_ITK_UCHAR_VEC) 
    {
        /* Image type must be uchar vector */
        this->m_ss_img->convert (PLM_IMG_TYPE_ITK_UCHAR_VEC);
    }
    else {
        /* Image type must be uint32_t */
        this->m_ss_img->convert (PLM_IMG_TYPE_ITK_ULONG);
    }

    /* Set metadata */
    // this->m_ss_img->set_metadata ("I am a", "Structure set");

    this->m_ss_img->save_image ((const char*) ss_img_fn);
}

void
Rtss::save_labelmap (const Pstring &labelmap_fn)
{
    this->m_labelmap->save_image ((const char*) labelmap_fn);
}

void
Rtss::save_prefix (const Pstring &output_prefix)
{
    if (!m_ss_img) {
        return;
    }

    if (!m_cxt) {
        printf ("WTF???\n");
    }

    for (size_t i = 0; i < m_cxt->num_structures; i++)
    {
        Pstring fn;
        Rtss_structure *curr_structure = m_cxt->slist[i];
        int bit = curr_structure->bit;

        if (bit == -1) continue;
        UCharImageType::Pointer prefix_img 
            = ss_img_extract_bit (m_ss_img, bit);

        compose_prefix_fn (&fn, output_prefix, curr_structure->name, "mha");
        itk_image_save (prefix_img, (const char*) fn);
    }
}

void
Rtss::save_ss_list (const Pstring &ss_list_fn)
{
    ss_list_save (this->m_cxt, (const char*) ss_list_fn);
}

void
Rtss::save_xio (Xio_ct_transform *xio_transform, Xio_version xio_version, 
    const Pstring &output_dir)
{
    xio_structures_save (this->m_cxt, &m_meta, xio_transform,
        xio_version, (const char*) output_dir);
}

UInt32ImageType::Pointer
Rtss::get_ss_img_uint32 (void)
{
    if (!this->m_ss_img) {
        print_and_exit ("Sorry, can't get_ss_img()\n");
    }
    m_ss_img->convert (PLM_IMG_TYPE_ITK_ULONG);
    return this->m_ss_img->m_itk_uint32;
}

UCharVecImageType::Pointer
Rtss::get_ss_img_uchar_vec (void)
{
    if (!this->m_ss_img) {
        print_and_exit ("Sorry, can't get_ss_img()\n");
    }
    m_ss_img->convert (PLM_IMG_TYPE_ITK_UCHAR_VEC);
    return this->m_ss_img->m_itk_uchar_vec;
}

void
Rtss::apply_dicom_dir (const Slice_index *rdd)
{
    if (!this->m_cxt) {
        return;
    }

    if (!rdd || !rdd->m_loaded) {
        return;
    }

    /* Geometry */
    for (int d = 0; d < 3; d++) {
        this->m_cxt->m_offset[d] = rdd->m_pih.m_origin[d];
        this->m_cxt->m_dim[d] = rdd->m_pih.Size(d);
        this->m_cxt->m_spacing[d] = rdd->m_pih.m_spacing[d];
    }

    /* Slice numbers and slice uids */
    for (size_t i = 0; i < this->m_cxt->num_structures; i++) {
        Rtss_structure *curr_structure = this->m_cxt->slist[i];
        for (size_t j = 0; j < curr_structure->num_contours; j++) {
            Rtss_polyline *curr_polyline = curr_structure->pslist[j];
            if (curr_polyline->num_vertices <= 0) {
                continue;
            }
            rdd->get_slice_info (
                &curr_polyline->slice_no,
                &curr_polyline->ct_slice_uid,
                curr_polyline->z[0]);
        }
    }
}

void
Rtss::convert_ss_img_to_cxt (void)
{
    /* Only convert if ss_img found */
    if (!this->m_ss_img) {
        return;
    }

    /* Allocate memory for cxt */
    bool use_existing_bits;
    if (this->m_cxt) {
        use_existing_bits = true;
    }
    else {
        this->m_cxt = new Rtss_structure_set;
        use_existing_bits = false;
    }

    /* Copy geometry from ss_img to cxt */
    this->m_cxt->set_geometry (this->m_ss_img);

    if (this->m_ss_img->m_type == PLM_IMG_TYPE_GPUIT_UCHAR_VEC
        || this->m_ss_img->m_type == PLM_IMG_TYPE_ITK_UCHAR_VEC) 
    {
        /* Image type must be uchar vector */
        this->m_ss_img->convert (PLM_IMG_TYPE_ITK_UCHAR_VEC);

        /* Do extraction */
        lprintf ("Doing extraction\n");
        cxt_extract (this->m_cxt, this->m_ss_img->m_itk_uchar_vec, 
            -1, use_existing_bits);
    }
    else {
        /* Image type must be uint32_t */
        this->m_ss_img->convert (PLM_IMG_TYPE_ITK_ULONG);

        /* Do extraction */
        lprintf ("Doing extraction\n");
        cxt_extract (this->m_cxt, this->m_ss_img->m_itk_uint32, -1, 
            use_existing_bits);
    }
}

void
Rtss::convert_to_uchar_vec (void)
{
    if (!this->m_ss_img) {
        print_and_exit (
            "Error: convert_to_uchar_vec() requires an image");
    }
    this->m_ss_img->convert (PLM_IMG_TYPE_ITK_UCHAR_VEC);
}

void
Rtss::cxt_re_extract (void)
{
    this->m_cxt->free_all_polylines ();
    if (this->m_ss_img->m_type == PLM_IMG_TYPE_GPUIT_UCHAR_VEC
        || this->m_ss_img->m_type == PLM_IMG_TYPE_ITK_UCHAR_VEC) 
    {
        this->m_ss_img->convert (PLM_IMG_TYPE_ITK_UCHAR_VEC);
        cxt_extract (this->m_cxt, this->m_ss_img->m_itk_uchar_vec, 
            this->m_cxt->num_structures, true);
    }
    else {
        this->m_ss_img->convert (PLM_IMG_TYPE_ITK_ULONG);
        cxt_extract (this->m_cxt, this->m_ss_img->m_itk_uint32, 
            this->m_cxt->num_structures, true);
    }
}

void
Rtss::prune_empty (void)
{
    if (this->m_cxt) {
        this->m_cxt->prune_empty ();
    }
}

void
Rtss::rasterize (
    Plm_image_header *pih,
    bool want_labelmap,
    bool xor_overlapping
)
{
    /* Rasterize structure sets */
    Rasterizer rasterizer;

#if (PLM_CONFIG_USE_SS_IMAGE_VEC)
    printf ("Setting use_ss_img_vec to true!\n");
    bool use_ss_img_vec = true;
#else
    bool use_ss_img_vec = false;
#endif

    printf ("Rasterizing...\n");
    rasterizer.rasterize (this->m_cxt, pih, false, want_labelmap, true,
        use_ss_img_vec, xor_overlapping);

    /* Convert rasterized structure sets from vol to plm_image */
    printf ("Converting...\n");
    if (want_labelmap) {
        this->m_labelmap = new Plm_image;
        this->m_labelmap->set_gpuit (rasterizer.labelmap_vol);
        rasterizer.labelmap_vol = 0;
    }
    if (this->m_ss_img) {
        delete this->m_ss_img;
    }
    this->m_ss_img = new Plm_image;

    if (use_ss_img_vec) {
        this->m_ss_img->set_itk (rasterizer.m_ss_img->m_itk_uchar_vec);
    }
    else {
        this->m_ss_img->set_gpuit (rasterizer.m_ss_img->vol());
        rasterizer.m_ss_img->m_gpuit = 0;
    }

    printf ("Finished rasterization.\n");
}

void
Rtss::set_geometry (const Plm_image_header *pih)
{
    if (this->m_cxt) {
        this->m_cxt->set_geometry (pih);
    }
}

void
Rtss::find_rasterization_geometry (Plm_image_header *pih)
{
    if (this->m_cxt) {
        this->m_cxt->find_rasterization_geometry (pih);
    }
}

void
Rtss::warp (
    Xform *xf, 
    Plm_image_header *pih, 
    bool use_itk)
{
    Plm_image *tmp;

    if (this->m_labelmap) {
        printf ("Warping labelmap.\n");
        tmp = new Plm_image;
        plm_warp (tmp, 0, xf, pih, this->m_labelmap, 0, use_itk, 0);
        delete this->m_labelmap;
        this->m_labelmap = tmp;
        this->m_labelmap->convert (PLM_IMG_TYPE_ITK_ULONG);
    }

    if (this->m_ss_img) {
        printf ("Warping ss_img.\n");
        tmp = new Plm_image;
        plm_warp (tmp, 0, xf, pih, this->m_ss_img, 0, use_itk, 0);
        delete this->m_ss_img;
        this->m_ss_img = tmp;
    }

    /* The cxt polylines are now obsolete, but we can't delete it because 
       it contains our "bits", used e.g. by prefix extraction.  */
}

void
Rtss::warp (
    Xform *xf, 
    Plm_image_header *pih, 
    Warp_parms *parms)
{
    this->warp (xf, pih, parms->use_itk);
}
