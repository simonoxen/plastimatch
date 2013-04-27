/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"

#if GDCM_VERSION_1
#include "gdcm1_dose.h"
#include "gdcm1_rtss.h"
#endif
#include "cxt_extract.h"
#include "cxt_io.h"
#include "dir_list.h"
#include "file_util.h"
#include "itk_image_save.h"
#include "itk_image_type.h"
#include "logfile.h"
#include "path_util.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_path.h"
#include "plm_warp.h"
#include "pointset.h"
#include "print_and_exit.h"
#include "pstring.h"
#include "rasterizer.h"
#include "rt_study.h"
#include "rtss_structure.h"
#include "rtss_structure_set.h"
#include "segmentation.h"
#include "slice_index.h"
#include "ss_list_io.h"
#include "ss_img_extract.h"
#include "string_util.h"
#include "warp_parms.h"
#include "xio_structures.h"

class Segmentation_private {
public:
    Metadata *m_meta;           /* Metadata specific to this ss_image */
    Plm_image *m_labelmap;      /* Structure set lossy bitmap form */
    Plm_image *m_ss_img;        /* Structure set in lossless bitmap form */
    Rtss_structure_set::Pointer m_cxt;  /* Structure set in polyline form */

public:
    Segmentation_private () {
        m_meta = new Metadata;
        m_labelmap = 0;
        m_ss_img = 0;
    }
    ~Segmentation_private () {
        delete m_meta;
    }
};

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

Segmentation::Segmentation (Rt_study *rtds)
{
    this->d_ptr = new Segmentation_private;

    if (rtds) {
        d_ptr->m_meta->set_parent (rtds->get_metadata());
    }
}

Segmentation::~Segmentation ()
{
    clear ();

    delete this->d_ptr;
}

void
Segmentation::clear ()
{
    if (d_ptr->m_cxt) {
        d_ptr->m_cxt.reset();
    }
    if (d_ptr->m_ss_img) {
        delete d_ptr->m_ss_img;
        d_ptr->m_ss_img = 0;
    }
    if (d_ptr->m_labelmap) {
        delete d_ptr->m_labelmap;
        d_ptr->m_labelmap = 0;
    }
}

void
Segmentation::load (const char *ss_img, const char *ss_list)
{
    /* Load ss_img */
    if (d_ptr->m_ss_img) {
        delete d_ptr->m_ss_img;
    }
    if (ss_img && file_exists (ss_img)) {
        d_ptr->m_ss_img = plm_image_load_native (ss_img);
    }

    /* Load ss_list */
    if (d_ptr->m_cxt) {
        d_ptr->m_cxt.reset();
    }
    if (ss_list && file_exists (ss_list)) {
        printf ("Trying to load ss_list: %s\n", ss_list);
        d_ptr->m_cxt.reset (ss_list_load (0, ss_list));
    }
}

void
Segmentation::load_prefix (const char *prefix_dir)
{
    Pstring pd = prefix_dir;
    this->load_prefix (pd);
}

void
Segmentation::load_prefix (const Pstring &prefix_dir)
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
        /* Look at filename, make sure it is an mha or nrrd file */
        const char *entry = dl.entries[i];
	if (!extension_is (entry, ".mha") 
            && !extension_is (entry, ".nrrd"))
        {
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
        /* Look at filename, make sure it is an mha or nrrd file */
        const char *entry = dl.entries[i];
	if (!extension_is (entry, ".mha") 
            && !extension_is (entry, ".nrrd"))
        {
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
            d_ptr->m_ss_img = new Plm_image;
            ss_img = UCharVecImageType::New ();
            itk_image_set_header (ss_img, pih);
            ss_img->SetVectorLength (out_vec_len);
            ss_img->Allocate ();

            /* GCS NOTE: For some reason, ss_img->FillBuffer (0) 
               doesn't do what I want. */
            itk::VariableLengthVector<unsigned char> v;
            v.SetSize (out_vec_len);
            v.Fill (0);
            ss_img->FillBuffer (v);
#if defined (commentout)
            typedef itk::ImageRegionIterator< 
                UCharVecImageType > UCharVecIteratorType;
            UCharVecIteratorType it (ss_img, pih.m_region);
            for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
                it.Set (v);
            }
#endif

            d_ptr->m_ss_img->set_itk (ss_img);
            Plm_image_header::clone (&ss_img_pih, &pih);

            /* Create ss_list to hold strucure names */
            d_ptr->m_cxt = Rtss_structure_set::New();
            d_ptr->m_cxt->set_geometry (d_ptr->m_ss_img);

            first = false;
        } else {
            if (!Plm_image_header::compare (&pih, &ss_img_pih)) {
                print_and_exit ("Image size mismatch when loading prefix_dir");
            }
        }

        /* Add name to ss_list */
        d_ptr->m_cxt->add_structure (
            structure_name, "", 
            d_ptr->m_cxt->num_structures + 1,
            bit);
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

        /* Move to next bit */
        bit++;
    }
}

void
Segmentation::load_cxt (const Pstring &input_fn, Slice_index *rdd)
{
    d_ptr->m_cxt = Rtss_structure_set::New();
    cxt_load (d_ptr->m_cxt.get(), d_ptr->m_meta, rdd, (const char*) input_fn);
}

void
Segmentation::load_gdcm_rtss (const char *input_fn, Slice_index *rdd)
{
#if GDCM_VERSION_1
    d_ptr->m_cxt = Rtss_structure_set::New();
    gdcm_rtss_load (d_ptr->m_cxt.get(), d_ptr->m_meta, rdd, input_fn);
#endif
}

void
Segmentation::load_xio (const Xio_studyset& studyset)
{
    d_ptr->m_cxt = Rtss_structure_set::New();
    printf ("calling xio_structures_load\n");
    xio_structures_load (d_ptr->m_cxt.get(), studyset);
}

size_t
Segmentation::get_num_structures ()
{
    if (d_ptr->m_cxt) {
        return d_ptr->m_cxt->num_structures;
    }
    return 0;
}

std::string
Segmentation::get_structure_name (size_t index)
{
    if (d_ptr->m_cxt) {
        return d_ptr->m_cxt->get_structure_name (index);
    }
    return 0;
}

UCharImageType::Pointer
Segmentation::get_structure_image (int index)
{
    if (!d_ptr->m_ss_img) {
        print_and_exit (
            "Error extracting unknown structure image (no ssi %d)\n", index);
    }

    if (!d_ptr->m_cxt) {
        print_and_exit (
            "Error extracting unknown structure image (no cxt %d)\n", index);
    }

    Rtss_structure *curr_structure = d_ptr->m_cxt->slist[index];
    int bit = curr_structure->bit;

    if (bit == -1) {
        print_and_exit (
            "Error extracting unknown structure image (no bit %d)\n", index);
    }
    UCharImageType::Pointer prefix_img 
        = ss_img_extract_bit (d_ptr->m_ss_img, bit);

    return prefix_img;
}

void
Segmentation::save_colormap (const Pstring &colormap_fn)
{
    ss_list_save_colormap (d_ptr->m_cxt.get(), (const char*) colormap_fn);
}

void
Segmentation::save_cxt (
    Slice_index *rdd, 
    const Pstring &cxt_fn, 
    bool prune_empty
)
{
    cxt_save (d_ptr->m_cxt.get(), d_ptr->m_meta, rdd, (const char*) cxt_fn, 
        prune_empty);
}

void
Segmentation::save_gdcm_rtss (
    const char *output_dir, 
    Slice_index *rdd
)
{
    char fn[_MAX_PATH];

    /* Perform destructive keyholization of the cxt.  This is necessary 
       because DICOM-RT requires that structures with holes be defined 
       using a single structure */
    d_ptr->m_cxt->keyholize ();

    /* Some systems (GE ADW) do not allow special characters in 
       structure names.  */
    d_ptr->m_cxt->adjust_structure_names ();

    if (rdd) {
        this->apply_dicom_dir (rdd);
    }

    snprintf (fn, _MAX_PATH, "%s/%s", output_dir, "rtss.dcm");

#if GDCM_VERSION_1
    gdcm_rtss_save (d_ptr->m_cxt.get(), d_ptr->m_meta, rdd, fn);
#else
    /* GDCM 2 not implemented -- you're out of luck. */
#endif
}

void
Segmentation::save_fcsv (
    const Rtss_structure *curr_structure, 
    const Pstring& fn
)
{
    Labeled_pointset pointset;

    for (size_t j = 0; j < curr_structure->num_contours; j++) {
        Rtss_contour *curr_polyline = curr_structure->pslist[j];
        for (int k = 0; k < curr_polyline->num_vertices; k++) {
            pointset.insert_lps ("", curr_polyline->x[k],
                curr_polyline->y[k], curr_polyline->z[k]);
        }
    }

    pointset.save_fcsv ((const char*) fn);
}

void
Segmentation::save_prefix_fcsv (const Pstring &output_prefix)
{
    if (!d_ptr->m_cxt) {
        print_and_exit (
            "Error: save_prefix_fcsv() tried to save a RTSS without a CXT\n");
    }

    for (size_t i = 0; i < d_ptr->m_cxt->num_structures; i++)
    {
        Pstring fn;
        Rtss_structure *curr_structure = d_ptr->m_cxt->slist[i];

        compose_prefix_fn (&fn, output_prefix, curr_structure->name, "fcsv");
        save_fcsv (curr_structure, fn);
    }
}

void
Segmentation::save_ss_image (const Pstring &ss_img_fn)
{
    if (!d_ptr->m_ss_img) {
        print_and_exit (
            "Error: save_ss_image() tried to write a non-existant file");
    }
    if (d_ptr->m_ss_img->m_type == PLM_IMG_TYPE_GPUIT_UCHAR_VEC
        || d_ptr->m_ss_img->m_type == PLM_IMG_TYPE_ITK_UCHAR_VEC) 
    {
        /* Image type must be uchar vector */
        d_ptr->m_ss_img->convert (PLM_IMG_TYPE_ITK_UCHAR_VEC);
    }
    else {
        /* Image type must be uint32_t */
        d_ptr->m_ss_img->convert (PLM_IMG_TYPE_ITK_ULONG);
    }

    /* Set metadata */
    // d_ptr->m_ss_img->set_metadata ("I am a", "Structure set");

    d_ptr->m_ss_img->save_image ((const char*) ss_img_fn);
}

void
Segmentation::save_labelmap (const Pstring &labelmap_fn)
{
    d_ptr->m_labelmap->save_image ((const char*) labelmap_fn);
}

void
Segmentation::save_prefix (const std::string &output_prefix,
    const std::string& extension)
{
    if (!d_ptr->m_ss_img) {
        return;
    }

    if (!d_ptr->m_cxt) {
        printf ("WTF???\n");
    }

    for (size_t i = 0; i < d_ptr->m_cxt->num_structures; i++)
    {
        std::string fn;
        Rtss_structure *curr_structure = d_ptr->m_cxt->slist[i];
        int bit = curr_structure->bit;

        if (bit == -1) continue;
        UCharImageType::Pointer prefix_img 
            = ss_img_extract_bit (d_ptr->m_ss_img, bit);

        fn = string_format ("%s/%s.%s", 
            output_prefix.c_str(),
            curr_structure->name.c_str(),
            extension.c_str());
        itk_image_save (prefix_img, fn.c_str());
    }
}

/* GCS FIX: This is obsolete, and should invoke the above function */
void
Segmentation::save_prefix (const Pstring &output_prefix)
{
    if (!d_ptr->m_ss_img) {
        return;
    }

    if (!d_ptr->m_cxt) {
        printf ("WTF???\n");
    }

    for (size_t i = 0; i < d_ptr->m_cxt->num_structures; i++)
    {
        Pstring fn;
        Rtss_structure *curr_structure = d_ptr->m_cxt->slist[i];
        int bit = curr_structure->bit;

        if (bit == -1) continue;
        UCharImageType::Pointer prefix_img 
            = ss_img_extract_bit (d_ptr->m_ss_img, bit);

        compose_prefix_fn (&fn, output_prefix, curr_structure->name, "mha");
        itk_image_save (prefix_img, (const char*) fn);
    }
}

void
Segmentation::save_prefix (const char *output_prefix)
{
    Pstring op = output_prefix;
    this->save_prefix (op);
}

void
Segmentation::save_ss_list (const Pstring &ss_list_fn)
{
    ss_list_save (d_ptr->m_cxt.get(), (const char*) ss_list_fn);
}

void
Segmentation::save_xio (Xio_ct_transform *xio_transform, Xio_version xio_version, 
    const Pstring &output_dir)
{
    xio_structures_save (d_ptr->m_cxt.get(), d_ptr->m_meta, xio_transform,
        xio_version, (const char*) output_dir);
}

UInt32ImageType::Pointer
Segmentation::get_ss_img_uint32 (void)
{
    if (!d_ptr->m_ss_img) {
        print_and_exit ("Sorry, can't get_ss_img()\n");
    }
    d_ptr->m_ss_img->convert (PLM_IMG_TYPE_ITK_ULONG);
    return d_ptr->m_ss_img->m_itk_uint32;
}

UCharVecImageType::Pointer
Segmentation::get_ss_img_uchar_vec (void)
{
    if (!d_ptr->m_ss_img) {
        print_and_exit ("Sorry, can't get_ss_img()\n");
    }
    d_ptr->m_ss_img->convert (PLM_IMG_TYPE_ITK_UCHAR_VEC);
    return d_ptr->m_ss_img->m_itk_uchar_vec;
}

void
Segmentation::apply_dicom_dir (const Slice_index *rdd)
{
    if (!d_ptr->m_cxt) {
        return;
    }

    if (!rdd || !rdd->m_loaded) {
        return;
    }

    d_ptr->m_cxt->apply_slice_index (rdd);
}

void
Segmentation::convert_ss_img_to_cxt (void)
{
    /* Only convert if ss_img found */
    if (!d_ptr->m_ss_img) {
        return;
    }

    /* Allocate memory for cxt */
    bool use_existing_bits;
    if (d_ptr->m_cxt) {
        use_existing_bits = true;
    }
    else {
        d_ptr->m_cxt = Rtss_structure_set::New();
        use_existing_bits = false;
    }

    /* Copy geometry from ss_img to cxt */
    d_ptr->m_cxt->set_geometry (d_ptr->m_ss_img);

    if (d_ptr->m_ss_img->m_type == PLM_IMG_TYPE_GPUIT_UCHAR_VEC
        || d_ptr->m_ss_img->m_type == PLM_IMG_TYPE_ITK_UCHAR_VEC) 
    {
        /* Image type must be uchar vector */
        d_ptr->m_ss_img->convert (PLM_IMG_TYPE_ITK_UCHAR_VEC);

        /* Do extraction */
        lprintf ("Doing extraction\n");
        cxt_extract (d_ptr->m_cxt.get(), d_ptr->m_ss_img->m_itk_uchar_vec, 
            -1, use_existing_bits);
    }
    else {
        /* Image type must be uint32_t */
        d_ptr->m_ss_img->convert (PLM_IMG_TYPE_ITK_ULONG);

        /* Do extraction */
        lprintf ("Doing extraction\n");
        cxt_extract (d_ptr->m_cxt.get(), d_ptr->m_ss_img->m_itk_uint32, -1, 
            use_existing_bits);
    }
}

void
Segmentation::convert_to_uchar_vec (void)
{
    if (!d_ptr->m_ss_img) {
        print_and_exit (
            "Error: convert_to_uchar_vec() requires an image");
    }
    d_ptr->m_ss_img->convert (PLM_IMG_TYPE_ITK_UCHAR_VEC);
}

void
Segmentation::cxt_re_extract (void)
{
    d_ptr->m_cxt->free_all_polylines ();
    if (d_ptr->m_ss_img->m_type == PLM_IMG_TYPE_GPUIT_UCHAR_VEC
        || d_ptr->m_ss_img->m_type == PLM_IMG_TYPE_ITK_UCHAR_VEC) 
    {
        d_ptr->m_ss_img->convert (PLM_IMG_TYPE_ITK_UCHAR_VEC);
        cxt_extract (d_ptr->m_cxt.get(), d_ptr->m_ss_img->m_itk_uchar_vec, 
            d_ptr->m_cxt->num_structures, true);
    }
    else {
        d_ptr->m_ss_img->convert (PLM_IMG_TYPE_ITK_ULONG);
        cxt_extract (d_ptr->m_cxt.get(), d_ptr->m_ss_img->m_itk_uint32, 
            d_ptr->m_cxt->num_structures, true);
    }
}

void
Segmentation::prune_empty (void)
{
    if (d_ptr->m_cxt) {
        d_ptr->m_cxt->prune_empty ();
    }
}

void
Segmentation::rasterize (
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
    rasterizer.rasterize (d_ptr->m_cxt.get(), pih, false, want_labelmap, true,
        use_ss_img_vec, xor_overlapping);

    /* Convert rasterized structure sets from vol to plm_image */
    printf ("Converting...\n");
    if (want_labelmap) {
        d_ptr->m_labelmap = new Plm_image;
        d_ptr->m_labelmap->set_volume (rasterizer.labelmap_vol);
        rasterizer.labelmap_vol = 0;
    }
    if (d_ptr->m_ss_img) {
        delete d_ptr->m_ss_img;
    }
    d_ptr->m_ss_img = new Plm_image;

    if (use_ss_img_vec) {
        d_ptr->m_ss_img->set_itk (rasterizer.m_ss_img->m_itk_uchar_vec);
    }
    else {
        Volume *v = rasterizer.m_ss_img->steal_volume();
        d_ptr->m_ss_img->set_volume (v);
    }

    printf ("Finished rasterization.\n");
}

void
Segmentation::set_geometry (const Plm_image_header *pih)
{
    if (d_ptr->m_cxt) {
        d_ptr->m_cxt->set_geometry (pih);
    }
}

void
Segmentation::find_rasterization_geometry (Plm_image_header *pih)
{
    if (d_ptr->m_cxt) {
        d_ptr->m_cxt->find_rasterization_geometry (pih);
    }
}

void
Segmentation::warp (
    Xform *xf, 
    Plm_image_header *pih, 
    bool use_itk)
{
    Plm_image *tmp;

    if (d_ptr->m_labelmap) {
        printf ("Warping labelmap.\n");
        tmp = new Plm_image;
        plm_warp (tmp, 0, xf, pih, d_ptr->m_labelmap, 0, use_itk, 0);
        delete d_ptr->m_labelmap;
        d_ptr->m_labelmap = tmp;
        d_ptr->m_labelmap->convert (PLM_IMG_TYPE_ITK_ULONG);
    }

    if (d_ptr->m_ss_img) {
        printf ("Warping ss_img.\n");
        tmp = new Plm_image;
        plm_warp (tmp, 0, xf, pih, d_ptr->m_ss_img, 0, use_itk, 0);
        delete d_ptr->m_ss_img;
        d_ptr->m_ss_img = tmp;
    }

    /* The cxt polylines are now obsolete, but we can't delete it because 
       it contains our "bits", used e.g. by prefix extraction.  */
}

void
Segmentation::warp (
    Xform *xf, 
    Plm_image_header *pih, 
    Warp_parms *parms)
{
    this->warp (xf, pih, parms->use_itk);
}

bool
Segmentation::have_ss_img ()
{
    return d_ptr->m_ss_img != 0;
}

void
Segmentation::set_ss_img (UCharImageType::Pointer ss_img)
{
    if (d_ptr->m_ss_img) {
        delete d_ptr->m_ss_img;
    }
    d_ptr->m_ss_img = new Plm_image;
    d_ptr->m_ss_img->set_itk (ss_img);
}

Plm_image*
Segmentation::get_ss_img ()
{
    return d_ptr->m_ss_img;
}

bool
Segmentation::have_structure_set ()
{
    return d_ptr->m_cxt != 0;
}


Rtss_structure_set::Pointer
Segmentation::get_structure_set ()
{
    return d_ptr->m_cxt;
}

Rtss_structure_set *
Segmentation::get_structure_set_raw ()
{
    return d_ptr->m_cxt.get();
}

void
Segmentation::set_structure_set (Rtss_structure_set::Pointer rtss_ss)
{
    d_ptr->m_cxt = rtss_ss;
}

void
Segmentation::set_structure_set (Rtss_structure_set *rtss_ss)
{
    d_ptr->m_cxt.reset (rtss_ss);
}
