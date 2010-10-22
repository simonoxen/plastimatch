/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "bstring_util.h"
#include "cxt_extract.h"
#include "cxt_to_mha.h"
#include "cxt_apply_dicom.h"
#include "itk_image_save.h"
#include "gdcm_rtss.h"
#include "plm_warp.h"
#include "rtss.h"
#include "ss_image.h"
#include "ss_img_extract.h"
#include "ss_list_io.h"
#include "xio_structures.h"

static void
compose_prefix_fn (
    CBString *fn, 
    const CBString &output_prefix, 
    const CBString &structure_name
)
{
    fn->format ("%s_%s.%s", 
	(const char*) output_prefix, 
	(const char*) structure_name, 
	"mha");
}

void
Ss_image::load (const char *ss_img, const char *ss_list)
{
    /* Load ss_img */
    if (m_ss_img) {
	delete this->m_ss_img;
    }
    if (ss_img) {
	this->m_ss_img = plm_image_load_native (ss_img);
    }

    /* Load ss_list */
    if (this->m_ss_list) {
	delete this->m_ss_list;
    }
    if (ss_list) {
	this->m_ss_list = ss_list_load (0, ss_list);
    }
}

void
Ss_image::load_cxt (const CBString &input_fn)
{
    this->m_cxt = new Rtss;
    cxt_load (this->m_cxt, (const char*) input_fn);
}

void
Ss_image::load_gdcm_rtss (const char *input_fn, const char *dicom_dir)
{
    this->m_cxt = new Rtss;
    gdcm_rtss_load (this->m_cxt, input_fn, dicom_dir);
}

void
Ss_image::load_xio (char *input_dir)
{
    this->m_cxt = new Rtss;
    printf ("calling xio_structures_load\n");
    xio_structures_load (this->m_cxt, input_dir);
}

void
Ss_image::save_colormap (const CBString &colormap_fn)
{
    ss_list_save_colormap (this->m_cxt, (const char*) colormap_fn);
}

void
Ss_image::save_cxt (const CBString &cxt_fn, bool prune_empty)
{
    cxt_save (this->m_cxt, (const char*) cxt_fn, false);
}

void
Ss_image::save_gdcm_rtss (const char *dicom_dir)
{
    char fn[_MAX_PATH];
    this->m_cxt->adjust_structure_names ();
    if (bstring_empty (this->m_cxt->ct_study_uid)) {
	/* No structure association currently available.
	   Associate with dicom_dir. */
	cxt_apply_dicom_dir (this->m_cxt, dicom_dir);
    }
    snprintf (fn, _MAX_PATH, "%s/%s", dicom_dir, "ss.dcm");
    gdcm_rtss_save (this->m_cxt, fn, dicom_dir);
}

void
Ss_image::save_ss_image (const CBString &ss_img_fn)
{
    if (!this->m_ss_img) {
	print_and_exit (
	    "Error: save_ss_image() tried to write a non-existant file");
    }
#if (PLM_USE_4D_SS_IMAGE)
    /* Save as 4D uchar */
    this->m_ss_img->convert_to_itk_uchar_4d ();
    this->m_ss_img->save_image ((const char*) ss_img_fn);
#else
    /* Save as 3D uint32 */
    this->m_ss_img->save_image ((const char*) ss_img_fn);
#endif
}

void
Ss_image::save_labelmap (const CBString &labelmap_fn)
{
    this->m_labelmap->save_image ((const char*) labelmap_fn);
}

void
Ss_image::save_prefix (const CBString &output_prefix)
{
    int i;

    if (!m_ss_img) {
	return;
    }

    if (!m_ss_list) {
	printf ("WTF???\n");
    }

    printf ("Trying to save prefix...\n");

    for (i = 0; i < m_ss_list->num_structures; i++)
    {
	CBString fn;
	printf ("Structure %d\n", i);
	printf ("m_ss_list = %p\n", m_ss_list);
	printf ("m_ss_list[%d] = %p\n", i, m_ss_list->slist[i]);
	
	Rtss_structure *curr_structure = m_ss_list->slist[i];
	int bit = curr_structure->bit;

	if (bit == -1) continue;

	m_ss_img->convert (PLM_IMG_TYPE_ITK_ULONG);
	UCharImageType::Pointer prefix_img = ss_img_extract (
	    m_ss_img->m_itk_uint32, bit);

	compose_prefix_fn (&fn, output_prefix, curr_structure->name);
	printf ("Trying to save prefix image: [%d,%d], %s\n", 
	    i, bit, (const char*) fn);
	itk_image_save (prefix_img, (const char*) fn);
    }
    printf ("Done.\n");
}

void
Ss_image::save_ss_list (const CBString &ss_list_fn)
{
    ss_list_save (this->m_cxt, (const char*) ss_list_fn);
}

void
Ss_image::save_xio (Xio_ct_transform *xio_transform, Xio_version xio_version, 
    const CBString &output_dir)
{
    xio_structures_save (this->m_cxt, xio_transform,
	xio_version, (const char*) output_dir);
}

UInt32ImageType::Pointer
Ss_image::get_ss_img (void)
{
    if (!this->m_ss_img) {
	print_and_exit ("Sorry, can't get_ss_img()\n");
    }
    return this->m_ss_img->m_itk_uint32;
}

Rtss*
Ss_image::get_ss_list (void)
{
    if (!this->m_ss_list) {
	print_and_exit ("Sorry, can't get_ss_list()\n");
    }
    return this->m_ss_list;
}

void
Ss_image::apply_dicom_dir (const CBString &dicom_dir)
{
    if (this->m_cxt) {
	cxt_apply_dicom_dir (this->m_cxt, (const char*) dicom_dir);
    }
}

void
Ss_image::convert_ss_img_to_cxt (void)
{
    int num_structs = -1;

    /* Only convert if ss_img found */
    if (!this->m_ss_img) {
	return;
    }

    /* Allocate memory for cxt */
    if (this->m_cxt) {
	delete this->m_cxt;
    }
    this->m_cxt = new Rtss;

    /* Copy geometry from ss_img to cxt */
    this->m_cxt->set_geometry_from_plm_image (
	this->m_ss_img);

    /* Extract polylines */
    num_structs = this->m_ss_list->num_structures;
    /* Image type must be uint32_t for cxt_extract */
    this->m_ss_img->convert (PLM_IMG_TYPE_ITK_ULONG);

    /* Do extraction */
    printf ("Running marching squares\n");
    if (this->m_ss_list) {
	this->m_cxt->clone_empty (this->m_ss_list);
	cxt_extract (this->m_cxt, this->m_ss_img->m_itk_uint32, -1, true);
    } else {
	cxt_extract (this->m_cxt, this->m_ss_img->m_itk_uint32, -1, false);
    }
}

void
Ss_image::cxt_re_extract (void)
{
    this->m_cxt->free_all_polylines ();
    this->m_ss_img->convert (PLM_IMG_TYPE_ITK_ULONG);
    cxt_extract (this->m_cxt, this->m_ss_img->m_itk_uint32, 
	this->m_cxt->num_structures, true);
}

void
Ss_image::prune_empty (void)
{
    if (this->m_cxt) {
	this->m_cxt->prune_empty ();
    }
}

void
Ss_image::rasterize (void)
{
    /* Rasterize structure sets */
    Cxt_to_mha_state *ctm_state;
    printf ("Rasterizing...\n");
    ctm_state = cxt_to_mha_create (this->m_cxt);

    /* Convert rasterized structure sets from vol to plm_image */
    printf ("Converting...\n");
    this->m_labelmap = new Plm_image;
    this->m_labelmap->set_gpuit (ctm_state->labelmap_vol);
    ctm_state->labelmap_vol = 0;
    if (this->m_ss_img) {
	delete this->m_ss_img;
    }
    this->m_ss_img = new Plm_image;
    this->m_ss_img->set_gpuit (ctm_state->ss_img_vol);
    ctm_state->ss_img_vol = 0;

    /* We're done with cxt_state now */
    cxt_to_mha_destroy (ctm_state);

    /* Clone the set of names */
    this->m_ss_list = this->m_cxt->clone_empty (this->m_ss_list);

    printf ("Finished rasterization.\n");
}

void
Ss_image::set_geometry_from_plm_image_header (Plm_image_header *pih)
{
    if (this->m_cxt) {
	this->m_cxt->set_geometry_from_plm_image_header (pih);
    }
}

void
Ss_image::warp (
    Xform *xf, 
    Plm_image_header *pih, 
    Warp_parms *parms)
{
    Plm_image *tmp;

    tmp = new Plm_image;
    plm_warp (tmp, 0, xf, pih, this->m_labelmap, 0, parms->use_itk, 0);
    delete this->m_labelmap;
    this->m_labelmap = tmp;
    this->m_labelmap->convert (PLM_IMG_TYPE_ITK_ULONG);

    tmp = new Plm_image;
    plm_warp (tmp, 0, xf, pih, this->m_ss_img, 0, parms->use_itk, 0);
    delete this->m_ss_img;
    this->m_ss_img = tmp;
    this->m_ss_img->convert (PLM_IMG_TYPE_ITK_ULONG);

    /* The cxt is now obsolete, but we can't delete it because it 
       contains our "bits", used e.g. by prefix extraction.  */
}
