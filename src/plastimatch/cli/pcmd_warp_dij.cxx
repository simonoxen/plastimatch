/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/*  Warp one or more dij matrices based on a vector field */
#include "plmcli_config.h"
#include <math.h>
#include <time.h>
#include "itkImage.h"
#include "itkInterpolateImagePointsFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkVectorLinearInterpolateImageFunction.h"

#include "plmbase.h"

#include "pcmd_warp.h"
#include "plm_path.h"
#include "print_and_exit.h"

typedef unsigned short ushort;
typedef unsigned long ulong;

typedef struct __Ctatts Ctatts;
struct __Ctatts {
    int slice_dimension;
    int slice_number;
    float pixel_size;
    float slice_distance;
    float pos_isocenter_x;
    float pos_isocenter_y;
    float pos_isocenter_z;
    int num_of_voxels_in_ct_cube_z;
};

typedef struct __Dif Dif;
struct __Dif {
    float delta_x;
    float delta_y;
    float delta_z;
    int dimension_ct_x;
    int dimension_ct_y;
    int dimension_ct_z;
    int dimension_dose_x;
    int dimension_dose_y;
    int dimension_dose_z;
    int iso_index_ct_x;
    int iso_index_ct_y;
    int iso_index_ct_z;
    int iso_index_dose_x;
    int iso_index_dose_y;
    int iso_index_dose_z;
};

typedef struct __Pencil_Beam Pencil_Beam;
struct __Pencil_Beam {
    float energy;
    float spot_x;
    float spot_y;
    int nvox;
    ushort* vox;
};

typedef struct __Dij_Matrix Dij_Matrix;
struct __Dij_Matrix {
    float gantry_angle;
    float table_angle;
    float collimator_angle;
    float spot_spacing_dx;
    float spot_spacing_dy;
    float voxel_size_dx;
    float voxel_size_dy;
    float voxel_size_dz;
    int dose_cube_size[3];
    int num_pencil_beams;
    float absolute_dose_coefficient;
};

void
dij_parse_error (void)
{
    fprintf (stderr, "Parse error in reading dij_matrix\n");
    exit (-1);
}

void
dij_write_error (void)
{
    fprintf (stderr, "Error writing dij_matrix\n");
    exit (-1);
}

void
ctatts_parse_error (void)
{
    fprintf (stderr, "Parse error in reading ctatts file\n");
    exit (-1);
}

void
dif_parse_error (void)
{
    fprintf (stderr, "Parse error in reading dif file\n");
    exit (-1);
}

void
load_dif (Dif* dif, const char* dif_in)
{
    int i;
    float f;
    FILE* fp;
    const int BUFLEN = 1024;
    char buf[BUFLEN];

    fp = fopen (dif_in, "rt");
    if (!fp) {
	fprintf (stderr, "Error opening dif file for read: %s\n", dif_in);
	exit (-1);
    }

    /* GCS FIX: I should give an error if not all lines are found */
    while (1) {
	if (!fgets (buf, BUFLEN, fp)) {
	    break;
	}
	if (buf[0] == '\0' || buf[0] == '\n') {
	    /* Empty lines are ok */
	}
	else if (sscanf (buf, "Delta-X %f", &f)) {
	    dif->delta_x = f;
	}
	else if (sscanf (buf, "Delta-Y %f", &f)) {
	    dif->delta_y = f;
	}
	else if (sscanf (buf, "Delta-Z %f", &f)) {
	    dif->delta_z = f;
	}
	else if (sscanf (buf, "Dimension-CT-X %d", &i)) {
	    dif->dimension_ct_x = i;
	}
	else if (sscanf (buf, "Dimension-CT-Y %d", &i)) {
	    dif->dimension_ct_y = i;
	}
	else if (sscanf (buf, "Dimension-CT-Z %d", &i)) {
	    dif->dimension_ct_z = i;
	}
	else if (sscanf (buf, "Dimension-Dose-X %d", &i)) {
	    dif->dimension_dose_x = i;
	}
	else if (sscanf (buf, "Dimension-Dose-Y %d", &i)) {
	    dif->dimension_dose_y = i;
	}
	else if (sscanf (buf, "Dimension-Dose-Z %d", &i)) {
	    dif->dimension_dose_z = i;
	}
	else if (sscanf (buf, "ISO-Index-CT-X %d", &i)) {
	    dif->iso_index_ct_x = i;
	}
	else if (sscanf (buf, "ISO-Index-CT-Y %d", &i)) {
	    dif->iso_index_ct_y = i;
	}
	else if (sscanf (buf, "ISO-Index-CT-Z %d", &i)) {
	    dif->iso_index_ct_z = i;
	}
	else if (sscanf (buf, "ISO-Index-Dose-X %d", &i)) {
	    dif->iso_index_dose_x = i;
	}
	else if (sscanf (buf, "ISO-Index-Dose-Y %d", &i)) {
	    dif->iso_index_dose_y = i;
	}
	else if (sscanf (buf, "ISO-Index-Dose-Z %d", &i)) {
	    dif->iso_index_dose_z = i;
	}
	else {
	    fprintf (stdout, "Bogus Line = %s\n", buf);
	    dif_parse_error ();
	}
    }
    fclose (fp);
}

void
load_ctatts (Ctatts* ctatts, const char* ctatts_in)
{
    int i;
    float f;
    FILE* fp;
    const int BUFLEN = 1024;
    char buf[BUFLEN];

    fp = fopen (ctatts_in, "rt");
    if (!fp) {
	fprintf (stderr, "Error opening ctatts file for read: %s\n", ctatts_in);
	exit (-1);
    }

    /* Skip first line */
    fgets (buf, BUFLEN, fp);

    /* GCS FIX: I should give an error if not all lines are found */
    while (1) {
	if (!fgets (buf, BUFLEN, fp)) {
	    break;
	}
	if (buf[0] == '\0' || buf[0] == '\n') {
	    /* Empty lines are ok */
	}
	else if (sscanf (buf, "slice_dimension %d", &i)) {
	    ctatts->slice_dimension = i;
	}
	else if (sscanf (buf, "slice_number %d", &i)) {
	    ctatts->slice_number = i;
	}
	else if (sscanf (buf, "pixel_size %f", &f)) {
	    ctatts->pixel_size = f;
	}
	else if (sscanf (buf, "slice_distance %f", &f)) {
	    ctatts->slice_distance = f;
	}
	else if (sscanf (buf, "Pos-Isocenter-X %f", &f)) {
	    ctatts->pos_isocenter_x = f;
	}
	else if (sscanf (buf, "Pos-Isocenter-Y %f", &f)) {
	    ctatts->pos_isocenter_y = f;
	}
	else if (sscanf (buf, "Pos-Isocenter-Z %f", &f)) {
	    ctatts->pos_isocenter_z = f;
	}
	else if (sscanf (buf, "Number-Of-Voxels-in-CT-Cube-Z %d", &i)) {
	    ctatts->num_of_voxels_in_ct_cube_z = i;
	}
	else {
	    ctatts_parse_error ();
	}
    }
    fclose (fp);
}

/* This is a "dumb array".  It doesn't know about pixel sizes etc. 
   (x,y,z) are in beamlet coordinate system (IEC, I think).  */
FloatImageType::Pointer
make_output_image (Dij_Matrix* dij_matrix)
{
    FloatImageType::Pointer image = FloatImageType::New();

    FloatImageType::IndexType start;
    start[0] =   0;
    start[1] =   0;
    start[2] =   0;

    FloatImageType::SizeType size;
    size[0]  = dij_matrix->dose_cube_size[0];
    size[1]  = dij_matrix->dose_cube_size[1];
    size[2]  = dij_matrix->dose_cube_size[2];

    FloatImageType::RegionType region;
    region.SetSize (size);
    region.SetIndex (start);
    image->SetRegions (region);

    image->Allocate ();

    return image;
}

void
clear_output_image (FloatImageType::Pointer img)
{
    typedef itk::ImageRegionIterator< FloatImageType > ImageIterator;
    ImageIterator it1 (img, img->GetBufferedRegion());
    it1.GoToBegin();
    while (!it1.IsAtEnd()) {
	it1.Set (0.0);
	++it1;
    }
}

/* Resample using trilinear interpolation */
void
update_output_image (FloatImageType::Pointer img,
		     ulong* discarded,
		     Dij_Matrix* dij_matrix,
		     float dose_wx, float dose_wy, float dose_wz, 
		     ushort value)
{
    int x, y, z;
    int x0 = (int) floor(dose_wx);
    int y0 = (int) floor(dose_wy);
    int z0 = (int) floor(dose_wz);
    float x0f = 1.0 - (dose_wx - floor(dose_wx));
    float y0f = 1.0 - (dose_wy - floor(dose_wy));
    float z0f = 1.0 - (dose_wz - floor(dose_wz));
    ushort vx[2], vy[2], vz[2];

    FloatImageType::IndexType idx;

    vx[0] = (ushort) (value * x0f + 0.5);
    vx[1] = value - vx[0];
    for (x = 0; x < 2; x++) {
	if (vx[x] == 0) continue;
	idx[0] = x0 + x;
	if (idx[0] < 0 || idx[0] >= dij_matrix->dose_cube_size[0]) {
	    *discarded += vx[x];
	    continue;
	}
	vy[0] = (ushort) (vx[x] * y0f + 0.5);
	vy[1] = vx[x] - vy[0];
        for (y = 0; y < 2; y++) {
	    if (vy[y] == 0) continue;
	    idx[1] = y0 + y;
	    if (idx[1] < 0 || idx[1] >= dij_matrix->dose_cube_size[1]) {
		*discarded += vy[y];
		continue;
	    }
	    vz[0] = (ushort) (vy[y] * z0f + 0.5);
	    vz[1] = vy[y] - vz[0];
	    for (z = 0; z < 2; z++) {
		if (vz[z] == 0) continue;
		idx[2] = z0 + z;
		if (idx[2] < 0 || idx[2] >= dij_matrix->dose_cube_size[2]) {
		    *discarded += vz[z];
		    continue;
		}
		img->SetPixel(idx, img->GetPixel(idx)+vz[z]);
	    }
	}
    }
}

void
read_pencil_beam (Pencil_Beam* pb, FILE* fp)
{
    int rc;
    rc = fread (&pb->energy, sizeof(float), 1, fp);
    if (rc != 1) dij_parse_error();
    rc = fread (&pb->spot_x, sizeof(float), 1, fp);
    if (rc != 1) dij_parse_error();
    rc = fread (&pb->spot_y, sizeof(float), 1, fp);
    if (rc != 1) dij_parse_error();
    rc = fread (&pb->nvox, sizeof(int), 1, fp);
    if (rc != 1) dij_parse_error();

    pb->vox = (ushort*) malloc (3*pb->nvox*sizeof(ushort));

    rc = fread (pb->vox, sizeof(ushort), 3*pb->nvox, fp);
    if (rc != 3*pb->nvox) dij_parse_error();
}

void
write_pencil_beam (Pencil_Beam* pb, FloatImageType::Pointer img, FILE* fp)
{
    int rc;
    int nvox = 0;
    long index = 0;
    long nvox_loc, eob_loc;

    rc = fwrite (&pb->energy, sizeof(float), 1, fp);
    if (rc != 1) dij_write_error();
    rc = fwrite (&pb->spot_x, sizeof(float), 1, fp);
    if (rc != 1) dij_write_error();
    rc = fwrite (&pb->spot_y, sizeof(float), 1, fp);
    if (rc != 1) dij_write_error();
    nvox_loc = ftell (fp);
    rc = fwrite (&pb->nvox, sizeof(int), 1, fp);
    if (rc != 1) dij_write_error();

    typedef itk::ImageRegionIterator< FloatImageType > ImageIterator;
    ImageIterator it1 (img, img->GetBufferedRegion());
    it1.GoToBegin();
    while (!it1.IsAtEnd()) {
	ushort vox[3];
	vox[2] = (ushort) it1.Get();
	if (vox[2] > 0) {
	    vox[0] = (ushort) index & 0xFF;
	    vox[1] = (ushort) index << 16;
	    rc = fwrite (vox, sizeof(ushort), 3, fp);
	    if (rc != 3) dij_write_error();
	    nvox++;
	}
	++index;
	++it1;
    }
    eob_loc = ftell (fp);
    fseek (fp, nvox_loc, SEEK_SET);
    rc = fwrite (&nvox, sizeof(int), 1, fp);
    if (rc != 1) dij_write_error();
    fseek (fp, eob_loc, SEEK_SET);
}

void
read_dij_header (Dij_Matrix* dij_matrix, FILE* fp)
{
    int rc;

    rc = fread (&dij_matrix->gantry_angle, sizeof(float), 1, fp);
    if (rc != 1) dij_parse_error();
    rc = fread (&dij_matrix->table_angle, sizeof(float), 1, fp);
    if (rc != 1) dij_parse_error();
    rc = fread (&dij_matrix->collimator_angle, sizeof(float), 1, fp);
    if (rc != 1) dij_parse_error();
    rc = fread (&dij_matrix->spot_spacing_dx, sizeof(float), 1, fp);
    if (rc != 1) dij_parse_error();
    rc = fread (&dij_matrix->spot_spacing_dy, sizeof(float), 1, fp);
    if (rc != 1) dij_parse_error();
    rc = fread (&dij_matrix->voxel_size_dx, sizeof(float), 1, fp);
    if (rc != 1) dij_parse_error();
    rc = fread (&dij_matrix->voxel_size_dy, sizeof(float), 1, fp);
    if (rc != 1) dij_parse_error();
    rc = fread (&dij_matrix->voxel_size_dz, sizeof(float), 1, fp);
    if (rc != 1) dij_parse_error();
    rc = fread (&dij_matrix->dose_cube_size, sizeof(int), 3, fp);
    if (rc != 3) dij_parse_error();
    rc = fread (&dij_matrix->num_pencil_beams, sizeof(int), 1, fp);
    if (rc != 1) dij_parse_error();
    rc = fread (&dij_matrix->absolute_dose_coefficient, sizeof(float), 1, fp);
    if (rc != 1) dij_parse_error();
}

void
write_dij_header (Dij_Matrix* dij_matrix, FILE* fp)
{
    int rc;

    rc = fwrite (&dij_matrix->gantry_angle, sizeof(float), 1, fp);
    if (rc != 1) dij_write_error();
    rc = fwrite (&dij_matrix->table_angle, sizeof(float), 1, fp);
    if (rc != 1) dij_write_error();
    rc = fwrite (&dij_matrix->collimator_angle, sizeof(float), 1, fp);
    if (rc != 1) dij_write_error();
    rc = fwrite (&dij_matrix->spot_spacing_dx, sizeof(float), 1, fp);
    if (rc != 1) dij_write_error();
    rc = fwrite (&dij_matrix->spot_spacing_dy, sizeof(float), 1, fp);
    if (rc != 1) dij_write_error();
    rc = fwrite (&dij_matrix->voxel_size_dx, sizeof(float), 1, fp);
    if (rc != 1) dij_write_error();
    rc = fwrite (&dij_matrix->voxel_size_dy, sizeof(float), 1, fp);
    if (rc != 1) dij_write_error();
    rc = fwrite (&dij_matrix->voxel_size_dz, sizeof(float), 1, fp);
    if (rc != 1) dij_write_error();
    rc = fwrite (&dij_matrix->dose_cube_size, sizeof(int), 3, fp);
    if (rc != 3) dij_write_error();
    rc = fwrite (&dij_matrix->num_pencil_beams, sizeof(int), 1, fp);
    if (rc != 1) dij_write_error();
    rc = fwrite (&dij_matrix->absolute_dose_coefficient, sizeof(float), 1, fp);
    if (rc != 1) dij_write_error();
}

void
warp_pencil_beam (DeformationFieldType::Pointer vf, 
		  FloatImageType::Pointer oimg, 
		  Dij_Matrix* dij_matrix, Ctatts* ctatts, 
		  Dif* dif, Pencil_Beam* pb)
{
    int i;
    float dose_offset_x, dose_offset_y, dose_offset_z;
    float vf_origin_x, vf_origin_y, vf_origin_z;
    ulong total = 0, discarded = 0;

    typedef itk::VectorLinearInterpolateImageFunction <
			DeformationFieldType, double > InterpolatorType;
    typedef InterpolatorType::PointType PointType;
    typedef InterpolatorType::OutputType OutputType;
    PointType pt;
    InterpolatorType::Pointer interpolator = InterpolatorType::New();
    interpolator->SetInputImage(vf);

    /* Compute shift from (pixel-centered) origin in original CT coordinates 
       to (pixel-centered) origin in konrad dose cube.  The (x,y,z) refer to 
       konrad coordinates, which conform to IEC. */
    dose_offset_x = ctatts->pos_isocenter_x - (dif->iso_index_dose_x * dif->delta_x)
			- (ctatts->pixel_size / 2.0);
    dose_offset_y = ctatts->pos_isocenter_y - (dif->iso_index_ct_y * dif->delta_y)
			- (ctatts->slice_distance / 2.0);
    dose_offset_z = ((ctatts->num_of_voxels_in_ct_cube_z + dif->iso_index_dose_z)
			* dif->delta_z) - ctatts->pos_isocenter_z
			- (ctatts->pixel_size / 2.0);

    /* Get origin location within vector field */
    const DeformationFieldType::PointType& ori = vf->GetOrigin();
    vf_origin_x = ori[0];
    vf_origin_y = ori[1];
    vf_origin_z = ori[2];

    /* Loop: For each voxel of this pencil beam */
    for (i = 0; i < pb->nvox; i++) {
	long dose_x, dose_y, dose_z, tmp;
	float ct_x, ct_y, ct_z;
	float dose_wx, dose_wy, dose_wz;
	long index = (pb->vox[i*3+1] << 16) + pb->vox[i*3];
	ushort value = pb->vox[i*3+2];

	/* Get coordinate of voxel within dose grid (in konrad-IEC coords) */
	dose_x = index % dif->dimension_dose_x;
	tmp = (index - dose_x) / dif->dimension_dose_x;
	dose_y = tmp % dif->dimension_dose_y;
	dose_z = (tmp - dose_y) / dif->dimension_dose_y;

	/* Compute voxel location in coordinate of original CT (mha-RAI coords) */
	ct_x = vf_origin_x + dose_offset_x + dose_x * dif->delta_x;
	ct_y = vf_origin_y + dose_offset_z - dose_z * dif->delta_z;
	ct_z = vf_origin_z + dose_offset_y + dose_y * dif->delta_y;

	/* Get deformation vector at that voxel location */
	pt[0] = ct_x; pt[1] = ct_y; pt[2] = ct_z;
	bool bvalue = interpolator->IsInsideBuffer (pt);
	if (!bvalue) {
	    printf ("Warning: beamlet dose (%g %g %g) outside vector field\n",
		    ct_x, ct_y, ct_z);
	    continue;
	}
	OutputType oval = interpolator->Evaluate (pt);

	/* Accumulate dose to output grid */
	dose_wx = dose_x + (oval[0] / dif->delta_x);
	dose_wy = dose_y + (oval[2] / dif->delta_y);
	dose_wz = dose_z - (oval[1] / dif->delta_z);
#if defined (commentout)
	printf ("%g %g %g -> %g %g %g\n", ct_x, ct_y, ct_z,
		oval[0], oval[1], oval[2]);
	printf ("%d %d %d -> %g %g %g\n", dose_x, dose_y, dose_z,
		dose_wx, dose_wy, dose_wz);
#endif
	total += value;
	update_output_image (oimg, &discarded, dij_matrix, 
			    dose_wx, dose_wy, dose_wz, value);
    }
    printf ("Discarded dose: %8lu/%8lu (%6.4f %%)\n", discarded, total, 
	    ((double)discarded/total));
}

void
convert_vector_field (
    DeformationFieldType::Pointer vf, 
    Ctatts* ctatts, 
    Dif* dif,
    const char* dij_in, 
    const char* dij_out)
{
    int i;
    FILE *fp_in, *fp_out;
    Dij_Matrix dij_matrix;
    Pencil_Beam pb;

    fp_in = fopen (dij_in, "rb");
    if (!fp_in) {
	fprintf (stderr, "Error opening dij file for read: %s\n", dij_in);
	exit (-1);
    }
    fp_out = fopen (dij_out, "wb");
    if (!fp_out) {
	fprintf (stderr, "Error opening dij file for write: %s\n", dij_out);
	exit (-1);
    }

    /* Load the header */
    read_dij_header (&dij_matrix, fp_in);
    printf ("Found %d pencil beams\n", dij_matrix.num_pencil_beams);
    write_dij_header (&dij_matrix, fp_out);

    /* Create a new image to hold the warped output */
    FloatImageType::Pointer oimg = make_output_image (&dij_matrix);

    /* For each pencil beam, load, warp, and write warped */
    for (i = 0; i < dij_matrix.num_pencil_beams; i++) {
	clear_output_image (oimg);
	read_pencil_beam (&pb, fp_in);
	warp_pencil_beam (vf, oimg, &dij_matrix, ctatts, dif, &pb);
	write_pencil_beam (&pb, oimg, fp_out);
	free (pb.vox);
    }

    /* Done! */
    fclose (fp_in);
    fclose (fp_out);
}

void
warp_dij_main (Warp_parms* parms)
{
    print_and_exit (
	"Warping of Dij matrices has been disabled due to lack of interest.\n"
	"Please contact plastimatch developers if you need this.\n");

#if defined (commentout)
    DeformationFieldType::Pointer vf = DeformationFieldType::New();
    Ctatts ctatts;
    Dif dif;

    printf ("Loading vector field...\n");
    vf = itk_image_load_float_field (parms->vf_in_fn);

    printf ("Loading ctatts and dif...\n");
    load_ctatts (&ctatts, (const char*) parms->ctatts_in_fn);
    load_dif (&dif, (const char*) parms->dif_in_fn);

    convert_vector_field (vf, &ctatts, &dif, parms->input_fn, 
	(const char*) parms->output_dij_fn);
#endif
}
