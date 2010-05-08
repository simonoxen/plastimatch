/* This program is used to convert from XiO dose files to mha */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "plm_config.h"
#include "plm_int.h"

#define XIO_VERSION_450       1
#define XIO_VERSION_421       2
#define XIO_VERSION_UNKNOWN   3

#define XIO_DATATYPE_UINT32     5

enum Xio_patient_position {
    UNKNOWN,
    HFS,
    HFP,
    FFS,
    FFP,
};

enum Xio_patient_position
xio_io_patient_position (
    const char *pt_position_str
)
{
    // Convert string to patient position
    if (!strcmp (pt_position_str, "hfs")) {
        return HFS;
    } else if (!strcmp (pt_position_str, "hfp")) {
        return HFP;
    } else if (!strcmp (pt_position_str, "ffs")) {
        return FFS;
    } else if (!strcmp (pt_position_str, "ffp")) {
        return FFP;
    } else {
	return UNKNOWN;
    }
}

// Swap endianness of 32-bit integer
void
int_endian (uint32_t *arg)
{
    char lenbuf[4];
    char tmpc;
    memcpy (lenbuf, (const char *) arg, 4);
    tmpc = lenbuf[0]; lenbuf[0] = lenbuf[3]; lenbuf[3] = tmpc;
    tmpc = lenbuf[1]; lenbuf[1] = lenbuf[2]; lenbuf[2] = tmpc;
    memcpy ((char *) arg, lenbuf, 4);
}

int
main (int argc, char *argv[])
{
    FILE *ifp; FILE *ofp;

    char buf[1024];

    size_t read_result;

    int i, j, k;
    int o, p, q;

    float ***data;

    uint32_t dose;
    float normalized_dose;

    int rc;
    int dummy;

    // XiO header info
    int xio_version;
    int xio_sources;
    double xio_dose_scalefactor, xio_dose_weight;
    int xio_datatype;
    // Dimensions
    double rx; double ry; double rz;
    double ox; double oy; double oz;
    int nx; int ny; int nz;
    // Element spacing
    double dx; double dy; double dz;
    // Offset (top left corner of first slice)
    double topx; double topy; double topz;

    enum Xio_patient_position pt_position;

    if (argc != 4) {
        printf ("Usage: cms_dose_to_mha cmsdose outputfile.mha patientposition\n");
        exit (0);
    }

    // Open input file
    ifp = fopen (argv[1], "rb");
    if (ifp == NULL) {
        fprintf (stderr, "ERROR: Cannot open input file.\n");
        exit (1);
    }

    // Patient position
    pt_position = xio_io_patient_position(argv[3]);

    if (pt_position == FFS || pt_position == FFP) {
	fprintf (stderr, "ERROR: Feet-first patient positions not yet implemented.\n");
	exit (1);
    }

    if (pt_position == UNKNOWN) {
	fprintf (stderr, "ERROR: Unknown patient position, should be (hfs|hfp|ffs|ffp).\n");
	exit (1);
    }

    // Line 1: XiO file format version
    fgets (buf, sizeof(buf), ifp);
    if (!strncmp (buf, "006d101e", strlen ("006d101e"))) {
        xio_version = XIO_VERSION_450;
    } else if (!strncmp (buf, "004f101e", strlen ("004f101e"))) {
        xio_version = XIO_VERSION_421;
    } else {
        xio_version = XIO_VERSION_UNKNOWN;
    }

    if (xio_version != XIO_VERSION_450
        && xio_version != XIO_VERSION_421) {
        printf ("WARNING: Unknown XiO file format version: %s\n", buf);
    }

    // Skipping line
    fgets (buf, sizeof(buf), ifp);

    // Line 2: Number of subplans or beams
    fgets (buf, sizeof(buf), ifp);
    rc = sscanf (buf, "%1d", &xio_sources);

    if (rc != 1) {
        fprintf (stderr, "ERROR: Cannot parse sources/subplans: %s\n", buf);
        exit (1);
    }

    printf ("Dose file is a sum of %d sources/subplans:\n", xio_sources);

    // One line for each source/subplan
    for (i = 1; i <= xio_sources; i++) {
        fgets (buf, sizeof(buf), ifp);
        printf ("Source/subplan %d: %s", i, buf);
    }

    // Dose normalization info
    fgets (buf, sizeof(buf), ifp);

    rc = sscanf (buf, "%lf,%lf", &xio_dose_scalefactor, &xio_dose_weight);

    if (rc != 2) {
        fprintf (stderr, "ERROR: Cannot parse dose normalization: %s\n", buf);
        exit (1);
    }

    printf ("Dose scale factor = %f\n", xio_dose_scalefactor);
    printf ("Dose weight = %f\n", xio_dose_weight);

    // Skipping line
    fgets (buf, sizeof(buf), ifp);

    // Data type
    fgets (buf, sizeof(buf), ifp);
    rc = sscanf (buf, "%1d", &xio_datatype);

    if (rc != 1) {
        fprintf (stderr, "ERROR: Cannot parse datatype: %s\n", buf);
        exit (1);
    }

    if (xio_datatype != XIO_DATATYPE_UINT32) {
        fprintf (stderr, "ERROR: Only unsigned 32-bit integer data is currently supported: %s\n", buf);
        exit (1);
    }

    // Dose cube definition
    fgets (buf, sizeof(buf), ifp);

    rc = sscanf (buf, "%d,%lf,%lf,%lf,%lf,%lf,%lf,%d,%d,%d",
                 &dummy, &rx, &rz, &ry, &ox, &oz, &oy, &nx, &nz, &ny);

    if (rc != 10) {
        fprintf (stderr, "ERROR: Cannot parse dose dose cube definition: %s\n", buf);
        exit (1);
    }

    printf ("rx = %lf, ry = %lf, rz = %lf\n", rx, ry, rz);
    printf ("ox = %lf, oy = %lf, oz = %lf\n", ox, oy, oz);
    printf ("nx = %d, ny = %d, nz = %d\n", nx, ny, nz);

    // Calculate element spacing
    dx = rx / (nx - 1);
    dy = ry / (ny - 1);
    dz = rz / (nz - 1);

    // Calculate offset
    if (pt_position == HFS) {
	topx = ox - (rx / 2);
	topy = oy - (ry / 2);
	topz = -oz - (rz / 2);
    } else if (pt_position == HFP) {
	topx = ox * 2 - (rx / 2);
	topy = oy - (ry / 2);
	topz = oz - (rz / 2);
    }

    // Skip rest of header
    fseek (ifp, -nx * ny * nz * sizeof(dose), SEEK_END);

    // Allocate memory for dose cube
    data = malloc (nz * sizeof(float**));
    for (o = 0; o < nz; o++) {
        data[o] = malloc (ny * sizeof(float*));
        for (p = 0; p < ny; p++) {
            data[o][p] = malloc (nx * sizeof(float));
            for (q = 0; q < nx; q++) {
                data[o][p][q] = 0;
            }
        }
    }

    // Read dose from XiO
    printf ("Reading dose cube at offset %ld...", ftell (ifp));

    for (j = 0; j < ny; j++) {
        for (k = 0; k < nz; k++) {
            for (i = 0; i < nx; i++) {
                // Read one point in dose cube
                read_result = fread (&dose, sizeof(dose), 1, ifp);
                if (read_result != 1) {
                    printf ("FAILED.\n");
                    fprintf (stderr, "ERROR: Cannot read dose cube.\n");
                    exit (1);
                }

                // XiO is big endian. Swap to little endian if neccesarry.
                if (!CMAKE_WORDS_BIGENDIAN) int_endian (&dose);

                // Normalize and save floating point
                normalized_dose = dose * xio_dose_weight * xio_dose_scalefactor;
                data[k][j][i] = normalized_dose;
            }
        }
    }

    printf ("Done.\n");
    printf ("Writing dose cube...");

    fclose (ifp);

    // Open output file
    ofp = fopen (argv[2], "wb");
    if (ofp == NULL) {
        printf ("FAILED.\n");
        fprintf (stderr, "ERROR: Cannot create output file for writing.\n");
        exit (-1);
    }

    // Write mha header
    fprintf (ofp, "ObjectType = Image\n");
    fprintf (ofp, "NDims = 3\n");
    fprintf (ofp, "BinaryData = True\n");
    fprintf (ofp, "BinaryDataByteOrderMSB = False\n");
    fprintf (ofp, "Offset = %f %f %f\n", topx, topz, topy);
    fprintf (ofp, "ElementSpacing = %f %f %f\n", dx, dz, dy);
    fprintf (ofp, "DimSize = %d %d %d\n", nx, nz, ny);

    fprintf (ofp, "AnatomicalOrientation = RAI\n");
    fprintf (ofp, "TransformMatrix = 1 0 0 0 1 0 0 0 1\n");
    fprintf (ofp, "CenterOfRotation = 0 0 0\n");
    fprintf (ofp, "ElementType = MET_FLOAT\n");
    fprintf (ofp, "ElementDataFile = LOCAL\n");

    // Write dose cube
    if (pt_position == HFS) {
	for (j = 0; j < ny; j++) {
	    for (k = nz - 1; k >= 0; k--) {
		for (i = 0; i < nx; i++) {
		    fwrite (&(data[k][j][i]), sizeof(float), 1, ofp);
		}
	    }
	}
    } else if (pt_position == HFP) {
	for (j = 0; j < ny; j++) {
	    for (k = 0; k < nz; k++) {
		for (i = nx - 1; i >= 0; i--) {
		    fwrite (&(data[k][j][i]), sizeof(float), 1, ofp);
		}
	    }
	}
    }

    fclose (ofp);

    printf ("Done.\n");

    return 0;
}
