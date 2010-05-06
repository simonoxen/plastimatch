/* This program is used to convert from mha to an XiO dose file, given another XiO dose file as template */
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
int_endian (int *arg)
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
    FILE *ifp; FILE *ofp; FILE *ifp2;

    char buf[1024];

    char *result = NULL;
    size_t read_result;

    int i; int j; int k;
    double CMS_rx; double CMS_ry; double CMS_rz; double MHA_rx; double MHA_ry; double MHA_rz;
    double CMS_ox; double CMS_oy; double CMS_oz; double MHA_ox; double MHA_oy; double MHA_oz;
    int CMS_nPtsX; int CMS_nPtsY; int CMS_nPtsZ; int MHA_nPtsX; int MHA_nPtsY; int MHA_nPtsZ;
    double MHA_dx; double MHA_dy; double MHA_dz; // Element spacing
    double MHA_startX; double MHA_startY; double MHA_startZ; // Offset (top left corner of first slice)

    u8 header;
    u32 dose; float dose2;

    int o, p, q;
    u32 ***data;

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

    enum Xio_patient_position pt_position;

    long header_size;

    if (argc != 5) {
        printf ("Usage: mha2cms.c mhafile.mha newdosename dosetemplate patientposition");
        exit (0);
    }

    // Open input file
    ifp = fopen (argv[1], "rb");

    if (ifp == NULL) {
        fprintf (stderr, "ERROR: Cannot open input file.\n");
        exit (1);
    }

    // Patient position
    pt_position = xio_io_patient_position(argv[4]);

    if (pt_position == FFS || pt_position == FFP) {
	fprintf (stderr, "ERROR: Feet-first patient positions not yet implemented.\n");
	exit (1);
    }

    if (pt_position == UNKNOWN) {
	fprintf (stderr, "ERROR: Unknown patient position, should be (hfs|hfp|ffs|ffp).\n");
	exit (1);
    }

//  *** PARSE MHA HEADER ***

    printf ("\n*** MHA HEADER\n");

    for (i = 0; i < 12; i++) {
        fgets (buf, sizeof(buf), ifp);

        result = strstr (buf, "DimSize");
        if (result != NULL) {
            result = strtok (result, " ");
            result = strtok (NULL, " "); // Skip the stuff before the equal sign
            result = strtok (NULL, " ");
            MHA_nPtsX = atoi (result);
            result = strtok (NULL, " ");
            MHA_nPtsY = atoi (result);
            result = strtok (NULL, " ");
            MHA_nPtsZ = atoi (result);
            printf ("MHA nPts (x,y,z) are: %d%s%d%s%d\n", MHA_nPtsX, ",", MHA_nPtsY, ",", MHA_nPtsZ);
        }

        result = strstr (buf, "ElementSpacing");
        if (result != NULL) {
            result = strtok (result, " ");
            result = strtok (NULL, " "); // Skip the stuff before the equal sign
            result = strtok (NULL, " ");
            MHA_dx = atof (result);
            result = strtok (NULL, " ");
            MHA_dy = atof (result);
            result = strtok (NULL, " ");
            MHA_dz = atof (result);
            printf ("MHA_dimSpacings (x,y,z) are: %f%s%f%s%f\n", MHA_dx, ",", MHA_dy, ",", MHA_dz);
        }

        result = strstr (buf, "Offset");
        if (result != NULL) {
            result = strtok (result, " ");
            result = strtok (NULL, " "); //skip the stuff before the equal sign
            result = strtok (NULL, " ");
            MHA_startX = atof (result);
            result = strtok (NULL, " ");
            MHA_startY = atof (result);
            result = strtok (NULL, " ");
            MHA_startZ = atof (result);
            printf ("MHA_startCoords (x,y,z) is: %f%s%f%s%f\n", MHA_startX, ",", MHA_startY, ",", MHA_startZ);
        }
    }

    MHA_rx = MHA_dx * (MHA_nPtsX - 1); CMS_rx = MHA_rx;
    MHA_ry = MHA_dy * (MHA_nPtsY - 1); CMS_rz = MHA_ry;
    MHA_rz = MHA_dz * (MHA_nPtsZ - 1); CMS_ry = MHA_rz;
    printf ("MHA_ranges (x,y,z) are: %f%s%f%s%f\n", MHA_rx, ",", MHA_ry, ",", MHA_rz);
    printf ("CMS_ranges (x,y,z) are: %f%s%f%s%f\n", CMS_rx, ",", CMS_ry, ",", CMS_rz);

    MHA_ox = MHA_startX + MHA_rx / 2;
    MHA_oy = MHA_startY + MHA_ry / 2;
    MHA_oz = MHA_startZ + MHA_rz / 2;

    CMS_nPtsX = MHA_nPtsX; CMS_nPtsY = MHA_nPtsZ; CMS_nPtsZ = MHA_nPtsY;

    if (pt_position == HFS) {
	CMS_ox = MHA_ox; CMS_oy = MHA_oz; CMS_oz = -MHA_oy;
    } else if (pt_position == HFP) {
	CMS_ox = MHA_ox / 2; CMS_oy = MHA_oz; CMS_oz = MHA_oy;
    }

    printf ("MHA_offets (x,y,z) are: %f%s%f%s%f\n", MHA_ox, ",", MHA_oy, ",", MHA_oz);
    printf ("CMS_offsets (x,y,z) are: %f%s%f%s%f\n", CMS_ox, ",", CMS_oy, ",", CMS_oz);

//  *** PARSE DOSE TEMPLATE ***

    ifp2 = fopen (argv[3], "rb");
    if (!ifp2) {
        printf ("Error: Cannot open dose template for reading.\n");
        return 1;
    }

    printf ("\n*** DOSE TEMPLATE HEADER\n");

    // Line 1: XiO file format version
    fgets (buf, 1024, ifp2);
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
    fgets (buf, 1024, ifp2);

    // Line 2: Number of subplans or beams
    fgets (buf, 1024, ifp2);
    rc = sscanf (buf, "%1d", &xio_sources);

    if (rc != 1) {
        fprintf (stderr, "ERROR: Cannot parse sources/subplans: %s\n", buf);
        fclose (ifp);
        fclose (ifp2);
        exit (1);
    }

    printf ("Dose file is a sum of %d sources/subplans:\n", xio_sources);

    // One line for each source/subplan
    for (i = 1; i <= xio_sources; i++) {
        fgets (buf, 1024, ifp2);
        printf ("Source/subplan %d: %s", i, buf);
    }

    // Dose normalization info
    fgets (buf, 1024, ifp2);

    rc = sscanf (buf, "%lf,%lf", &xio_dose_scalefactor, &xio_dose_weight);

    if (rc != 2) {
        fprintf (stderr, "ERROR: Cannot parse dose normalization: %s\n", buf);
        exit (1);
    }

    printf ("Dose scale factor = %f\n", xio_dose_scalefactor);
    printf ("Dose weight = %f\n", xio_dose_weight);

    // Skipping line
    fgets (buf, 1024, ifp2);

    // Data type
    fgets (buf, 1024, ifp2);
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
    fgets (buf, 1024, ifp2);

    rc = sscanf (buf, "%d,%lf,%lf,%lf,%lf,%lf,%lf,%d,%d,%d",
                 &dummy, &rx, &rz, &ry, &ox, &oz, &oy, &nx, &nz, &ny);

    if (rc != 10) {
        fprintf (stderr, "ERROR: Cannot parse dose dose cube definition: %s\n", buf);
        exit (1);
    }

    printf ("rx = %lf, ry = %lf, rz = %lf\n", rx, ry, rz);
    printf ("ox = %lf, oy = %lf, oz = %lf\n", ox, oy, oz);
    printf ("nx = %d, ny = %d, nz = %d\n", nx, ny, nz);

    printf ("\n");

    // Skip rest of header
    fseek (ifp2, -nx * ny * nz * sizeof(dose), SEEK_END);
    header_size = ftell (ifp2);

//  *** READ MHA DOSE CUBE ***

    // Allocate memory for dose cube
    data = malloc (MHA_nPtsX * sizeof(float**));
    for (o = 0; o < MHA_nPtsX; o++) {
        data[o] = malloc (MHA_nPtsZ * sizeof(float*));
        for (p = 0; p < MHA_nPtsZ; p++) {
            data[o][p] = malloc (MHA_nPtsY * sizeof(float));
            for (q = 0; q < MHA_nPtsY; q++) {
                data[o][p][q] = 0;
            }
        }
    }

    // Read dose cube

    printf ("Reading dose cube...");

    fseek (ifp, -MHA_nPtsX * MHA_nPtsY * MHA_nPtsZ * sizeof(dose2), SEEK_END);

    for (k = 0; k < MHA_nPtsZ; k++) {
        for (j = 0; j < MHA_nPtsY; j++) {
            for (i = 0; i < MHA_nPtsX; i++) {
                read_result = fread (&dose2, sizeof(dose2), 1, ifp);
                if (read_result != 1) {
                    printf ("FAILED.\n");
                    fprintf (stderr, "ERROR: Cannot read dose cube.\n");
                    exit (1);
                }
                dose = dose2 / xio_dose_weight / xio_dose_scalefactor;
                int_endian (&dose); // mha is little endian, XiO is big endian
                data[i][k][j] = dose;
            }
        }
    }

    printf ("Done.\n");

    fclose (ifp);

//  *** WRITE XIO HEADER FROM DOSE TEMPLATE ***

    ofp = fopen (argv[2], "wb");
    if (ofp == NULL) {
        printf ("ERROR: Cannot create output file for writing.\n");
        return 1;
    }

    printf ("Writing XiO header from dose template, size is %ld...", header_size);

    fseek (ifp2, 0, SEEK_SET);

    for (i = 0; i < header_size; i++) {
        read_result = fread (&header, sizeof(header), 1, ifp2);
        if (read_result != 1) {
            printf ("FAILED.\n");
            fprintf (stderr, "ERROR: Cannot read dose template header.\n");
            exit (1);
        }
        fwrite (&header, sizeof(header), 1, ofp);
    }

    printf ("Done.\n");

//  *** WRITE DOSE CUBE FROM MHA FILE ***

    printf ("Writing dose cube...");

    if (pt_position == HFS) {
	for (k = 0; k < MHA_nPtsZ; k++) {
	    for (j = MHA_nPtsY - 1; j >= 0; j--) { // Going through the CMS Z
		for (i = 0; i < MHA_nPtsX; i++) {
		    fwrite (&(data[i][k][j]), sizeof(u32), 1, ofp);
		}	    
	    }
	}
    } else if (pt_position == HFP) {
	for (k = 0; k < MHA_nPtsZ; k++) {
	    for (j = 0; j < MHA_nPtsY; j++) { // Going through the CMS Z
		for (i = MHA_nPtsX - 1; i >= 0; i--) {
		    fwrite (&(data[i][k][j]), sizeof(u32), 1, ofp);
		}	    
	    }
	}
    }

    printf ("Done.\n");

    fclose (ofp);
    fclose (ifp2);

    return 0;
}
