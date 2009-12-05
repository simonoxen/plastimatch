/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <cstdio>
#include <string>
#include <memory>
#include <vector>
#include "file_util.h"
#include "hnd_io.h"
#include "plm_int.h"
#include "print_and_exit.h"
#include "timer.h"

using namespace std;

#define ROWS 768
#define COLS 1024

//#define DSR 1
#define DSR 2

void
hnd_load (Proj_image *proj, char *fn)
{
    FILE *fp;
    fpos_t pstart;
    int i;

    printf ("Looking for %s\n", fn);

    if (!file_exists (fn)) {
        print_and_exit ("Error: hnd_file (%s) does not exist.\n", fn);
    }

    fp = fopen (fn, "rb");
    if (fp == NULL) {
        print_and_exit ("Cannot open %s for read\n", fn);
    }

    fgetpos (fp, &pstart);
    unsigned char *buffer = (unsigned char *) malloc (10000000);
    if (buffer == NULL) {
        print_and_exit ("malloc error");
    }
    int TFileLength = fread (buffer, 1, 10000000, fp);
    free (buffer);
    fsetpos (fp, &pstart);

    Hnd_header hnd;

    fread (( void *) hnd.sFileType, sizeof(char), 32, fp);
    fread (( void *) &hnd.FileLength, sizeof(uint32_t), 1, fp);
    fread (( void *) hnd.sChecksumSpec, sizeof(char), 4, fp);
    fread (( void *) &hnd.nCheckSum, sizeof(uint32_t), 1, fp);
    fread (( void *) hnd.sCreationDate, sizeof(char), 8, fp);
    fread (( void *) hnd.sCreationTime, sizeof(char), 8, fp);
    fread (( void *) hnd.sPatientID, sizeof(char), 16, fp);
    fread (( void *) &hnd.nPatientSer, sizeof(uint32_t), 1, fp);
    fread (( void *) hnd.sSeriesID, sizeof(char), 16, fp);
    fread (( void *) &hnd.nSeriesSer, sizeof(uint32_t), 1, fp);
    fread (( void *) hnd.sSliceID, sizeof(char), 16, fp);
    fread (( void *) &hnd.nSliceSer, sizeof(uint32_t), 1, fp);
    fread (( void *) &hnd.SizeX, sizeof(uint32_t), 1, fp);
    fread (( void *) &hnd.SizeY, sizeof(uint32_t), 1, fp);
    fread (( void *) &hnd.dSliceZPos, sizeof(double), 1, fp);
    fread (( void *) hnd.sModality, sizeof(char), 16, fp);
    fread (( void *) &hnd.nWindow, sizeof(uint32_t), 1, fp);
    fread (( void *) &hnd.nLevel, sizeof(uint32_t), 1, fp);
    fread (( void *) &hnd.nPixelOffset, sizeof(uint32_t), 1, fp);
    fread (( void *) hnd.sImageType, sizeof(char), 4, fp);
    fread (( void *) &hnd.dGantryRtn, sizeof(double), 1, fp);
    fread (( void *) &hnd.dSAD, sizeof(double), 1, fp);
    fread (( void *) &hnd.dSFD, sizeof(double), 1, fp);
    fread (( void *) &hnd.dCollX1, sizeof(double), 1, fp);
    fread (( void *) &hnd.dCollX2, sizeof(double), 1, fp);
    fread (( void *) &hnd.dCollY1, sizeof(double), 1, fp);
    fread (( void *) &hnd.dCollY2, sizeof(double), 1, fp);
    fread (( void *) &hnd.dCollRtn, sizeof(double), 1, fp);
    fread (( void *) &hnd.dFieldX, sizeof(double), 1, fp);
    fread (( void *) &hnd.dFieldY, sizeof(double), 1, fp);
    fread (( void *) &hnd.dBladeX1, sizeof(double), 1, fp);
    fread (( void *) &hnd.dBladeX2, sizeof(double), 1, fp);
    fread (( void *) &hnd.dBladeY1, sizeof(double), 1, fp);
    fread (( void *) &hnd.dBladeY2, sizeof(double), 1, fp);
    fread (( void *) &hnd.dIDUPosLng, sizeof(double), 1, fp);
    fread (( void *) &hnd.dIDUPosLat, sizeof(double), 1, fp);
    fread (( void *) &hnd.dIDUPosVrt, sizeof(double), 1, fp);
    fread (( void *) &hnd.dIDUPosRtn, sizeof(double), 1, fp);
    fread (( void *) &hnd.dPatientSupportAngle, sizeof(double), 1, fp);
    fread (( void *) &hnd.dTableTopEccentricAngle, sizeof(double), 1, fp);
    fread (( void *) &hnd.dCouchVrt, sizeof(double), 1, fp);
    fread (( void *) &hnd.dCouchLng, sizeof(double), 1, fp);
    fread (( void *) &hnd.dCouchLat, sizeof(double), 1, fp);
    fread (( void *) &hnd.dIDUResolutionX, sizeof(double), 1, fp);
    fread (( void *) &hnd.dIDUResolutionY, sizeof(double), 1, fp);
    fread (( void *) &hnd.dImageResolutionX, sizeof(double), 1, fp);
    fread (( void *) &hnd.dImageResolutionY, sizeof(double), 1, fp);
    fread (( void *) &hnd.dEnergy, sizeof(double), 1, fp);
    fread (( void *) &hnd.dDoseRate, sizeof(double), 1, fp);
    fread (( void *) &hnd.dXRayKV, sizeof(double), 1, fp);
    fread (( void *) &hnd.dXRayMA, sizeof(double), 1, fp);
    fread (( void *) &hnd.dMetersetExposure, sizeof(double), 1, fp);
    fread (( void *) &hnd.dAcqAdjustment, sizeof(double), 1, fp);
    fread (( void *) &hnd.dCTProjectionAngle, sizeof(double), 1, fp);
    fread (( void *) &hnd.dCTNormChamber, sizeof(double), 1, fp);
    fread (( void *) &hnd.dGatingTimeTag, sizeof(double), 1, fp);
    fread (( void *) &hnd.dGating4DInfoX, sizeof(double), 1, fp);
    fread (( void *) &hnd.dGating4DInfoY, sizeof(double), 1, fp);
    fread (( void *) &hnd.dGating4DInfoZ, sizeof(double), 1, fp);
    fread (( void *) &hnd.dGating4DInfoTime, sizeof(double), 1, fp);

    unsigned char *SkipHead = (unsigned char *) 
	malloc (sizeof(unsigned char) * 1024);

    int LookupTableSize = (int)((float) hnd.SizeX 
	* ((float) hnd.SizeY - 1.0) / 4.0 + 0.5);

    unsigned char *LookupTable = (unsigned char *) 
	malloc (sizeof(unsigned char) * LookupTableSize);

    int CDataLength = (TFileLength - 1024 - LookupTableSize);
    unsigned char *CD = (unsigned char *) 
	malloc (sizeof(unsigned char) * CDataLength);

    int *LookupType = (int *) 
	malloc (sizeof(int) * (hnd.SizeX * hnd.SizeY + 4));
    long *Img = (long *) malloc (sizeof(long) * hnd.SizeX * hnd.SizeY);
#if defined (commentout)
    unsigned short *uImg = (unsigned short*) 
	malloc (sizeof(unsigned short) * hnd.SizeX * hnd.SizeY / DSR / DSR);
#endif

    if (!SkipHead || !LookupTable || !CD || !LookupType || !Img) {
        print_and_exit ("Malloc error\n");
    }

    /* GCS FIX: Is this right?  gcc gives me warning about precedence. 
       In C language, operators are grouped like this:
       FileLength - hFileLength << 16   ->   (FileLength - hFileLength) << 16
       Anyway, these are not used...
    */
#if defined (commentout)
    int hFileLength = FileLength >> 16;
    int lFileLength = FileLength - hFileLength << 16;
    int fFileLength = lFileLength << 16 + hFileLength;

    printf ("Setting position...\n");

    fsetpos (fp, &pstart);
    fread (SkipHead, 1, 1024, fp);

    int ActLook = fread (LookupTable, 1, LookupTableSize, fp);
    int k = hnd.SizeX + 1;

    for (int i = 0; i < LookupTableSize; i++) {
        int sLook0 = 3 & (LookupTable[i] >> (0 * 2));
        int sLook1 = 3 & (LookupTable[i] >> (1 * 2));
        int sLook2 = 3 & (LookupTable[i] >> (2 * 2));
        int sLook3 = 3 & (LookupTable[i] >> (3 * 2));
        LookupType[i * 4 + hnd.SizeX + 1] = sLook0;
        LookupType[i * 4 + hnd.SizeX + 1 + 1] = sLook1;
        LookupType[i * 4 + hnd.SizeX + 1 + 2] = sLook2;
        LookupType[i * 4 + hnd.SizeX + 1 + 3] = sLook3;
    }
    int ActCDLength = fread (CD, 1, CDataLength, fp);
#endif

    //fclose (fp);

    /* ------------------------------------------------------------- */
    uint32_t* buf;
    unsigned char pt_lut[1024*768];
    uint32_t a;
    float b;
    unsigned char v;
    int lut_idx, lut_off;
    size_t num_read;
    char dc;
    short ds;
    long dl, diff;

    buf = (uint32_t*) malloc (sizeof(uint32_t) * ROWS * COLS);

    /* Read LUT */
    fseek (fp, 1024, SEEK_SET);
    fread (pt_lut, sizeof(unsigned char), (ROWS-1)*COLS / 4, fp);

    /* Read first row */
    for (i = 0; i < COLS; i++) {
	fread (&a, sizeof(uint32_t), 1, fp);
	buf[i] = a;
	b = a;
    }

    /* Read first pixel of second row */
    fread (&a, sizeof(uint32_t), 1, fp);
    buf[i++] = a;
    b = a;
    
    /* Decompress the rest */
    lut_idx = 0;
    lut_off = 0;
    while (i < COLS*ROWS) {
	uint32_t r11, r12, r21;

	r11 = buf[i-COLS-1];
	r12 = buf[i-COLS];
	r21 = buf[i-1];
	v = pt_lut[lut_idx];
	switch (lut_off) {
	case 0:
	    v = v & 0x03;
	    lut_off ++;
	    break;
	case 1:
	    v = (v & 0x0C) >> 2;
	    lut_off ++;
	    break;
	case 2:
	    v = (v & 0x30) >> 4;
	    lut_off ++;
	    break;
	case 3:
	    v = (v & 0xC0) >> 6;
	    lut_off = 0;
	    lut_idx ++;
	    break;
	}
	switch (v) {
	case 0:
	    num_read = fread (&dc, sizeof(unsigned char), 1, fp);
	    if (num_read != 1) goto read_error;
	    diff = dc;
	    break;
	case 1:
	    num_read = fread (&ds, sizeof(unsigned short), 1, fp);
	    if (num_read != 1) goto read_error;
	    diff = ds;
	    break;
	case 2:
	    num_read = fread (&dl, sizeof(uint32_t), 1, fp);
	    if (num_read != 1) goto read_error;
	    diff = dl;
	    break;
	}

	buf[i] = r21 + r12 + diff - r11;
	b = buf[i];
	//if (buf[i] > 65535) b = 65535;
	i++;
    }

    free (buf);
    fclose (fp);
    //return 0;
    return;

 read_error:

    fprintf (stderr, "Error reading hnd file\n");
    free (buf);
    fclose (fp);
    //return -1;
}
