/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* code to convert a MHA fiel to an Analyze IMG/HDR set */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config.h"
#include "dbh.h"

void swap_long(unsigned char *pntr)
{
	unsigned char b0, b1, b2, b3;

	b0 = *pntr;
	b1 = *(pntr+1);
	b2 = *(pntr+2);
	b3 = *(pntr+3);
	
	*pntr = b3;
	*(pntr+1) = b2;
	*(pntr+2) = b1;
	*(pntr+3) = b0;
}
        
void swap_short(unsigned char *pntr)
{
	unsigned char b0, b1;
	
	b0 = *pntr;
	b1 = *(pntr+1);
	
	*pntr = b1;
	*(pntr+1) = b0;
}


void swap_hdr(struct dsr *pntr)
{
	swap_long((unsigned char *) &pntr->hk.sizeof_hdr) ;
	swap_long((unsigned char *) &pntr->hk.extents) ;
	swap_short((unsigned char *) &pntr->hk.session_error) ;
	swap_short((unsigned char *) &pntr->dime.dim[0]) ;
	swap_short((unsigned char *) &pntr->dime.dim[1]) ;
	swap_short((unsigned char *) &pntr->dime.dim[2]) ;
	swap_short((unsigned char *) &pntr->dime.dim[3]) ;
	swap_short((unsigned char *) &pntr->dime.dim[4]) ;
	swap_short((unsigned char *) &pntr->dime.dim[5]) ;
	swap_short((unsigned char *) &pntr->dime.dim[6]) ;
	swap_short((unsigned char *) &pntr->dime.dim[7]) ;
	swap_short((unsigned char *) &pntr->dime.datatype) ;
	swap_short((unsigned char *) &pntr->dime.bitpix) ;
	swap_long((unsigned char *) &pntr->dime.pixdim[0]) ;
	swap_long((unsigned char *) &pntr->dime.pixdim[1]) ;
	swap_long((unsigned char *) &pntr->dime.pixdim[2]) ;
	swap_long((unsigned char *) &pntr->dime.pixdim[3]) ;
	swap_long((unsigned char *) &pntr->dime.pixdim[4]) ;
	swap_long((unsigned char *) &pntr->dime.pixdim[5]) ;
	swap_long((unsigned char *) &pntr->dime.pixdim[6]) ;
	swap_long((unsigned char *) &pntr->dime.pixdim[7]) ;
	swap_long((unsigned char *) &pntr->dime.vox_offset) ;
	swap_long((unsigned char *) &pntr->dime.funused1) ;
	swap_long((unsigned char *) &pntr->dime.funused2) ;
	swap_long((unsigned char *) &pntr->dime.cal_max) ;
	swap_long((unsigned char *) &pntr->dime.cal_min) ;
	swap_long((unsigned char *) &pntr->dime.compressed) ;
	swap_long((unsigned char *) &pntr->dime.verified) ;
	swap_short((unsigned char *) &pntr->dime.dim_un0) ;
	swap_long((unsigned char *) &pntr->dime.glmax) ;
	swap_long((unsigned char *) &pntr->dime.glmin) ;
}
       

int main(int argc, char *argv[])
{
    struct dsr *hdr;
	char buf[1024];
	FILE *fp, *fp1;
	int i, si[10];
	long bytesz;
	unsigned char c;

	hdr = (struct dsr *) calloc(1, sizeof(struct dsr));

	if (argc!=3) {
		printf("Illegal usage of the program!\n");
		printf("Usage: %s input_MHA_filename output_Analyze_filename_wo_extension.\n", argv[0]);
		return -1;
	}

    printf ("Loading...\n");

    fp = fopen (argv[1], "rb");
    if (!fp) { 
		printf ("Could not open mha file for read\n");
		return -1;
    }
	hdr->hk.sizeof_hdr = 348;
	hdr->hk.regular = 'r';
	hdr->dime.dim[0] = 4;
	hdr->dime.dim[4] = 1; // maybe used for vector field?
	hdr->dime.bitpix = 8;
	hdr->dime.vox_offset = 0;
	hdr->dime.pixdim[0] = 1.;
	for (i=0; i<30; i++) { // 30 lines of MHA should have passed the header 
		fgets(buf,1024,fp);
		if (strstr(buf, "DimSize")!=NULL) {
			sscanf(&(buf[9]), "%d%d%d", &(si[1]), &(si[2]), &(si[3]));
			hdr->dime.dim[1] = (short int) si[1];
			hdr->dime.dim[2] = (short int) si[2];
			hdr->dime.dim[3] = (short int) si[3];
		} else if (strstr(buf, "ElementSpacing")!=NULL) {
			sscanf(&(buf[16]), "%f%f%f", &(hdr->dime.pixdim[1]), &(hdr->dime.pixdim[2]), &(hdr->dime.pixdim[3]));
		} else if (strstr(buf, "ElementType")!=NULL) {
			if (!strcmp(buf, "ElementType = MET_SHORT\n")) {
				hdr->dime.bitpix = 16;
				hdr->dime.datatype = DT_SIGNED_SHORT;
				hdr->dime.glmax = 32767;
				hdr->dime.glmin = -32768;
			} else if (!strcmp(buf, "ElementType = MET_USHORT\n")) {
				hdr->dime.bitpix = 16;
				hdr->dime.datatype = DT_SIGNED_SHORT; // seems Analyze doesn't have unsigned short
				hdr->dime.glmax = 65535;
				hdr->dime.glmin = 0;
			} else if (!strcmp(buf, "ElementType = MET_UCHAR\n")) {
				hdr->dime.bitpix = 8;
				hdr->dime.datatype = DT_UNSIGNED_CHAR;
				hdr->dime.glmax = 255;
				hdr->dime.glmin = 0;
			} else if (!strcmp(buf, "ElementType = MET_FLOAT\n")) {
				hdr->dime.bitpix = 32;
				hdr->dime.datatype = DT_FLOAT;
				hdr->dime.glmax = 32767;
				hdr->dime.glmin = -32768;
			}
		} else if (strstr(buf, "ElementNumberOfChannels")!=NULL) {
			sscanf(&(buf[26]), "%d", &(si[4]));
			hdr->dime.dim[4] = (short int) si[4];
			if (hdr->dime.dim[4]!=1) {
				printf("The file has multiple volumes!\n It could be a vector field file, but the ordering maybe wrong. Double check!!\n");
			}
		} else if (strstr(buf, "Offset")!=NULL) {
		}
	}

	printf("Writing...\n");
	//header
	strcpy(buf, argv[2]);
	strcat(buf, ".hdr");
	fp1 = fopen(buf, "wb");
	fwrite(hdr, sizeof(struct dsr), 1, fp1);
	fclose(fp1);

	//image
	strcpy(buf, argv[2]);
	strcat(buf, ".img");
	fp1 = fopen(buf, "wb");

	bytesz = hdr->dime.dim[1] * hdr->dime.dim[2] * hdr->dime.dim[3] * hdr->dime.dim[4] * hdr->dime.bitpix/8;
	printf("%ld bytes of image data!\n", bytesz);
	fseek(fp, -bytesz, SEEK_END);

    while (!feof(fp)) {
		fread(&c,1,1,fp);
		fwrite(&c,1,1,fp1);
    }

	fclose(fp);
	fclose(fp1);
	return 0;
}
