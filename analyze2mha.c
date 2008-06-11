/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* code to convert a pair of Analyze IMG/HDR files to a MHA */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "plm_config.h"
#include "dbh.h"

char header_pat[] = 
    "ObjectType = Image\n"
    "NDims = 3\n"
    "BinaryData = True\n"
    "BinaryDataByteOrderMSB = False\n"
    "Offset = 0. 0. 0.\n"
    "ElementSpacing = %f %f %f\n"
    "DimSize = %d %d %d\n"
	"AnatomicalOrientation = RPS\n"
    "ElementType = MET_SHORT\n"
    "ElementDataFile = LOCAL\n"
    ;

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

	hdr = (struct dsr *) calloc(1, sizeof(struct dsr));

	if (argc!=3) {
		printf("Illegal usage of the program!\n");
		printf("Usage: %s input_Analyze_filename_wo_extension output_MHA_filename.\n", argv[0]);
		return -1;
	}

    printf ("Loading and converting ...\n");

	strcpy(buf, argv[1]);
	strcat(buf, ".hdr");
    fp = fopen (buf, "rb");
    if (!fp) { 
		printf ("Could not open the Analyze header file %s for read!\n", buf);
		return -1;
    }
	fread(hdr, sizeof(struct dsr), 1, fp);
	fclose(fp);

	strcpy(buf, argv[1]);
	strcat(buf, ".img");
    fp = fopen (buf, "rb");
    if (!fp) { 
		printf ("Could not open the Analyze image file %s for read!\n", buf);
		return -1;
    }

    if (!(fp1 = fopen(argv[2],"wb"))) {
		printf ("Error opening MHA file %s for write\n", argv[2]);
		return -1;
    }

    fprintf (fp1, header_pat, hdr->dime.pixdim[1], hdr->dime.pixdim[2], hdr->dime.pixdim[3],
		hdr->dime.dim[1], hdr->dime.dim[2], hdr->dime.dim[3]);

	if (hdr->dime.datatype = DT_SIGNED_SHORT) {
		while (!feof(fp)) {
			short s;
			fread(&s,2,1,fp);
			fwrite(&s,2,1,fp1);
		}
	} else if (hdr->dime.datatype = DT_UNSIGNED_CHAR) {
		printf("The Analyze file has data type UNSIGNED CHAR. A conversion to SHORT is being done.\n");
		while (!feof(fp)) {
			short s;
			unsigned c;
			fread(&c,1,1,fp);
			s = c;
			fwrite(&s,2,1,fp1);
		}
	} else {
		printf("The Analyze file does not have data type SHORT or UNSIGNED CHAR! Is it really an image file?\n");
		return -2;
	}

    fclose (fp);
    fclose (fp1);
    return 0;
}
