#include <stdio.h>

#define ROWS 768
#define COLS 1024

int 
main (int argc, char* argv[]) 
{
    FILE *fp_hnd, *fp_pfm;
    unsigned char pt_lut[1024*768];
    unsigned long* buf;
    int i, lut_idx, lut_off;
    char dc;
    short ds;
    long dl, diff;
    unsigned long a;
    float b;
    unsigned char v;

    if (argc != 3) {
	printf ("Usage: hndtopfm hndfile pfmfile\n");
	return 1;
    }

    buf = (unsigned long*) malloc (sizeof(unsigned long) * ROWS * COLS);

    fp_hnd = fopen (argv[1], "rb");
    if (!fp_hnd) {
	printf ("Error, cannot open file %s for read\n", argv[1]);
	exit (1);
    }
    fp_pfm = fopen (argv[2], "wb");
    if (!fp_hnd) {
	printf ("Error, cannot open file %s for write\n", argv[2]);
	exit (1);
    }

    /* Read image */
    fseek (fp_hnd, 1024, SEEK_SET);
    fread (pt_lut, sizeof(unsigned char), (ROWS-1)*COLS / 4, fp_hnd);

    /* Read first row */
    for (i = 0; i < COLS; i++) {
	fread (&a, sizeof(unsigned long), 1, fp_hnd);
	buf[i] = a;
	b = a;
	fwrite (&b, sizeof(short), 1, fp_pfm);
    }

    /* Read first pixel of second row */
    fread (&a, sizeof(unsigned long), 1, fp_hnd);
    buf[i++] = a;
    b = a;
    fwrite (&b, sizeof(short), 1, fp_pfm);
    
    /* Decompress the rest */
    lut_idx = 0;
    lut_off = 0;
    while (i < COLS*ROWS) {
	unsigned long r11, r12, r21;
	unsigned short b;

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
	    fread (&dc, sizeof(unsigned char), 1, fp_hnd);
	    diff = dc;
	    break;
	case 1:
	    fread (&ds, sizeof(unsigned short), 1, fp_hnd);
	    diff = ds;
	    break;
	case 2:
	    fread (&dl, sizeof(unsigned long), 1, fp_hnd);
	    diff = dl;
	    break;
	}

	buf[i] = - r11 + r21 + r12 + diff;
	b = buf[i];
	//if (buf[i] > 65535) b = 65535;
	fwrite (&b, sizeof(float), 1, fp_pfm);
	i++;
    }

    fclose (fp_hnd);

    /* Write results */
    fclose (fp_pfm);

    return 0;
}
