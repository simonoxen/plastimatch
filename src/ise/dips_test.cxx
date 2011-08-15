/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* -----------------------------------------------------------------------
   igtalk 0      // simulates panel 0
   igtalk 1      // simulates panel 1
   ----------------------------------------------------------------------- */
//#include "config.h"
#define _USE_32BIT_TIME_T 1
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <io.h>
#include "dips_if.h"

#define FLUORO_MODE 0
//#define HIRES_IMAGE_HEIGHT 1536
//#define HIRES_IMAGE_WIDTH 2048
#define HIRES_IMAGE_HEIGHT 2304
#define HIRES_IMAGE_WIDTH 3200

void igpax_poll_images (PANEL* panelp, unsigned short* pixelp);

int
main (int argc, char* argv[])
{
    int cmd_rc = 0;
    char panel_name[12], pixel_name[12];
    HANDLE panelh, pixelh;
    int selector;
    PANEL* panelp;
    unsigned short* pixelp;

    if (argc != 2) {
        printf ("Usage: igpax panel_number\n");
	exit (2);
    }
    selector = atoi(argv[1]);

    /* Set up shared memory */
    sprintf (panel_name, "PANEL%i", selector);
    sprintf (pixel_name, "PIXEL%i", selector);
    panelh = CreateFileMapping (INVALID_HANDLE_VALUE, NULL, 
			PAGE_READWRITE,	0, sizeof (PANEL), 
			panel_name);
    if (!panelh) {
	fprintf (stderr, "Error opening shared memory for panel\n");
	exit (1);
    }
    pixelh = CreateFileMapping (INVALID_HANDLE_VALUE, NULL, 
			PAGE_READWRITE,	0, HIRES_IMAGE_WIDTH*HIRES_IMAGE_HEIGHT*2, 
			pixel_name);
    if (!panelh) {
	fprintf (stderr, "Error opening shared memory for pixel\n");
	exit (1);
    }
    panelp = (struct PANEL*) MapViewOfFile (panelh, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (!panelp) {
	fprintf (stderr, "Error mapping shared memory for panel\n");
	exit (1);
    }
    pixelp = (unsigned short*) MapViewOfFile (pixelh, FILE_MAP_ALL_ACCESS, 
						0, 0, 0);
    if (!pixelp) {
	fprintf (stderr, "Error mapping shared memory for pixel\n");
	exit (1);
    }
    panelp->status = READ;
    panelp->time = 0;
    panelp->ale = 0;
    panelp->xs = HIRES_IMAGE_WIDTH;
    panelp->ys = HIRES_IMAGE_HEIGHT;
    panelp->depth = 2;
    panelp->pixel = (short*) pixelp;

    srand ((unsigned) time(NULL));

    /* Wait for images */
    igpax_poll_images (panelp, pixelp);

    /* Never gets here */
    return 0;
}

/* A READ value of 1 means that DIPS has read the image. */
void
igpax_poll_images (PANEL* panelp, unsigned short* pixelp)
{
    int x_size = HIRES_IMAGE_WIDTH;
    int y_size = HIRES_IMAGE_HEIGHT;
    int x, y;
    unsigned short* p;

    while (1) {
	while ((panelp->status & READ) == 0) {
	    SleepEx (500, FALSE);
	}

	/* Fill in dummy image */
	p = pixelp;
	for (y = 0; y < y_size; y++) {
	    for (x = 0; x < x_size; x++) {
		if (x < 500 || x > x_size - 500) {
		    *p = 4400;
		} else if (y < 500 || y > y_size - 500) {
		    *p = 5500;
		} else {
		    *p = 8800;
		}
		*p += (rand() % 1000);
		p++;
	    }
        }
	printf ("\"Captured\" image!\n");

	/* Set timestamp */
	time (&panelp->time);

	/* For DIPS 3, we don't want so often ;-) */
	SleepEx (5000, FALSE);

	printf ("Hit any key to send another image to dips\n");
	getchar ();

	/* Let DIPS know we have an image */
	panelp->status = VALID;
    }
}
