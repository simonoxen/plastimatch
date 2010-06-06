/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "plm_image_header.h"
#include "plm_register_loadable.h"

void
plm_register_loadable (
    FloatImageType::ConstPointer fixed, 
    FloatImageType::ConstPointer moving
)
{
    FILE *fp;

    Plm_image_header pih;

    fp = fopen ("/tmp/plm_register_loadable.txt", "a");
    fprintf (fp, "Hello world\n");

    pih.set_from_itk_image (fixed);
    fprintf (fp, "F Size = %d %d %d\n", pih.Size(0), pih.Size(1), pih.Size(2));
    fprintf (fp, "F Spacing = %f %f %f\n", pih.m_spacing[0], pih.m_spacing[1], 
	pih.m_spacing[2]);

    pih.set_from_itk_image (moving);
    fprintf (fp, "M Size = %d %d %d\n", pih.Size(0), pih.Size(1), pih.Size(2));
    fprintf (fp, "M Spacing = %f %f %f\n", pih.m_spacing[0], pih.m_spacing[1], 
	pih.m_spacing[2]);
    fclose (fp);
}
