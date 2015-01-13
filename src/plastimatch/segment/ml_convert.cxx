/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmsegment_config.h"
#include <stdio.h>
#include "itkImageRegionIterator.h"

#include "dir_list.h"
#include "file_util.h"
#include "logfile.h"
#include "ml_convert.h"
#include "plm_image.h"

class Ml_convert_private
{
public:
    std::string label_filename;
    std::string feature_dir;
    std::string output_filename;
};

Ml_convert::Ml_convert ()
{
    d_ptr = new Ml_convert_private;
}

Ml_convert::~Ml_convert ()
{
    delete d_ptr;
}


void Ml_convert::set_label_filename (const std::string& label_filename)
{
    d_ptr->label_filename = label_filename;
}

void Ml_convert::set_feature_directory (const std::string& feature_dir)
{
    d_ptr->feature_dir = feature_dir;
}

void Ml_convert::set_output_filename (const std::string& output_filename)
{
    d_ptr->output_filename = output_filename;
}

void Ml_convert::run ()
{
    /* Create files for ping-pong -- unfortunately we have to use FILE* 
       due to lack of fstream support for temporary filenames */
    FILE *fp[2], *current, *previous;
    fp[0] = make_tempfile ();
    fp[1] = make_tempfile ();
    current = fp[0];
    previous = fp[0];

    /* Load labelmap */
    lprintf ("Processing labelmap\n");
    Plm_image::Pointer labelmap = Plm_image::New (d_ptr->label_filename);
    UCharImageType::Pointer labelmap_itk = labelmap->itk_uchar();

    /* Dump labels to file */
    UCharImageType::RegionType rg = labelmap_itk->GetLargestPossibleRegion ();
    itk::ImageRegionIterator< UCharImageType > labelmap_it (labelmap_itk, rg);
    for (labelmap_it.GoToBegin(); !labelmap_it.IsAtEnd(); ++labelmap_it) {
        unsigned char v = (unsigned char) labelmap_it.Get();
        fprintf (current, "%d|\n", v == 0 ? -1 : 1);
    }

    /* Loop through feature files */
    int idx = 0;
    Dir_list dir_list (d_ptr->feature_dir);
    for (int i = 0; i < dir_list.num_entries; i++) {
        std::string dir_entry = dir_list.entry (i);

        /* Skip directories */
        if (is_directory(dir_entry)) {
            continue;
        }

        /* Load a feature image */
        Plm_image::Pointer feature = Plm_image::New (dir_entry);
        if (!feature->have_image()) {
            continue;
        }
        FloatImageType::Pointer feature_itk = labelmap->itk_float();

        /* Set up input and output file */
        lprintf ("Processing %s\n", dir_list.entries[i]);
        if (current == fp[0]) {
            previous = fp[0];
            current = fp[1];
        } else {
            previous = fp[1];
            current = fp[0];
        }
        rewind (previous);
        rewind (current);
        
        /* Loop through pixels, appending them to each line of file */
        itk::ImageRegionIterator< FloatImageType > feature_it (
            feature_itk, feature_itk->GetLargestPossibleRegion ());
        for (feature_it.GoToBegin(); !feature_it.IsAtEnd(); ++feature_it) {
            float v = (float) feature_it.Get();
            while (int c = getc (previous)) {
                if (c == '\n') {
                    break;
                }
                putc (c, current);
            }
            fprintf (current, " %d:%f\n", idx, v);
        }
        idx ++;
        break;
    }

    /* Finally, re-write temp file into output file */
    lprintf ("Processing final output\n");
    rewind (current);
    FILE *final_output = plm_fopen (d_ptr->output_filename.c_str(), "wb");
    while (int c = getc (current)) {
        if (c == EOF) {
            break;
        }
        putc (c, final_output);
    }
    fclose (fp[0]);
    fclose (fp[1]);
    fclose (final_output);
}
