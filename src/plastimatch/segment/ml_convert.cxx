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
#include "plm_timer.h"

class Ml_convert_private
{
public:
    std::string label_filename;
    std::list<std::string> feature_dir;
    std::string output_filename;
    std::string output_format;
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

void Ml_convert::add_feature_path (const std::string& feature_path)
{
    d_ptr->feature_dir.push_back (feature_path);
}

void Ml_convert::set_output_filename (const std::string& output_filename)
{
    d_ptr->output_filename = output_filename;
}

void Ml_convert::set_output_format (const std::string& output_format)
{
    d_ptr->output_format = output_format;
}

void Ml_convert::run ()
{
    Plm_timer pli;

    /* Create files for ping-pong -- unfortunately we have to use FILE* 
       due to lack of fstream support for temporary filenames */
    FILE *fp[2], *current, *previous;
    fp[0] = make_tempfile ();
    fp[1] = make_tempfile ();
    current = fp[0];
    previous = fp[0];

    bool vw_format = true;
    if (d_ptr->output_format == "libsvm") {
        vw_format = false;
    }

#define BUFSIZE 16*1024*1024
    char buf[BUFSIZE];
    size_t chars_in_buf = 0;
    char *buf_ptr;

    /* Load labelmap */
    lprintf ("Processing labelmap\n");
    Plm_image::Pointer labelmap = Plm_image::New (d_ptr->label_filename);
    UCharImageType::Pointer labelmap_itk = labelmap->itk_uchar();

    /* Dump labels to file */
    UCharImageType::RegionType rg = labelmap_itk->GetLargestPossibleRegion ();
    itk::ImageRegionIterator< UCharImageType > labelmap_it (labelmap_itk, rg);
    for (labelmap_it.GoToBegin(); !labelmap_it.IsAtEnd(); ++labelmap_it) {
        unsigned char v = (unsigned char) labelmap_it.Get();
        fprintf (current, "%d %s\n", 
            v == 0 ? -1 : 1, vw_format ? "|" : "");
    }

    /* Compile a complete list of feature input files */
    std::list<std::string> all_feature_files;
    std::list<std::string>::iterator fpath_it;
    for (fpath_it = d_ptr->feature_dir.begin();
         fpath_it != d_ptr->feature_dir.end();
         fpath_it++)
    {
        if (is_directory(*fpath_it)) {
            Dir_list dir_list (*fpath_it);
            for (int i = 0; i < dir_list.num_entries; i++) {
                /* Skip directories */
                std::string dir_entry = dir_list.entry (i);
                if (is_directory(dir_entry)) {
                    continue;
                }
                all_feature_files.push_back (dir_entry);
            }
        }
        else {
            all_feature_files.push_back (*fpath_it);
        }
    }
    
    /* Loop through feature files */
    int idx = 0;
    for (fpath_it = all_feature_files.begin();
         fpath_it != all_feature_files.end();
         fpath_it++)
    {
        std::string dir_entry = *fpath_it;

        /* Load a feature image */
        Plm_image::Pointer feature = Plm_image::New (dir_entry);
        if (!feature->have_image()) {
            continue;
        }
        FloatImageType::Pointer feature_itk = feature->itk_float();

        /* Set up input and output file */
        lprintf ("Processing %s\n", dir_entry.c_str());
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
        pli.start ();
        buf_ptr = 0;
        itk::ImageRegionIterator< FloatImageType > feature_it (
            feature_itk, feature_itk->GetLargestPossibleRegion ());
        for (feature_it.GoToBegin(); !feature_it.IsAtEnd(); ++feature_it) {
            /* Get pixel value */
            float v = (float) feature_it.Get();

            /* Copy existing line from previous file into current file */
            bool eol_found = false;
            while (1) {
                if (chars_in_buf == 0) {
                    chars_in_buf = fread (buf, 1, BUFSIZE, previous);
                    if (chars_in_buf == 0) {
                        break;
                    }
                    buf_ptr = buf;
                }
                size_t write_size = 0;
                for (write_size = 0; write_size < chars_in_buf; write_size++) {
                    if (buf_ptr[write_size] == '\n') {
                        eol_found = true;
                        break;
                    }
                }
                fwrite (buf_ptr, 1, write_size, current);
                buf_ptr += write_size;
                chars_in_buf -= write_size;
                if (eol_found) {
                    buf_ptr += 1;
                    chars_in_buf -= 1;
                    break;
                }
            }

            /* Append new value */
            if (vw_format) {
                fprintf (current, " %s:%f\n", dir_entry.c_str(), v);
            } else {
                fprintf (current, " %d:%f\n", idx+1, v);
            }
        }
        idx ++;
        printf ("Time = %f\n", (float) pli.report());
    }

    /* Finally, re-write temp file into final output file */
    lprintf ("Processing final output\n");
    rewind (current);
    FILE *final_output = plm_fopen (d_ptr->output_filename.c_str(), "wb");
    pli.start ();
    while ((chars_in_buf = fread (buf, 1, BUFSIZE, current)) != 0) {
        fwrite (buf, 1, chars_in_buf, final_output);
    }
    printf ("Time = %f\n", (float) pli.report());
    fclose (fp[0]);
    fclose (fp[1]);
    fclose (final_output);
}
