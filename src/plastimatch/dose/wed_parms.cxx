/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>

#include "plm_math.h"
#include "print_and_exit.h"
#include "string_util.h"
#include "wed_parms.h"

#ifndef NULL
#define NULL ((void*)0)
#endif

Wed_Parms::Wed_Parms ()
{
    this->debug = 0;
    this->group = 0;
    this->mode = 0;
    this->ray_step = 1.0f;
    this->input_ct_fn[0] = '\0';
    this->output_ct_fn[0] = '\0';

    this->src[0] = -1000.f;
    this->src[1] = 0.f;
    this->src[2] = 0.f;
    this->isocenter[0] = 0.f;
    this->isocenter[1] = 0.f;
    this->isocenter[2] = 0.f;
    this->beam_res = 1.f;

    this->vup[0] = 0.f;
    this->vup[1] = 0.f;
    this->vup[2] = 1.f;
    this->have_ires = false;
    this->ires[0] = 200;
    this->ires[1] = 200;
    this->have_ic = false;
    this->ic[0] = 99.5f;
    this->ic[1] = 99.5f;
    this->ap_spacing[0] = 1.;
    this->ap_spacing[1] = 1.;
    this->ap_offset = 100;

    this->dew_dim[0] = -999.;
    this->dew_dim[1] = -999.;
    this->dew_dim[2] = -999.;
    this->dew_origin[0] = -999.;
    this->dew_origin[1] = -999.;
    this->dew_origin[2] = -999.;
    this->dew_spacing[0] = -999.;
    this->dew_spacing[1] = -999.;
    this->dew_spacing[2] = -999.;
}

Wed_Parms::~Wed_Parms ()
{
}

static void
print_usage (void)
{
    printf ("Usage: wed config_file\n");
    printf ("Options:\n");
    printf ("\t--dew (reverse wed calculation)\n");
    printf ("\t--group <input .txt file> (computes multiple wed computations)\n");
    exit (1);
}

int
Wed_Parms::get_group_lines(char* groupfile)
{
  std::string line;
  std::ifstream text(groupfile);
  int numlines = 0;
  if (text.is_open())  {
    while (text.good()) {
      getline(text,line);	    
      if ( (!line.empty()) && (line.compare(0,1,"#")) )  {
	numlines++;
      }
    }
  }
  return numlines;

}

void
Wed_Parms::parse_group(int argc, char** argv, int linenumber)
{

  int linecounter = 0;

  for (int i=1; i<argc; i++) {
    if (!strcmp (argv[i], "--group")) {
      std::string line;
      std::ifstream text(argv[i+1]);
      if (text.is_open()) {
	while (text.good()) {
	  getline(text,line);
	  if ( (!line.empty()) && (line.compare(0,1,"#")) )  {

	    if (linecounter == linenumber)  {

	      std::string pvol_file;
	      std::string dose_file;
	      std::string dose_wed_file;

	      std::stringstream linestream(line);

	      linestream >> pvol_file >> dose_file >> dose_wed_file;

	      if (pvol_file.size()>=4)  {
		if (pvol_file.compare(pvol_file.size()-4,4,".mha"))  {
		  print_and_exit ("%s is not in <name>.mha format.\n", pvol_file.c_str());
		  return;
		}
	      }
	      else {print_and_exit ("%s is not in <name>.mha format.\n", pvol_file.c_str());}

	      if (dose_file.size()>=4)  {
		if (dose_file.compare(dose_file.size()-4,4,".mha"))  {
		  print_and_exit ("%s is not an .mha file.\n", dose_file.c_str());
		  return;
		}
	      }
	      else {print_and_exit ("%s is not in <name>.mha format.\n", dose_file.c_str());}


	      //	      std::cout<<pvol_file<<" "<<dose_file<<" "<<dose_wed_file<<std::endl;

	      strncpy (this->input_ct_fn, pvol_file.c_str(), _MAX_PATH);
	      //	      strncpy (this->input_dose_fn, dose_file.c_str(), _MAX_PATH);
	      this->input_dose_fn = dose_file.c_str();

	      //add "_wed" to  pvol_file names
	      pvol_file.insert (pvol_file.size()-4,"_wed");   
	      strncpy (this->output_ct_fn, pvol_file.c_str(), _MAX_PATH);
	      this->output_dose_fn = dose_wed_file.c_str();

	    }
	    linecounter++;
	  }
	}
      }
    }
  }
}

int
Wed_Parms::set_key_val (
    const char* key, 
    const char* val, 
    int section
)
{
    switch (section) {

        /* [INPUT SETTINGS] */
    case 0:
        if (!strcmp (key, "ray_step")) {
            if (sscanf (val, "%f", &this->ray_step) != 1) {
                goto error_exit;
            }
        }
        //Whether wed or reverse, input patient and rpl vol
        else if (!strcmp (key, "patient")) {
            strncpy (this->input_ct_fn, val, _MAX_PATH);
        }
        else if (!strcmp (key, "rpl_vol")) {
            this->rpl_vol_fn = val;
        }
	//Any mode will use the skin dose if specified
        else if (!strcmp (key, "skin")) {
            this->skin_fn = val;
        }
        //If normal wed procedure, input dose
	if (this->mode==0)  {
	  if (!strcmp (key, "dose")) {
	    this->input_dose_fn = val;
	  }
	}
	//If reverse wed procedure, input dose_wed
	if (this->mode==1)  {
	  if (!strcmp (key, "dose_wed")) {
	    this->input_dose_fn = val;
	  }
	}
	//If in depth/segmentation mode, input segment
	if (this->mode==2)  {
	  if (!strcmp (key, "segment")) {
	    this->input_dose_fn = val;
	  }
	}
	

        break;
        /* [OUTPUT SETTINGS] */
    case 1:
        //If normal wed procedure, output patient_wed and dose_wed
        if (this->mode==0)  {
	  if (!strcmp (key, "patient_wed")) {
            strncpy (this->output_ct_fn, val, _MAX_PATH);
	  }
	  else if (!strcmp (key, "dose_wed")) {
            this->output_dose_fn = val;
	  }
	}
	//If reverse wed  procedure, output only dose
        if (this->mode==1)  {
	  if (!strcmp (key, "dose")) {
            this->output_dose_fn = val;
	  }
	}

	//If in depth/segmentation mode, output depth matrix
        if (this->mode==2)  {
	  if (!strcmp (key, "depth")) {
            this->output_depth_fn = val;
	  }
	  else if (!strcmp (key, "aperture")) {
            this->output_ap_fn = val;
	  }
	}


        break;
        /* [BEAM] */
    case 2:
        if (!strcmp (key, "pos")) {
            if (sscanf (val, "%f %f %f", 
                    &(this->src[0]), 
                    &(this->src[1]), 
                    &(this->src[2])) != 3)
            {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "isocenter")) {
            if (sscanf (val, "%f %f %f", 
                    &(this->isocenter[0]), 
                    &(this->isocenter[1]), 
                    &(this->isocenter[2])) != 3)
            {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "res")) {
            if (sscanf (val, "%f", &(this->beam_res)) != 1) {
                goto error_exit;
            }
        }

        break;

        /* [APERTURE] */
    case 3:
        if (!strcmp (key, "up")) {
            if (sscanf (val, "%f %f %f", &(this->vup[0]), &(this->vup[1]), &(this->vup[2])) != 3) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "center")) {
            if (sscanf (val, "%f %f", &(this->ic[0]), &(this->ic[1])) != 2) {
                goto error_exit;
            }
            this->have_ic = true;
        }
        else if (!strcmp (key, "offset")) {
            if (sscanf (val, "%f", &(this->ap_offset)) != 1) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "spacing")) {
            if (sscanf (val, "%f %f", 
                    &(this->ap_spacing[0]), 
                    &(this->ap_spacing[1])) != 2)
            {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "resolution")) {
            if (sscanf (val, "%i %i", 
                    &(this->ires[0]), 
                    &(this->ires[1])) != 2)
            {
                goto error_exit;
            }
	    this->have_ires = true;

        }
        break;


        /* [DEW VOLUME] */
    case 4:
        if (!strcmp (key, "dew_dim")) {
            if (sscanf (val, "%f %f %f", &(this->dew_dim[0]), &(this->dew_dim[1]), &(this->dew_dim[2])) != 3) {
                goto error_exit;
            }
        }
        else if (!strcmp (key, "dew_origin")) {
            if (sscanf (val, "%f %f %f", &(this->dew_origin[0]), &(this->dew_origin[1]), &(this->dew_origin[2])) != 3) {
                goto error_exit;
            }
        }
        if (!strcmp (key, "dew_spacing")) {
            if (sscanf (val, "%f %f %f", &(this->dew_spacing[0]), &(this->dew_spacing[1]), &(this->dew_spacing[2])) != 3) {
                goto error_exit;
            }
        }
        break;

    }
    return 0;

error_exit:
    print_and_exit ("Unknown (key,val) combination: (%s,%s)\n", key, val);
    return -1;
}



void
Wed_Parms::parse_config (
    const char* config_fn
)
{
    /* Read file into string */
    std::ifstream t (config_fn);
    std::stringstream buffer;
    buffer << t.rdbuf();

    std::string buf;
    std::string buf_ori;    /* An extra copy for diagnostics */
    int section = -1;

    std::stringstream ss (buffer.str());

    while (getline (ss, buf)) {
        buf_ori = buf;
        buf = trim (buf);
        buf_ori = trim (buf_ori, "\r\n");

        if (buf == "") continue;
        if (buf[0] == '#') continue;

        if (buf[0] == '[') {
            if (buf.find ("[INPUT SETTINGS]") != std::string::npos
                || buf.find ("[input settings]") != std::string::npos)
            {
                section = 0;
                continue;
            }
            if (buf.find ("[OUTPUT SETTINGS]") != std::string::npos
                || buf.find ("[output settings]") != std::string::npos)
            {
                section = 1;
                continue;
            }
            else if (buf.find ("[BEAM]") != std::string::npos
                || buf.find ("[beam]") != std::string::npos)
            {
                section = 2;
                continue;
            }
            else if (buf.find ("[APERTURE]") != std::string::npos
                || buf.find ("[aperture]") != std::string::npos)
            {
                section = 3;
                continue;
            }
            else if (buf.find ("[DEW VOLUME]") != std::string::npos
                || buf.find ("[dew volume]") != std::string::npos)
            {
                section = 4;
                continue;
            }
            else {
                printf ("Parse error: %s\n", buf_ori.c_str());
            }
        }

        size_t key_loc = buf.find ("=");
        if (key_loc == std::string::npos) {
            continue;
        }

        std::string key = buf.substr (0, key_loc);
        std::string val = buf.substr (key_loc+1);
        key = trim (key);
        val = trim (val);

        if (key != "" && val != "") {
            if (this->set_key_val (key.c_str(), val.c_str(), section) < 0) {
                printf ("Parse error: %s\n", buf_ori.c_str());
            }
        }
    }
}

bool
Wed_Parms::parse_args (int argc, char** argv)
{
    int i;
    for (i=1; i<argc; i++) {
        if (argv[i][0] != '-') break;

        if (!strcmp (argv[i], "--debug")) {
	  this->debug = 1;
        }
	if (!strcmp (argv[i], "--group")) {
	  if (!argv[i+1])  { //group needs an argument
	    print_usage ();
            return false;
	  }
	  else {
	    this->group = get_group_lines(argv[i+1]);
	    return true;
	  }
        }
	if (!strcmp (argv[i], "--dew")) {
	  this->mode = 1;
        }

	if (!strcmp (argv[i], "--segdepth")) {
	  this->mode = 2;
        }

        else {
            print_usage ();
            break;
        }
    }

    if (!argv[i]) {
        print_usage ();
    } else {
        this->parse_config (argv[i]);
    }

#if 0
    if (this->d_lut == NULL) {
        /* measured bragg curve not supplied, try to generate */
        if (!this->generate ()) {
            return false;
        }
    }
    // JAS 2012.08.10
    //   Hack so that I can reuse the proton code.  The values
    //   don't actually matter.
    scene->beam->E0 = 1.0;
    scene->beam->spread = 1.0;
    scene->beam->dmax = 1.0;
#endif

    //Input CT always required
    if (this->input_ct_fn[0] == '\0') {
        fprintf (stderr, "\n** ERROR: Input patient image not specified in configuration file!\n");
        return false;
    }

    //Input "dose" always required
    if (this->input_dose_fn[0] == '\0') {
        fprintf (stderr, "\n** ERROR: Input dose not specified in configuration file!\n");
        return false;
    }

    //For wed mode, patient wed name is required.
    if (this->mode==0)  {
      if (this->output_ct_fn[0] == '\0') {
        fprintf (stderr, "\n** ERROR: Output file for patient water equivalent depth volume not specified in configuration file!\n");
        return false;
      }
    }

    //For wed or dew mode, output dose name required.
    if ((this->mode==0)||(this->mode==1))  {
	if (this->output_dose_fn[0] == '\0') {
	  fprintf (stderr, "\n** ERROR: Output file for dose volume not specified in configuration file!\n");
	  return false;
	}
    }

   //For depth/segmentation  mode, aperture and depth volumes required.
    if (this->mode==2)  {
      if (this->output_depth_fn[0] == '\0') {
        fprintf (stderr, "\n** ERROR: Output file for depths not specified in configuration file!\n");
        return false;
      }

      if (this->output_ap_fn[0] == '\0') {
        fprintf (stderr, "\n** ERROR: Output file for aperture not specified in configuration file!\n");
        return false;
      }
    }
    return true;
}
