/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>
#include <stdlib.h>
#include <string.h>
#if defined (HAVE_GETOPT_LONG)
#include <getopt.h>
#else
#include "getopt.h"
#endif
#include "plm_registration.h"
#include "itk_image.h"
#include "itk_optim.h"
#include "xform.h"
//#include "plm_version.h"

#define BUFLEN 2048


int
set_key_val (Registration_Parms* regp, char* key, char* val, int section)
{
    Stage_Parms* stage = 0;
    if (section != 0) {
	stage = regp->stages[regp->num_stages-1];
    }

    //printf ("Got k/v: |%s|=|%s|\n", key, val);

    /* The following keywords are only allowed globally */
    if (!strcmp (key, "fixed")) {
	if (section != 0) goto error_not_stages;
	strncpy (regp->fixed_fn, val, _MAX_PATH);
    }
    else if (!strcmp (key, "moving")) {
	if (section != 0) goto error_not_stages;
	strncpy (regp->moving_fn, val, _MAX_PATH);
    }
    else if (!strcmp (key, "fixed_mask")) {
	if (section != 0) goto error_not_stages;
	strncpy (regp->fixed_mask_fn, val, _MAX_PATH);
    }
    else if (!strcmp (key, "moving_mask")) {
	if (section != 0) goto error_not_stages;
	strncpy (regp->moving_mask_fn, val, _MAX_PATH);
    }
    else if (!strcmp (key, "xf_in") || !strcmp (key, "xform_in") || !strcmp (key, "vf_in")) {
	if (section != 0) goto error_not_stages;
	strncpy (regp->xf_in_fn, val, _MAX_PATH);
    }

    /* The following keywords are allowed either globally or in stages */
    else if (!strcmp (key, "img_out_fmt")) {
	int fmt = IMG_OUT_FMT_AUTO;
	if (!strcmp (val, "dicom")) {
	    fmt = IMG_OUT_FMT_DICOM;
	} else {
	    goto error_exit;
	}
	if (section == 0) {
	    regp->img_out_fmt = fmt;
	} else {
	    stage->img_out_fmt = fmt;
	}
    }
    else if (!strcmp (key, "img_out")) {
	if (section == 0) {
	    strncpy (regp->img_out_fn, val, _MAX_PATH);
	} else {
	    strncpy (stage->img_out_fn, val, _MAX_PATH);
	}
    }
    else if (!strcmp (key, "vf_out")) {
	if (section == 0) {
	    strncpy (regp->vf_out_fn, val, _MAX_PATH);
	} else {
	    strncpy (stage->vf_out_fn, val, _MAX_PATH);
	}
    }
    else if (!strcmp (key, "xf_out") || !strcmp (key, "xform_out")) {
	if (section == 0) {
	    strncpy (regp->xf_out_fn, val, _MAX_PATH);
	} else {
	    strncpy (stage->xf_out_fn, val, _MAX_PATH);
	}
    }

    /* The following keywords are only allowed in stages */
    else if (!strcmp (key, "xform")) {
	if (section == 0) goto error_not_global;
	if (!strcmp(val,"rigid") || !strcmp(val,"versor")) {
	    stage->xform_type = STAGE_TRANSFORM_VERSOR;
	}
	else if (!strcmp (val,"affine")) {
	    stage->xform_type = STAGE_TRANSFORM_AFFINE;
	}
	else if (!strcmp (val,"translation")) {
	    stage->xform_type = STAGE_TRANSFORM_TRANSLATION;
	}
	else if (!strcmp (val,"bspline")) {
	    stage->xform_type = STAGE_TRANSFORM_BSPLINE;
	}
	else if (!strcmp (val,"vf")) {
	    stage->xform_type = STAGE_TRANSFORM_VECTOR_FIELD;
	}
	else {
	    goto error_exit;
	}
    }
    else if (!strcmp (key, "optim")) {
	if (section == 0) goto error_not_global;
	if (!strcmp(val,"none")) {
	    stage->optim_type = OPTIMIZATION_NO_REGISTRATION;
	}
	else if (!strcmp(val,"amoeba")) {
	    stage->optim_type = OPTIMIZATION_AMOEBA;
	}
	else if (!strcmp(val,"rsg")) {
	    stage->optim_type = OPTIMIZATION_RSG;
	}
	else if (!strcmp(val,"versor")) {
	    stage->optim_type = OPTIMIZATION_VERSOR;
	}
	else if (!strcmp(val,"lbfgs")) {
	    stage->optim_type = OPTIMIZATION_LBFGS;
	}
	else if (!strcmp(val,"lbfgsb")) {
	    stage->optim_type = OPTIMIZATION_LBFGSB;
	}
	else if (!strcmp(val,"demons")) {
	    stage->optim_type = OPTIMIZATION_DEMONS;
	}
	else if (!strcmp(val,"steepest")) {
	    stage->optim_type = OPTIMIZATION_STEEPEST;
	}
	else {
	    goto error_exit;
	}
    }
    else if (!strcmp (key, "impl")) {
	if (section == 0) goto error_not_global;
	if (!strcmp(val,"none")) {
	    stage->impl_type = IMPLEMENTATION_NONE;
	}
	else if (!strcmp(val,"itk")) {
	    stage->impl_type = IMPLEMENTATION_ITK;
	}
	else if (!strcmp(val,"gpuit_cpu")) {
	    stage->impl_type = IMPLEMENTATION_GPUIT_CPU;
	}
	else if (!strcmp(val,"gpuit_brook")) {
	    stage->impl_type = IMPLEMENTATION_GPUIT_BROOK;
	}
	else {
	    goto error_exit;
	}
    }
    else if (!strcmp (key, "metric")) {
	if (section == 0) goto error_not_global;
	if (!strcmp(val,"mse")) {
	    stage->metric_type = METRIC_MSE;
	}
	else if (!strcmp(val,"mi")) {
	    stage->metric_type = METRIC_MI;
	}
	else if (!strcmp(val,"mattes")) {
	    stage->metric_type = METRIC_MI_MATTES;
	}
	else {
	    goto error_exit;
	}
    }
    else if (!strcmp (key, "background_val")) {
	if (section == 0) goto error_not_global;
	if (sscanf (val, "%g", &stage->background_val) != 1) {
	    goto error_exit;
	}
    }
    else if (!strcmp (key, "background_max")) {
	if (section == 0) goto error_not_global;
	if (sscanf (val, "%g", &stage->background_max) != 1) {
	    goto error_exit;
	}
    }
    else if (!strcmp (key, "min_its")) {
	if (section == 0) goto error_not_global;
	if (sscanf (val, "%d", &stage->min_its) != 1) {
	    goto error_exit;
	}
    }
    else if (!strcmp (key, "max_its")) {
	if (section == 0) goto error_not_global;
	if (sscanf (val, "%d", &stage->max_its) != 1) {
	    goto error_exit;
	}
    }
    else if (!strcmp (key, "grad_tol")) {
	if (section == 0) goto error_not_global;
	if (sscanf (val, "%g", &stage->grad_tol) != 1) {
	    goto error_exit;
	}
    }
    else if (!strcmp (key, "max_step")) {
	if (section == 0) goto error_not_global;
	if (sscanf (val, "%g", &stage->max_step) != 1) {
	    goto error_exit;
	}
    }
    else if (!strcmp (key, "min_step")) {
	if (section == 0) goto error_not_global;
	if (sscanf (val, "%g", &stage->min_step) != 1) {
	    goto error_exit;
	}
    }
    else if (!strcmp (key, "convergence_tol")) {
	if (section == 0) goto error_not_global;
	if (sscanf (val, "%g", &stage->convergence_tol) != 1) {
	    goto error_exit;
	}
    }
    else if (!strcmp (key, "mattes_histogram_bins") 
	    || !strcmp (key, "mi_histogram_bins")) {
	if (section == 0) goto error_not_global;
	if (sscanf (val, "%d", &stage->mi_histogram_bins) != 1) {
	    goto error_exit;
	}
    }
    else if (!strcmp (key, "mattes_num_spatial_samples")
	    || !strcmp (key, "mi_num_spatial_samples")) {
	if (section == 0) goto error_not_global;
	if (sscanf (val, "%d", &stage->mi_num_spatial_samples) != 1) {
	    goto error_exit;
	}
    }
    else if (!strcmp (key, "demons_std")) {
	if (section == 0) goto error_not_global;
	if (sscanf (val, "%g", &stage->demons_std) != 1) {
	    goto error_exit;
	}
    }
    else if (!strcmp (key, "demons_acceleration")) {
	if (section == 0) goto error_not_global;
	if (sscanf (val, "%g", &stage->demons_acceleration) != 1) {
	    goto error_exit;
	}
    }
    else if (!strcmp (key, "demons_homogenization")) {
	if (section == 0) goto error_not_global;
	if (sscanf (val, "%g", &stage->demons_homogenization) != 1) {
	    goto error_exit;
	}
    }
    else if (!strcmp (key, "demons_filter_width")) {
	if (section == 0) goto error_not_global;
	if (sscanf (val, "%d %d %d", &(stage->demons_filter_width[0]), &(stage->demons_filter_width[1]), &(stage->demons_filter_width[2])) != 3) {
	    goto error_exit;
	}
    }
    else if (!strcmp (key, "amoeba_parameter_tol")) {
	if (section == 0) goto error_not_global;
	if (sscanf (val, "%g", &(stage->amoeba_parameter_tol)) != 1) {
	    goto error_exit;
	}
    }
    else if (!strcmp (key, "res")) {
	if (section == 0) goto error_not_global;
	if (sscanf (val, "%d %d %d", &(stage->resolution[0]), &(stage->resolution[1]), &(stage->resolution[2])) != 3) {
	    goto error_exit;
	}
    }
    else if (!strcmp (key, "num_grid")) {
	if (section == 0) goto error_not_global;
	if (sscanf (val, "%d %d %d", &(stage->num_grid[0]), &(stage->num_grid[1]), &(stage->num_grid[2])) != 3) {
	    goto error_exit;
	}
	stage->grid_method = 0;
    }
    else if (!strcmp (key, "grid_spac")) {
	if (section == 0) goto error_not_global;
	if (sscanf (val, "%g %g %g", &(stage->grid_spac[0]), &(stage->grid_spac[1]), &(stage->grid_spac[2])) != 3) {
	    goto error_exit;
	}
	stage->grid_method = 1;
    }
    else if (!strcmp (key, "histoeq")) {
	if (section == 0) goto error_not_global;
	if (sscanf (val, "%d", &(stage->histoeq)) != 1) {
	    goto error_exit;
	}
    }
    else {
	goto error_exit;
    }
    return 0;

error_not_stages:
    printf ("This key (%s) not allowed in a stages section\n", key);
    return -1;

error_not_global:
    printf ("This key (%s) not is allowed in a global section\n", key);
    return -1;

error_exit:
    printf ("Unknown (key,val) combination: (%s,%s)\n", key, val);
    return -1;
}

int
get_command_file_line (FILE* fp, Registration_Parms* regp)
{
    char buf_ori[BUFLEN];    /* An extra copy for diagnostics */
    char buf[BUFLEN];
    char *p, *key, *val;
    int section = 0;

    while (1) {
	if (!fgets (buf,BUFLEN,fp)) return 0;
	strcpy (buf_ori,buf);
	p = buf;
	p += strspn (p, " \t\n\r");
	if (!p) continue;
	if (*p == '#') continue;
	if (*p == '[') {
	    p += strspn (p, " \t\n\r[");
	    p = strtok (p, "]");
	    if (!strcmp (p, "GLOBAL") || !strcmp (p, "global")) {
		section = 0;
		continue;
	    } else if (!strcmp (p, "STAGE") || !strcmp (p, "stage")) {
		section = 1;
		regp->num_stages ++;
		regp->stages = (Stage_Parms**) realloc (regp->stages, regp->num_stages * sizeof(Stage_Parms*));
		if (regp->num_stages == 1) {
		    regp->stages[regp->num_stages-1] = new Stage_Parms();
		} else {
		    regp->stages[regp->num_stages-1] = new Stage_Parms(*(regp->stages[regp->num_stages-2]));
		}
		continue;
	    } else if (!strcmp (p, "COMMENT") || !strcmp (p, "comment")) {
		section = 2;
		continue;
	    } else {
		printf ("Parse error: %s\n", buf_ori);
		return -1;
	    }
	}
	if (section == 2) continue;
	key = strtok (p, "=");
	val = strtok (NULL, "\n\r");
	if (key && val) {
	    if (set_key_val (regp, key, val, section) < 0) {
		printf ("Parse error: %s\n", buf_ori);
		return -1;
	    }
	}
    }
    return 0;
}

int
parse_command_file (Registration_Parms* regp, const char* options_fn)
{
    FILE* fp;

    /* Open file */
    fp = fopen (options_fn, "r");
    if (!fp) {
	printf ("Error: could not open file \"%s\" for read.\n", options_fn);
	return -1;
    }

    /* Loop through file, parsing each line, and adding it to regp */
    while (!feof(fp)) {
	if (get_command_file_line (fp, regp) < 0) {
	    return -1;
	}
    }

    /* Close file */
    fclose (fp);
    return 0;
}

