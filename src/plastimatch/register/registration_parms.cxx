/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmregister_config.h"
#include <fstream>
#include <list>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#if defined (_WIN32)
// win32 directory stuff
#else
#include <sys/types.h>
#include <dirent.h>
#endif

#include "parameter_parser.h"
#include "plm_path.h"
#include "print_and_exit.h"
#include "registration_parms.h"
#include "shared_parms.h"
#include "stage_parms.h"
#include "string_util.h"

class Registration_parms_private
{
public:
    std::string moving_fn;
    std::string fixed_fn;
    std::list<Stage_parms*> stages;
    Shared_parms *shared;
public:
    Registration_parms_private () {
        shared = new Shared_parms;
    }
    ~Registration_parms_private () {
        std::list<Stage_parms*>::iterator it;
        for (it = stages.begin(); it != stages.end(); it++) {
            delete *it;
        }
        delete shared;
    }
};

class Registration_parms_parser : public Parameter_parser
{
public:
    Registration_parms *rp;
public:
    Registration_parms_parser (Registration_parms *rp)
    {
        this->rp = rp;
    }
public:
    virtual int process_section (
        const std::string& section)
    {
        if (section == "GLOBAL") {
            return 0;
        }
        if (section == "STAGE") {
            rp->append_stage ();
            return 0;
        }
        if (section == "COMMENT") {
            return 0;
        }
        if (section == "PROCESS") {
            rp->append_process_stage ();
            return 0;
        }

        /* else, unknown section */
        return -1;
    }
    virtual int process_key_value (
        const std::string& section,
        const std::string& key, 
        const std::string& val)
    {
        return this->rp->set_key_value (section, key, val);
    }
};

Registration_parms::Registration_parms()
{
    d_ptr = new Registration_parms_private;

    img_out_fmt = IMG_OUT_FMT_AUTO;
    img_out_type = PLM_IMG_TYPE_UNDEFINED;
    *img_out_fn = 0;
    *xf_in_fn = 0;
    xf_out_itk = false;
    *vf_out_fn = 0;
    *log_fn = 0;
    init_type = STAGE_TRANSFORM_NONE;
    default_value = 0.0;
    num_stages = 0;
    *moving_dir = 0;
    *fixed_dir = 0;
    *img_out_dir = 0;
    *vf_out_dir = 0;
    for (int i=0; i<256; i++) {
        moving_jobs[i][0] = '\0';
        fixed_jobs[i][0] = '\0';
    }
    job_idx = 0;
    num_jobs = 1;
}

Registration_parms::~Registration_parms()
{
    delete d_ptr;
}

// JAS 2012.02.13 -- TODO: Move somewhere more appropriate
static void
check_trailing_slash (char* s)
{
    int i=0;
    while (s[i++] != '\0');

    if (s[i-2] != '/') {
        strcat (s, "/");
    }
}

int
populate_jobs (char jobs[255][_MAX_PATH], char* dir)
{
#if defined (_WIN32)
    // Win32 Version goes here
    return 0;
#else
    DIR *dp;
    struct dirent *ep;
    int z=0;
    char buffer[_MAX_PATH];

    dp = opendir (dir);

    if (dp != NULL) {
        while ((ep=readdir(dp))) {
            memset (buffer, 0, _MAX_PATH);
            if (!strcmp(ep->d_name, ".")) {
                continue;
            } else if (!strcmp(ep->d_name, "..")) {
                continue;
            }
            strncpy (jobs[z++], ep->d_name, _MAX_PATH);
        }
        (void) closedir (dp);
    } else {
        printf ("Error: Could not open %s\n", dir);
    }

    return z;
#endif
}

int 
Registration_parms::set_key_value (
    const std::string& section,
    const std::string& key, 
    const std::string& val)
{
    int rc;
    Stage_parms *stage = 0;
    Shared_parms *shared = 0;
    Process_parms::Pointer process;
    bool section_global = false;
    bool section_stage = false;
    bool section_process = false;

    if (section == "COMMENT") {
        return 0;
    }

    if (section == "GLOBAL") {
        shared = d_ptr->shared;
        section_global = true;
    }
    else if (section == "STAGE") {
        stage = d_ptr->stages.back();
        shared = stage->get_shared_parms();
        section_stage = true;
    }
    else if (section == "PROCESS") {
        stage = d_ptr->stages.back();
        process = stage->get_process_parms();
        section_process = true;
    }

    /* The following keywords are only allowed globally */
    if (key == "fixed") {
        if (!section_global) goto key_only_allowed_in_section_global;
        d_ptr->fixed_fn = val;
    }
    else if (key == "moving") {
        if (!section_global) goto key_only_allowed_in_section_global;
        d_ptr->moving_fn = val;
    }
    else if (key == "fixed_dir") {
        if (!section_global) goto key_only_allowed_in_section_global;
        strncpy (this->fixed_dir, val.c_str(), _MAX_PATH);
        check_trailing_slash (this->fixed_dir);
        this->num_jobs = populate_jobs (this->fixed_jobs, this->fixed_dir);
    }
    else if (key == "moving_dir") {
        if (!section_global) goto key_only_allowed_in_section_global;
        strncpy (this->moving_dir, val.c_str(), _MAX_PATH);
        check_trailing_slash (this->moving_dir);
        this->num_jobs = populate_jobs (this->moving_jobs, this->moving_dir);
    }
    else if (key == "img_out_dir") {
        if (!section_global) goto key_only_allowed_in_section_global;
        strncpy (this->img_out_dir, val.c_str(), _MAX_PATH);
        check_trailing_slash (this->img_out_dir);
    }
    else if (key == "vf_out_dir") {
        if (!section_global) goto key_only_allowed_in_section_global;
        strncpy (this->vf_out_dir, val.c_str(), _MAX_PATH);
        check_trailing_slash (this->vf_out_dir);
    }
    else if (key == "xf_in"
        || key == "xform_in"
        || key == "vf_in")
    {
        if (!section_global) goto key_only_allowed_in_section_global;
        strncpy (this->xf_in_fn, val.c_str(), _MAX_PATH);
    }
    else if (key == "log" || key == "logfile") {
        if (!section_global) goto key_only_allowed_in_section_global;
        strncpy (this->log_fn, val.c_str(), _MAX_PATH);
    }
    else if (key == "fixed_landmarks") {
        if (!section_global) goto key_only_allowed_in_section_stage;
        this->fixed_landmarks_fn = val;
    }
    else if (key == "moving_landmarks") {
        if (!section_global) goto key_only_allowed_in_section_stage;
        this->moving_landmarks_fn = val;
    }
    else if (key == "fixed_landmark_list") {
        if (!section_global) goto key_only_allowed_in_section_stage;
        this->fixed_landmarks_list = val;
    }
    else if (key == "moving_landmark_list") {
        if (!section_global) goto key_only_allowed_in_section_stage;
        this->moving_landmarks_list = val;
    }

    /* The following keywords are allowed either globally or in stages */
    else if (key == "background_val"
        || key == "default_value")
    {
        float f;
        if (sscanf (val.c_str(), "%g", &f) != 1) {
            goto error_exit;
        }
        if (section_global) {
            this->default_value = f;
        } else if (section_stage) {
            stage->default_value = f;
        } else {
            goto key_not_allowed_in_section_process;
        }
    }
    else if (key == "fixed_mask" || key == "fixed_roi") {
        if (section_process) goto key_not_allowed_in_section_process;
        shared->fixed_roi_fn = val;
    }
    else if (key == "moving_mask" || key == "moving_roi") {
        if (section_process) goto key_not_allowed_in_section_process;
        shared->moving_roi_fn = val;
    }
    else if (key == "fixed_roi_enable") {
        if (section_process) goto key_not_allowed_in_section_process;
        shared->fixed_roi_enable = string_value_true (val);
    }
    else if (key == "moving_roi_enable")
    {
        if (section_process) goto key_not_allowed_in_section_process;
        shared->moving_roi_enable = string_value_true (val);
    }
    else if (key == "legacy_subsampling") {
        if (section_process) goto key_not_allowed_in_section_process;
        shared->legacy_subsampling = string_value_true (val);
    }
    else if (key == "img_out" || key == "image_out") {
        if (section_global) {
            strncpy (this->img_out_fn, val.c_str(), _MAX_PATH);
        } else if (section_stage) {
            strncpy (stage->img_out_fn, val.c_str(), _MAX_PATH);
        } else {
            goto key_not_allowed_in_section_process;
        }
    }
    else if (key == "img_out_fmt") {
        int fmt = IMG_OUT_FMT_AUTO;
        if (val == "dicom") {
            fmt = IMG_OUT_FMT_DICOM;
        } else {
            goto error_exit;
        }
        if (section_global) {
            this->img_out_fmt = fmt;
        } else if (section_stage) {
            stage->img_out_fmt = fmt;
        } else {
            goto key_not_allowed_in_section_process;
        }
    }
    else if (key == "img_out_type") {
        Plm_image_type type = plm_image_type_parse (val.c_str());
        if (type == PLM_IMG_TYPE_UNDEFINED) {
            goto error_exit;
        }
        if (section_global) {
            this->img_out_type = type;
        } else if (section_stage) {
            stage->img_out_type = type;
        } else {
            goto key_not_allowed_in_section_process;
        }
    }
    else if (key == "vf_out") {
        if (section_global) {
            strncpy (this->vf_out_fn, val.c_str(), _MAX_PATH);
        } else if (section_stage) {
            strncpy (stage->vf_out_fn, val.c_str(), _MAX_PATH);
        } else {
            goto key_not_allowed_in_section_process;
        }
    }
    else if (key == "xf_out_itk") {
        bool value;
        if (string_value_true (val)) {
            value = true;
        } else {
            value = false;
        }
        if (section_global) {
            this->xf_out_itk = value;
        } else if (section_stage) {
            stage->xf_out_itk = value;
        } else {
            goto key_not_allowed_in_section_process;
        }
    }
    else if (key == "xf_out" || key == "xform_out") {
        /* xf_out is special.  You can have more than one of these.  
           This capability is used by the slicer plugin. */
        if (section_global) {
            this->xf_out_fn.push_back (val.c_str());
        } else if (section_stage) {
            stage->xf_out_fn.push_back (val.c_str());
        } else {
            goto key_not_allowed_in_section_process;
        }
    }
    else if (key == "warped_landmarks") {
        if (section_global) {
            this->warped_landmarks_fn = val;
        } else if (section_stage) {
            stage->warped_landmarks_fn = val;
        } else {
            goto key_not_allowed_in_section_process;
        }
    }

    /* The following keywords are only allowed in stages */
    else if (key == "resume") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (string_value_true (val)) {
            stage->resume_stage = true;
        }
    }
    else if (key == "xform") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (val == "translation") {
            stage->xform_type = STAGE_TRANSFORM_TRANSLATION;
        }
        else if (val == "rigid" || val == "versor") {
            stage->xform_type = STAGE_TRANSFORM_VERSOR;
        }
        else if (val == "quaternion") {
            stage->xform_type = STAGE_TRANSFORM_QUATERNION;
        }
        else if (val == "affine") {
            stage->xform_type = STAGE_TRANSFORM_AFFINE;
        }
        else if (val == "bspline") {
            stage->xform_type = STAGE_TRANSFORM_BSPLINE;
        }
        else if (val == "vf") {
            stage->xform_type = STAGE_TRANSFORM_VECTOR_FIELD;
        }
        else if (val == "align_center") {
            stage->xform_type = STAGE_TRANSFORM_ALIGN_CENTER;
        }
        else {
            goto error_exit;
        }
    }
    else if (key == "optim") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (val == "none") {
            stage->optim_type = OPTIMIZATION_NO_REGISTRATION;
        }
        else if (val == "amoeba") {
            stage->optim_type = OPTIMIZATION_AMOEBA;
        }
        else if (val == "oneplusone") {
            stage->optim_type = OPTIMIZATION_ONEPLUSONE;
        }
        else if (val == "frpr") {
            stage->optim_type = OPTIMIZATION_FRPR;
        }
        else if (val == "demons") {
            stage->optim_type = OPTIMIZATION_DEMONS;
        }
        else if (val == "grid") {
            stage->optim_type = OPTIMIZATION_GRID_SEARCH;
        }
        else if (val == "lbfgs") {
            stage->optim_type = OPTIMIZATION_LBFGS;
        }
        else if (val == "lbfgsb") {
            stage->optim_type = OPTIMIZATION_LBFGSB;
        }
        else if (val == "liblbfgs") {
            stage->optim_type = OPTIMIZATION_LIBLBFGS;
        }
        else if (val == "nocedal") {
            stage->optim_type = OPTIMIZATION_LBFGSB;
        }
        else if (val == "rsg") {
            stage->optim_type = OPTIMIZATION_RSG;
        }
        else if (val == "steepest") {
            stage->optim_type = OPTIMIZATION_STEEPEST;
        }
        else if (val == "versor") {
            stage->optim_type = OPTIMIZATION_VERSOR;
        }
        else {
            goto error_exit;
        }
    }
    else if (key == "impl") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (val == "none") {
            stage->impl_type = IMPLEMENTATION_NONE;
        }
        else if (val == "itk") {
            stage->impl_type = IMPLEMENTATION_ITK;
        }
        else if (val == "plastimatch") {
            stage->impl_type = IMPLEMENTATION_PLASTIMATCH;
        }
        else {
            goto error_exit;
        }
    }
    else if (key == "optim_subtype") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (val == "fsf") {
            stage->optim_subtype = OPTIMIZATION_SUB_FSF;
        }
        else if (val == "diffeomorphic") {
            stage->optim_subtype = OPTIMIZATION_SUB_DIFF_ITK;
        }
        else if (val == "log_domain") {
            stage->optim_subtype = OPTIMIZATION_SUB_LOGDOM_ITK;
        }
        else if (val == "sym_log_domain") {
            stage->optim_subtype = OPTIMIZATION_SUB_SYM_LOGDOM_ITK;
        }
        else {
            goto error_exit;
        }
    }
    else if (key == "threading") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (val == "single") {
            stage->threading_type = THREADING_CPU_SINGLE;
        }
        else if (val == "openmp") {
#if (OPENMP_FOUND)
            stage->threading_type = THREADING_CPU_OPENMP;
#else
            stage->threading_type = THREADING_CPU_SINGLE;
#endif
        }
        else if (val == "cuda") {
#if (CUDA_FOUND)
            stage->threading_type = THREADING_CUDA;
#elif (OPENMP_FOUND)
            stage->threading_type = THREADING_CPU_OPENMP;
#else
            stage->threading_type = THREADING_CPU_SINGLE;
#endif
        }
        else {
            goto error_exit;
        }
    }
    else if (key == "alg_flavor"
        || key == "flavor")
    {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (val.length() >= 1) {
            stage->alg_flavor = val[0];
        }
        else {
            goto error_exit;
        }
    }
    else if (key == "metric") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (val == "mse" || val == "MSE") {
            stage->metric_type = METRIC_MSE;
        }
        else if (val == "mi" || val == "MI") {
            stage->metric_type = METRIC_MI;
        }
        else if (val == "nmi" || val == "NMI") {
            stage->metric_type = METRIC_NMI;
        }
        else if (val == "mattes") {
            stage->metric_type = METRIC_MI_MATTES;
        }
        else {
            goto error_exit;
        }
    }
    else if (key == "histogram_type") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (val == "eqsp" || val == "EQSP") {
            stage->mi_histogram_type = HIST_EQSP;
        }
        else if (val == "vopt" || val == "VOPT") {
            stage->mi_histogram_type = HIST_VOPT;
        }
        else {
            goto error_exit;
        }
    }
    else if (key == "regularization")
    {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (val == "none") {
            stage->regularization_type = REGULARIZATION_NONE;
        }
        else if (val == "analytic") {
            stage->regularization_type = REGULARIZATION_BSPLINE_ANALYTIC;
        }
        else if (val == "semi_analytic") {
            stage->regularization_type = REGULARIZATION_BSPLINE_SEMI_ANALYTIC;
        }
        else if (val == "numeric") {
            stage->regularization_type = REGULARIZATION_BSPLINE_NUMERIC;
        }
        else {
            goto error_exit;
        }
    }
    else if (key == "regularization_lambda"
        || key == "young_modulus") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%f", &stage->regularization_lambda) != 1) {
            goto error_exit;
        }
    }
    else if (key == "background_max") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->background_max) != 1) {
            goto error_exit;
        }
    }
    else if (key == "min_its") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%d", &stage->min_its) != 1) {
            goto error_exit;
        }
    }
    else if (key == "iterations" 
        || key == "max_iterations"
        || key == "max_its"
        || key == "its")
    {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%d", &stage->max_its) != 1) {
            goto error_exit;
        }
    }
    else if (key == "learn_rate") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->learn_rate) != 1) {
            goto error_exit;
        }
    }
    else if (key == "grad_tol") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->grad_tol) != 1) {
            goto error_exit;
        }
    }
    else if (key == "pgtol") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%f", &stage->pgtol) != 1) {
            goto error_exit;
        }
    }
    else if (key == "max_step") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->max_step) != 1) {
            goto error_exit;
        }
    }
    else if (key == "min_step") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->min_step) != 1) {
            goto error_exit;
        }
    }
    else if (key == "rsg_grad_tol") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->rsg_grad_tol) != 1) {
            goto error_exit;
        }
    }
    else if (key == "translation_scale_factor") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%d", &stage->translation_scale_factor) != 1) {
            goto error_exit;
        }
    }
    else if (key == "convergence_tol") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->convergence_tol) != 1) {
            goto error_exit;
        }
    }
    else if (key == "opo_epsilon") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->opo_epsilon) != 1) {
            goto error_exit;
        }
    }
    else if (key == "opo_initial_search_rad") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->opo_initial_search_rad) != 1) {
            goto error_exit;
        }
    }
    else if (key == "frpr_step_tol") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->frpr_step_tol) != 1) {
            goto error_exit;
        }
    }
    else if (key == "frpr_step_length") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->frpr_step_length) != 1) {
            goto error_exit;
        }
    }
    else if (key == "frpr_max_line_its") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%d", &stage->frpr_max_line_its) != 1) {
            goto error_exit;
        }
    }
    else if (key == "mattes_histogram_bins" 
        || key == "mi_histogram_bins") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        rc = sscanf (val.c_str(), "%d %d", &stage->mi_histogram_bins_fixed,
            &stage->mi_histogram_bins_moving);
        if (rc == 1) {
            stage->mi_histogram_bins_moving = stage->mi_histogram_bins_fixed;
        } else if (rc != 2) {
            goto error_exit;
        }
    }
    else if (key == "mattes_fixed_minVal"
        ||key == "mi_fixed_minVal") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->mi_fixed_image_minVal) != 1) {
            goto error_exit;
        }
    }
    else if (key == "mattes_fixed_maxVal"
        ||key == "mi_fixed_maxVal") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->mi_fixed_image_maxVal) != 1) {
            goto error_exit;
        }
    }
    else if (key == "mattes_moving_minVal"
        ||key == "mi_moving_minVal") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->mi_moving_image_minVal) != 1) {
            goto error_exit;
        }
    }
    else if (key == "mattes_moving_maxVal"
        ||key == "mi_moving_maxVal") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->mi_moving_image_maxVal) != 1) {
            goto error_exit;
        }
    }
    else if (key == "num_samples"
        || key == "mattes_num_spatial_samples"
        || key == "mi_num_spatial_samples") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%d", &stage->mi_num_spatial_samples) != 1) {
            goto error_exit;
        }
    }
    else if (key == "num_samples_pct") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%f", &stage->mi_num_spatial_samples_pct) != 1) {
            goto error_exit;
        }
    }
    else if (key == "demons_std_deformation_field") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->demons_std) != 1) {
            goto error_exit;
        }
    }
    else if (key == "demons_std_update_field") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->demons_std_update_field) != 1) {
            goto error_exit;
        }
    }
    else if (key == "demons_step_length") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->demons_step_length) != 1) {
            goto error_exit;
        }
    }
    else if (key == "demons_smooth_deformation_field") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (string_value_true (val)) {
            stage->demons_smooth_deformation_field = true;
        }
        else
            stage->demons_smooth_deformation_field = false;
    }
    else if (key == "demons_smooth_update_field") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (string_value_true (val)) {
            stage->demons_smooth_update_field = true;
        }
        else
            stage->demons_smooth_update_field = false;
    }
    else if (key == "demons_gradient_type")
    {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (val == "symmetric") {
            stage->demons_gradient_type = SYMMETRIC;
        }
        else if (val == "fixed") {
            stage->demons_gradient_type = FIXED_IMAGE;
        }
        else if (val == "warped_moving") {
            stage->demons_gradient_type = WARPED_MOVING;
        }
        else if (val == "mapped_moving") {
            stage->demons_gradient_type = MAPPED_MOVING;
        }
        else {
            goto error_exit;
        }
    }
    else if (key == "num_approx_terms_log_demons") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%d", &stage->num_approx_terms_log_demons) != 1) {
            goto error_exit;
        }
    }
    else if (key == "demons_homogenization") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->demons_homogenization) != 1) {
            goto error_exit;
        }
    }
    else if (key == "demons_filter_width") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%d %d %d", 
                &(stage->demons_filter_width[0]), 
                &(stage->demons_filter_width[1]), 
                &(stage->demons_filter_width[2])) != 3) {
            goto error_exit;
        }
    }
    else if (key == "amoeba_parameter_tol") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &(stage->amoeba_parameter_tol)) != 1) {
            goto error_exit;
        }
    }
    else if (key == "gridsearch_min_overlap") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g %g %g", 
                &(stage->gridsearch_min_overlap[0]), 
                &(stage->gridsearch_min_overlap[1]), 
                &(stage->gridsearch_min_overlap[2])) != 3) {
            goto error_exit;
        }
    }
    else if (key == "landmark_stiffness") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g", &stage->landmark_stiffness) != 1) {
            goto error_exit;
        }
    }   
    else if (key == "landmark_flavor") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%c", &stage->landmark_flavor) != 1) {
            goto error_exit;
        }
    }   
    else if (key == "res" || key == "ss") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        stage->subsampling_type = SUBSAMPLING_VOXEL_RATE;
        if (sscanf (val.c_str(), "%g %g %g", 
                &(stage->fixed_subsample_rate[0]), 
                &(stage->fixed_subsample_rate[1]), 
                &(stage->fixed_subsample_rate[2])) != 3) {
            goto error_exit;
        }
        stage->moving_subsample_rate[0] = stage->fixed_subsample_rate[0];
        stage->moving_subsample_rate[1] = stage->fixed_subsample_rate[1];
        stage->moving_subsample_rate[2] = stage->fixed_subsample_rate[2];
    }
    else if (key == "ss_fixed" || key == "fixed_ss") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g %g %g", 
                &(stage->fixed_subsample_rate[0]), 
                &(stage->fixed_subsample_rate[1]), 
                &(stage->fixed_subsample_rate[2])) != 3) {
            goto error_exit;
        }
        if (stage->subsampling_type == SUBSAMPLING_AUTO) {
            stage->moving_subsample_rate[0] = stage->fixed_subsample_rate[0];
            stage->moving_subsample_rate[1] = stage->fixed_subsample_rate[1];
            stage->moving_subsample_rate[2] = stage->fixed_subsample_rate[2];
        }
        stage->subsampling_type = SUBSAMPLING_VOXEL_RATE;
    }
    else if (key == "ss_moving" || key == "moving_ss") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g %g %g", 
                &(stage->moving_subsample_rate[0]), 
                &(stage->moving_subsample_rate[1]), 
                &(stage->moving_subsample_rate[2])) != 3) {
            goto error_exit;
        }
        if (stage->subsampling_type == SUBSAMPLING_AUTO) {
            stage->fixed_subsample_rate[0] = stage->moving_subsample_rate[0];
            stage->fixed_subsample_rate[1] = stage->moving_subsample_rate[1];
            stage->fixed_subsample_rate[2] = stage->moving_subsample_rate[2];
        }
        stage->subsampling_type = SUBSAMPLING_VOXEL_RATE;
    }
    else if (key == "sampling_rate" || key == "sr") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        stage->subsampling_type = SUBSAMPLING_VOXEL_RATE;
        if (sscanf (val.c_str(), "%g %g %g", 
                &(stage->fixed_subsample_rate[0]), 
                &(stage->fixed_subsample_rate[1]), 
                &(stage->fixed_subsample_rate[2])) != 3) {
            goto error_exit;
        }
        stage->moving_subsample_rate[0] = stage->fixed_subsample_rate[0];
        stage->moving_subsample_rate[1] = stage->fixed_subsample_rate[1];
        stage->moving_subsample_rate[2] = stage->fixed_subsample_rate[2];
    }
    else if (key == "num_grid") {
        /* Obsolete */
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%d %d %d", 
                &(stage->num_grid[0]), 
                &(stage->num_grid[1]), 
                &(stage->num_grid[2])) != 3) {
            goto error_exit;
        }
        stage->grid_method = 0;
    }
    else if (key == "grid_spac"
        || key == "grid_spacing")
    {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%g %g %g", 
                &(stage->grid_spac[0]), 
                &(stage->grid_spac[1]), 
                &(stage->grid_spac[2])) != 3) {
            goto error_exit;
        }
        stage->grid_method = 1;
    }
    else if (key == "histo_equ") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (string_value_true (val)) {
            stage->histoeq = true;
        }
        else
            stage->histoeq= false;
    }
    else if (key == "thresh_mean_intensity") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (string_value_true (val)) {
            stage->thresh_mean_intensity = true;
        }
        else
            stage->thresh_mean_intensity= false;
    }
    else if (key == "num_hist_levels") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%d", &stage->num_hist_levels) != 1) {
            goto error_exit;
        }
    }
    else if (key == "num_matching_points") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        if (sscanf (val.c_str(), "%d", &stage->num_matching_points) != 1) {
            goto error_exit;
        }
    }
    else if (key == "debug_dir") {
        if (!section_stage) goto key_only_allowed_in_section_stage;
        stage->debug_dir = val;
    }

    /* The following keywords are only allowed in process section */
    else if (section_process) {
        Process_parms::Pointer pp = stage->get_process_parms ();
        if (key == "action") {
            pp->set_action (val);
        } else {
            pp->set_key_value (key, val);
        }
    }

    else {
        goto error_exit;
    }
    return 0;

key_only_allowed_in_section_global:
    print_and_exit (
        "This key (%s) is only allowed in a global section\n", key.c_str());
    return -1;

key_only_allowed_in_section_stage:
    print_and_exit (
        "This key (%s) is only allowed in a stage section\n", key.c_str());
    return -1;

key_not_allowed_in_section_process:
    print_and_exit (
        "This key (%s) not is allowed in a process section\n", key.c_str());
    return -1;

error_exit:
    print_and_exit (
        "Unknown (key,val) combination: (%s,%s)\n", key.c_str(), val.c_str());
    return -1;
}

int
Registration_parms::set_command_string (
    const std::string& command_string
)
{
    Registration_parms_parser rpp (this);
    return rpp.parse_config_string (command_string);
}

int
Registration_parms::parse_command_file (const char* options_fn)
{
    /* Read file into string */
    std::ifstream t (options_fn);
    std::stringstream buffer;
    buffer << t.rdbuf();

    /* Parse the string */
    return this->set_command_string (buffer.str());
}

/* JAS 2012.03.13
 *  This is a temp solution */
/* GCS 2012-12-28: Nb. regp->job_idx must be set prior to calling 
   this function */
void
Registration_parms::set_job_paths (void)
{
    /* Setup input paths */
    if (*(this->fixed_dir)) {
        d_ptr->fixed_fn = string_format (
            "%s%s", this->fixed_dir, this->fixed_jobs[this->job_idx]);
    }
    if (*(this->moving_dir)) {
        d_ptr->moving_fn = string_format (
            "%s%s", this->moving_dir, this->moving_jobs[this->job_idx]);
    }

    /* Setup output paths */
    /*   NOTE: For now, output files inherit moving image names */
    if (*(this->img_out_dir)) {
        if (!strcmp (this->img_out_dir, this->moving_dir)) {
            strcpy (this->img_out_fn, this->img_out_dir);
            strcat (this->img_out_fn, "warp/");
            strcat (this->img_out_fn, this->moving_jobs[this->job_idx]);
        } else {
            strcpy (this->img_out_fn, this->img_out_dir);
            strcat (this->img_out_fn, this->moving_jobs[this->job_idx]);
        }
        /* If not dicom, we give a default name */
        if (this->img_out_fmt != IMG_OUT_FMT_DICOM) {
            std::string fn = string_format ("%s.mha", this->img_out_fn);
            strcpy (this->img_out_fn, fn.c_str());
        }
    } else {
        /* Output directory not specifed but img_out was... smart fallback*/
        if (*(this->img_out_fn)) {
            strcpy (this->img_out_fn, this->moving_dir);
            strcat (this->img_out_fn, "warp/");
            strcat (this->img_out_fn, this->moving_jobs[this->job_idx]);
        }
    }
    if (*(this->vf_out_dir)) {
        if (!strcmp (this->vf_out_dir, this->moving_dir)) {
            strcpy (this->vf_out_fn, this->img_out_dir);
            strcat (this->vf_out_fn, "vf/");
            strcat (this->vf_out_fn, this->moving_jobs[this->job_idx]);
        } else {
            strcpy (this->vf_out_fn, this->vf_out_dir);
            strcat (this->vf_out_fn, this->moving_jobs[this->job_idx]);
        }
        /* Give a default name */
        std::string fn = string_format ("%s_vf.mha", this->vf_out_fn);
        strcpy (this->vf_out_fn, fn.c_str());
    } else {
        /* Output directory not specifed but vf_out was... smart fallback*/
        if (*(this->vf_out_fn)) {
            strcpy (this->vf_out_fn, this->moving_dir);
            strcat (this->vf_out_fn, "vf/");
            strcat (this->vf_out_fn, this->moving_jobs[this->job_idx]);
        }
    }
}

const std::string& 
Registration_parms::get_fixed_fn ()
{
    return d_ptr->fixed_fn;
}

const std::string& 
Registration_parms::get_moving_fn ()
{
    return d_ptr->moving_fn;
}

Shared_parms*
Registration_parms::get_shared_parms ()
{
    return d_ptr->shared;
}

std::list<Stage_parms*>& 
Registration_parms::get_stages ()
{
    return d_ptr->stages;
}

Stage_parms* 
Registration_parms::append_stage ()
{
    Stage_parms *sp;

    this->num_stages ++;
    if (this->num_stages == 1) {
        sp = new Stage_parms();
    } else {
        sp = new Stage_parms(*d_ptr->stages.back());
    }
    d_ptr->stages.push_back (sp);

    /* Some parameters that should be copied from global 
       to the first stage. */
    if (this->num_stages == 1) {
        sp->default_value = this->default_value;
    }

    sp->stage_no = this->num_stages;

    return sp;
}

Stage_parms* 
Registration_parms::append_process_stage ()
{
    Stage_parms *sp = this->append_stage ();

    Process_parms::Pointer pp = Process_parms::New ();
    sp->set_process_parms (pp);
    return sp;
}
