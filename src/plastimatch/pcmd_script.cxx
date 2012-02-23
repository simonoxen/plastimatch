#include "plm_config.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
extern "C"
{
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
}
#include "pcmd_script.h"
#include "plm_parms.h"
#include "plm_stages.h"


/* ----------- LUA HELPERS ------------- */
inline int
from_lua_getstring (lua_State* L, char* dest, const char* var_name)
{
    lua_pushstring (L, var_name);
    lua_gettable (L, -2);

    if (lua_tostring (L, -1)) {
        strcpy (dest, lua_tostring (L, -1));
        lua_pop (L, 1);
        return 1;
    } else {
        lua_pop (L, 1);
        return 0;
    }
}

inline int
from_lua_getint (lua_State* L, int* dest, const char* var_name)
{
    lua_pushstring (L, var_name);
    lua_gettable (L, -2);

    if (lua_tointeger (L, -1)) {
        *dest = lua_tointeger (L, -1);
        lua_pop (L, 1);
        return 1;
    } else {
        lua_pop (L, 1);
        return 0;
    }
}

inline int
from_lua_getfloat (lua_State* L, float* dest, const char* var_name)
{
    lua_pushstring (L, var_name);
    lua_gettable (L, -2);

    if (lua_isnumber (L, -1)) {
        *dest = lua_tonumber (L, -1);
        lua_pop (L, 1);
        return 1;
    } else {
        lua_pop (L, 1);
        return 0;
    }
}


inline int
from_lua_getint3 (lua_State* L, int* dest, const char* var_name)
{
    /* Get nested table by outer table key */
    lua_pushstring (L, var_name);
    lua_gettable (L, -2);   /* outer table */

    /* Does nested table exist? */
    if (!lua_istable (L, -1)) {
        lua_pop (L, 1);     /* outer table */
        return 0;
    }

    /* Iterate through nested table */
    for (int i=0; i<3; i++) {
        lua_pushnumber (L, i+1);
        lua_gettable (L, -2);   /* inner table */
        if (lua_isnumber (L, -1)) {
            dest[i] = lua_tointeger (L, -1);
        } else {
            lua_pop (L, 1); /* inner table */
            lua_pop (L, 1); /* outer table */
            return 0;
        }
        lua_pop (L, 1);  /* inner table */
    }
    lua_pop (L, 1); /* outer table */

    return 1;
}

inline int
from_lua_getfloat3 (lua_State* L, float* dest, const char* var_name)
{
    /* Get nested table by outer table key */
    lua_pushstring (L, var_name);
    lua_gettable (L, -2);   /* outer table */

    /* Does nested table exist? */
    if (!lua_istable (L, -1)) {
        lua_pop (L, 1);     /* outer table */
        return 0;
    }

    /* Iterate through nested table */
    for (int i=0; i<3; i++) {
        lua_pushnumber (L, i+1);
        lua_gettable (L, -2);   /* inner table */
        if (lua_isnumber (L, -1)) {
            dest[i] = lua_tonumber (L, -1);
        } else {
            lua_pop (L, 1); /* inner table */
            lua_pop (L, 1); /* outer table */
            return 0;
        }
        lua_pop (L, 1);  /* inner table */
    }
    lua_pop (L, 1); /* outer table */

    return 1;
}

int
lua_abort_global (lua_State* L, const char* bad_key)
{
    fprintf (stderr, "error -- global -- bad value specifed for %s -- aborted\n", bad_key);
    lua_pushnumber (L, -1);
    return 1; // # of return values
}

int
lua_abort_stage (lua_State* L, int num, const char* bad_key)
{
    fprintf (stderr, "error -- stage %i -- bad value specifed for %s -- aborted\n", num, bad_key);
    lua_pushnumber (L, -1);
    return 1; // # of return values
}
/* ------------------------------------- */


extern "C" int
PAPI_register (lua_State* L)
{
    Registration_parms regp;

    /* Get # of args */
    int argc = lua_gettop (L);
    regp.num_stages = argc-1;

    if (regp.num_stages < 1) {
        fprintf (stderr, "error -- no stages specifed for registration -- aborted\n");
        lua_pushnumber (L, -1);
        return 1;
    }


    char ret[255];
    int ret_int = 0;
    float ret_float;
    float ret_float3[3];

    /* Global Parms */
    from_lua_getstring (L, regp.fixed_fn,            "fixed");
    from_lua_getstring (L, regp.moving_fn,           "moving");
    from_lua_getstring (L, regp.fixed_mask_fn,       "fixed_mask");
    from_lua_getstring (L, regp.moving_mask_fn,      "moving_mask");
    from_lua_getstring (L, regp.xf_in_fn,            "xf_in");
    from_lua_getstring (L, regp.log_fn,              "log");
    from_lua_getstring (L, regp.img_out_fn,          "img_out");
    from_lua_getstring (L, regp.vf_out_fn,           "vf_out");

    if (from_lua_getstring (L, ret, "fixed_landmarks")) {
        regp.fixed_landmarks_fn = ret;
    }
    if (from_lua_getstring (L, ret, "moving_landmarks")) {
        regp.moving_landmarks_fn = ret;
    }
    if (from_lua_getstring (L, ret, "img_out_fmt")) {
        int fmt = IMG_OUT_FMT_AUTO;
        if (!strcmp (ret, "dicom")) {
            fmt = IMG_OUT_FMT_DICOM;
        } else {
            return lua_abort_global (L, "img_out_fmt");
        }
        regp.img_out_fmt = fmt;
    }
    if (from_lua_getstring (L, ret, "img_out_type")) {
        Plm_image_type type = plm_image_type_parse (ret);
        if (type == PLM_IMG_TYPE_UNDEFINED) {
            return lua_abort_global (L, "img_out_type");
        }
        regp.img_out_type = type;
    }
    if (from_lua_getstring (L, ret, "xf_out_itk")) {
        bool value = true;
        if (!strcmp (ret, "false")) {
            value = false;
        }
        regp.xf_out_itk = value;
    }
    if (from_lua_getstring (L, ret, "xf_out")) {
        /* xf_out is special.  You can have more than one of these.  
           This capability is used by the slicer plugin. */
        regp.xf_out_fn.push_back (ret);
    }
    if (from_lua_getstring (L, ret, "warped_landmarks")) {
        regp.warped_landmarks_fn = ret;
    }
    

#if 0
    printf ("   GLOBAL: fixed: %s\n", regp.fixed_fn);
    printf ("   GLOBAL: moving: %s\n", regp.moving_fn);
    printf ("   GLOBAL: fixed_mask: %s\n", regp.fixed_mask_fn);
    printf ("   GLOBAL: moving_mask: %s\n", regp.moving_mask_fn);
    printf ("   GLOBAL: xf_in: %s\n", regp.xf_in_fn);
    printf ("   GLOBAL: log: %s\n", regp.log_fn);
    printf ("   GLOBAL: img_out: %s\n", regp.img_out_fn);
    printf ("   GLOBAL: vf_out: %s\n", regp.vf_out_fn);
#endif

    regp.stages =
        (Stage_parms**)malloc (regp.num_stages * sizeof (Stage_parms*));
    for (int i=0; i<regp.num_stages; i++) {
    }


    Stage_parms* stage = NULL;
    for (int i=0; i<regp.num_stages; i++) {

        /* Build Stage Arrays */
        if (i == 0) {
            regp.stages[i] = new Stage_parms ();
        } else {
            regp.stages[i] = new Stage_parms (*(regp.stages[i-1]));
        }
        stage = regp.stages[i];

        stage->stage_no = i;


        /* Stage Parms */
        lua_pop (L, 1);

        if (from_lua_getstring (L, ret, "xform")) {
            if (!strcmp (ret,"translation")) {
                stage->xform_type = STAGE_TRANSFORM_TRANSLATION;
            }
            else if (!strcmp(ret,"rigid") || !strcmp(ret,"versor")) {
                stage->xform_type = STAGE_TRANSFORM_VERSOR;
            }
            else if (!strcmp (ret,"quaternion")) {
                stage->xform_type = STAGE_TRANSFORM_QUATERNION;
            }
            else if (!strcmp (ret,"affine")) {
                stage->xform_type = STAGE_TRANSFORM_AFFINE;
            }
            else if (!strcmp (ret,"bspline")) {
                stage->xform_type = STAGE_TRANSFORM_BSPLINE;
            }
            else if (!strcmp (ret,"vf")) {
                stage->xform_type = STAGE_TRANSFORM_VECTOR_FIELD;
            }
            else if (!strcmp (ret,"align_center")) {
                stage->xform_type = STAGE_TRANSFORM_ALIGN_CENTER;
            }
            else {
                return lua_abort_stage (L, i, "xform");
            }
        }
        if (from_lua_getstring (L, ret, "optim")) {
            if (!strcmp(ret,"none")) {
                stage->optim_type = OPTIMIZATION_NO_REGISTRATION;
            }
            else if (!strcmp(ret,"amoeba")) {
                stage->optim_type = OPTIMIZATION_AMOEBA;
            }
            else if (!strcmp(ret,"rsg")) {
                stage->optim_type = OPTIMIZATION_RSG;
            }
            else if (!strcmp(ret,"versor")) {
                stage->optim_type = OPTIMIZATION_VERSOR;
            }
            else if (!strcmp(ret,"lbfgs")) {
                stage->optim_type = OPTIMIZATION_LBFGS;
            }
            else if (!strcmp(ret,"lbfgsb")) {
                stage->optim_type = OPTIMIZATION_LBFGSB;
            }
            else if (!strcmp(ret,"liblbfgs")) {
                stage->optim_type = OPTIMIZATION_LIBLBFGS;
            }
            else if (!strcmp(ret,"demons")) {
                stage->optim_type = OPTIMIZATION_DEMONS;
            }
            else if (!strcmp(ret,"steepest")) {
                stage->optim_type = OPTIMIZATION_STEEPEST;
            }
            else {
                return lua_abort_stage (L, i, "optim");
            }
        }
        if (from_lua_getstring (L, ret, "impl")) {
            if (!strcmp(ret,"none")) {
                stage->impl_type = IMPLEMENTATION_NONE;
            }
            else if (!strcmp(ret,"itk")) {
                stage->impl_type = IMPLEMENTATION_ITK;
            }
            else if (!strcmp(ret,"plastimatch")) {
                stage->impl_type = IMPLEMENTATION_PLASTIMATCH;
            }
            else {
                return lua_abort_stage (L, i, "impl");
            }
        }
        if (from_lua_getstring (L, ret, "threading")) {
            if (!strcmp(ret,"single")) {
                stage->threading_type = THREADING_CPU_SINGLE;
            }
            else if (!strcmp(ret,"openmp")) {
#if (OPENMP_FOUND)
                stage->threading_type = THREADING_CPU_OPENMP;
#else
                stage->threading_type = THREADING_CPU_SINGLE;
#endif
            }
            else if (!strcmp(ret,"cuda")) {
#if (CUDA_FOUND)
                stage->threading_type = THREADING_CUDA;
#elif (OPENMP_FOUND)
                stage->threading_type = THREADING_CPU_OPENMP;
#else
                stage->threading_type = THREADING_CPU_SINGLE;
#endif
            }
            else {
                return lua_abort_stage (L, i, "threading");
            }
        }
        if (from_lua_getstring (L, ret, "flavor")) {
            if (strlen (ret) >= 1) {
                stage->alg_flavor = ret[0];
            } else {
                return lua_abort_stage (L, i, "flavor");
            }
        }
        if (from_lua_getstring (L, ret, "metric")) {
            if (!strcmp(ret,"mse") || !strcmp(ret,"MSE")) {
                stage->metric_type = METRIC_MSE;
            }
            else if (!strcmp(ret,"mi") || !strcmp(ret,"MI")) {
                stage->metric_type = METRIC_MI;
            }
            else if (!strcmp(ret,"mattes")) {
                stage->metric_type = METRIC_MI_MATTES;
            }
            else {
                return lua_abort_stage (L, i, "metric");
            }
        }
        if (from_lua_getstring (L, ret, "regularization")) {
            if (!strcmp(ret,"none")) {
                stage->regularization_type = REGULARIZATION_NONE;
            }
            else if (!strcmp(ret,"analytic")) {
                stage->regularization_type = REGULARIZATION_BSPLINE_ANALYTIC;
            }
            else if (!strcmp(ret,"semi-analytic")
                || !strcmp(ret,"semi_analytic")) {
                stage->regularization_type = REGULARIZATION_BSPLINE_SEMI_ANALYTIC;
            }
            else if (!strcmp(ret,"numeric")) {
                stage->regularization_type = REGULARIZATION_BSPLINE_NUMERIC;
            }
            else {
                return lua_abort_stage (L, i, "regularization");
            }
        }
        if (from_lua_getfloat (L, &ret_float, "regularization_lambda")) {
            stage->regularization_lambda = ret_float;
        }
        if (from_lua_getfloat (L, &ret_float, "background_val")) {
            stage->background_val = ret_float;
        }
        if (from_lua_getfloat (L, &ret_float, "background_max")) {
            stage->background_max = ret_float;
        }
        if (from_lua_getint (L, &ret_int, "min_its")) {
            stage->min_its = ret_int;
        }
        if (from_lua_getint (L, &ret_int, "max_its")) {
            stage->max_its = ret_int;
        }
        if (from_lua_getfloat3 (L, ret_float3, "grid_spac")) {
            stage->grid_spac[0] = ret_float3[0];
            stage->grid_spac[1] = ret_float3[1];
            stage->grid_spac[2] = ret_float3[2];
        }

#if 0
        printf ("   STAGE %i: xform: %i\n", i, stage->xform_type);
        printf ("   STAGE %i: optim: %i\n", i, stage->optim_type);
        printf ("   STAGE %i: impl:  %i\n", i, stage->impl_type);
        printf ("   STAGE %i: threading:  %i\n", i, stage->threading_type);
        printf ("   STAGE %i: metric:  %i\n", i, stage->metric_type);
        printf ("   STAGE %i: iterations:  %i\n", i, stage->max_its);
        printf ("   STAGE %i: grid_spac:  %f, %f, %f\n", i, stage->grid_spac[0], stage->grid_spac[1], stage->grid_spac[2]);
#endif
    
    }

    do_registration (&regp);

    lua_pushnumber (L, 0);
    return 1; // # of return values
}


static void
register_PAPI (lua_State* L)
{
    lua_register (L, "register", PAPI_register);
}


void
do_command_script (int argc, char *argv[])
{
    lua_State *L;
    char *script_fn = NULL;

    if (!strcmp (argv[1], "script")) {
        if (argc > 2) {
            script_fn = argv[2];
        } else {
            printf ("Usage: plastimatch script script_file\n");
            exit (1);
        }
    }

//    printf ("Opening: %s\n", script_fn);
    L = lua_open();
    luaL_openlibs(L);
    register_PAPI (L);
    luaL_dofile (L, script_fn);
    lua_close (L);
    printf ("\nDone.\n");
}
