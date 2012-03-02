/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
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
#include "lua_class_image.h"
#include "lua_class_xform.h"
#include "lua_util.h"
#include "pcmd_script.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_warp.h"
#include "xform.h"

/* Name of class as exposed to Lua */
#define THIS_CLASS LUA_CLASS_XFORM

/* Helpers */
void
init_xform_instance (lua_xform* lxf)
{
    memset (lxf->fn, '\0', sizeof(lxf->fn));
    lxf->pxf = NULL;
}


/*******************************************************/
/* Object Methods                                      */
/*******************************************************/
static int
xform_load (lua_State *L)
{
    const char* fn = luaL_optlstring (L, 1, NULL, NULL);
    
    if (!fn) {
        fprintf (stderr, "error -- xform.load() -- no file specified\n");
        return 0;
    }

    Xform* pxf = new Xform;
    xform_load (pxf, fn);

    if (!pxf) {
        fprintf (stderr, "error -- xform.load() -- unable to load %s\n", fn);
        return 0;
    }

    lua_xform *lxf = (lua_xform*)lua_new_instance (L,
                                    THIS_CLASS,
                                    sizeof(lua_xform));

    init_xform_instance (lxf);

    strcpy (lxf->fn, fn);
    lxf->pxf = pxf;

    return 1;
}

static int
xform_save (lua_State *L)
{
    lua_xform *lxf = (lua_xform*)get_obj_ptr (L, THIS_CLASS, 1);

    const char* fn = luaL_optlstring (L, 2, NULL, NULL);

    if (!fn) {
        if (lxf->fn[0] == '\0') {
            fprintf (stderr, "warning -- xform:save() -- filename must be specified when saving derived transforms\n");
            return 0;
        } else {
            /* Save over current transform on disk */
            xform_save (lxf->pxf, lxf->fn);
        }
    } else {
        /* "Save-As" new transform on disk */
        xform_save (lxf->pxf, fn);
        strcpy (lxf->fn, fn);
    }

    return 0;
}

static int
xform_save_vf (lua_State *L)
{
    lua_xform *lxf = (lua_xform*)get_obj_ptr (L, THIS_CLASS, 1);

    const char* fn = luaL_optlstring (L, 2, NULL, NULL);

    if (!fn) {
        fprintf (stderr, "warning -- xform:save_vf() -- filename must be specified\n");
        return 0;
    }

    // TODO
    fprintf (stderr, "warning -- xform:save_vf() -- developer too tired to implement feature\n");

    return 0;
}
/*******************************************************/


/*+++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/* Object Actions                                      */
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++*/

/* Action: garbage collection */
static int
xform_action_gc (lua_State *L)
{
    lua_xform *lxf = (lua_xform*)get_obj_ptr (L, THIS_CLASS, 1);
//    fprintf (stderr, "debug -- xform.__gc -- releasing %s [%p]\n", lxf->fn, lxf);
    delete lxf->pxf;
    return 0;
}

/* Action: add 
 *   [1] xform + image = warp
 *   [2] ????? + ????? = ????
 */
static int
xform_action_add (lua_State *L)
{
    lua_xform* lxf;
    lua_image* limg_in;
    lua_image* limg_out;
    Plm_image_header pih;

    /* xform * image */
    if ( (lua_check_type (L, THIS_CLASS, 1) && lua_check_type (L, LUA_CLASS_IMAGE, 2)) ||
         (lua_check_type (L, THIS_CLASS, 2) && lua_check_type (L, LUA_CLASS_IMAGE, 1))
    ) {

        /* Load the parms, but which is which? */
        if (lua_check_type (L, THIS_CLASS, 1)) {
            lxf     = (lua_xform*)get_obj_ptr (L, THIS_CLASS, 1);
            limg_in = (lua_image*)get_obj_ptr (L, LUA_CLASS_IMAGE, 2);
        }
        else if (lua_check_type (L, LUA_CLASS_IMAGE, 1)) {
            lxf     = (lua_xform*)get_obj_ptr (L, THIS_CLASS, 2);
            limg_in = (lua_image*)get_obj_ptr (L, LUA_CLASS_IMAGE, 1);
        }
        else {
            fprintf (stderr, "internal error -- xform.__add() -- please file bug report\n");
            exit (0);
        }

        limg_out =
            (lua_image*)lua_new_instance (L, LUA_CLASS_IMAGE, sizeof(lua_image));

        init_image_instance (limg_out);
        limg_out->pli = new Plm_image;
        pih.set_from_plm_image (limg_in->pli);

        plm_warp (
            limg_out->pli,  /* output image       */
            NULL,           /* output vf          */
            lxf->pxf,       /* xform              */
            &pih,           /* ouput geometry     */
            limg_in->pli,   /* input image        */
            -1000,          /* default hu value   */
            0,              /* 1: force ITK warp  */
            1               /* 1: Trilinear 0: nn */
        );

        return 1;
    }
    /* unsupported operand */
    else {
        fprintf (stderr, "warning -- xform.__add() -- invalid operand: returning (nil)\n");
        return 0;
    }
}
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++*/






/* Object Creation */

/* methods table for object */
static const luaL_reg
xform_methods[] = {
  {"load",          xform_load},
  {"save",          xform_save},
  {0, 0}
};

/* metatable of actions */
static const luaL_reg
xform_meta[] = {
  {"__gc",       xform_action_gc},
  {"__add",      xform_action_add},
  {0, 0}
};

int
register_lua_class_xform (lua_State *L)
{
    return register_lua_class (L, THIS_CLASS, xform_methods, xform_meta);
}
