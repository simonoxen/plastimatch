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
#include "pcmd_script.h"
#include "plm_image.h"
#include "volume.h"

/* Name of class as exposed to Lua */
#define CLASS_NAME "image"

/*-----------------------------------------------------*/
/*  Utility Functions                                  */
/*-----------------------------------------------------*/
/*-----------------------------------------------------*/



/*******************************************************/
/* Object Methods                                      */
/*******************************************************/
static int
image_load (lua_State *L)
{
    const char* fn = luaL_optlstring (L, 1, NULL, NULL);
    
    if (!fn) {
        fprintf (stderr, "error -- image.load() -- no file specified\n");
        return 0;
    }

    Plm_image* pli = plm_image_load (fn, PLM_IMG_TYPE_ITK_FLOAT);
    if (!pli) {
        fprintf (stderr, "error -- image.load() -- unable to load %s\n", fn);
        return 0;
    }

    lua_image *tmp = (lua_image*)lua_new_instance (L,
                                    CLASS_NAME,
                                    sizeof(lua_image));

    tmp->fn = fn;
    tmp->pli = pli;
    tmp->vol = pli->gpuit_float();

    return 1;
}

static int
image_save (lua_State *L)
{
    lua_image *limg = (lua_image*)get_obj_ptr (L, CLASS_NAME, 1);
    const char* fn = luaL_optlstring (L, 2, NULL, NULL);
    Plm_image *pli = limg->pli;
    
    fprintf (stderr, "debug -- image.save() -- called\n");
    if (!fn) {
        /* Save over current volume on disk */
        fprintf (stderr, "debug -- image.save() -- overwriting\n");
        pli->save_image (limg->fn);
    } else {
        /* "Save-As" new volume on disk */
        fprintf (stderr, "debug -- image.save() -- save-as\n");
        pli->save_image (fn);
    }

    return 0;
}

/*******************************************************/



/*-----------------------------------------------------*/
/* Object Registration & Creation                      */
/*-----------------------------------------------------*/

/* methods table for object */
static const luaL_reg
image_methods[] = {
  {"load",          image_load},
  {"save",          image_save},
  {0, 0}
};

//-------------------------------------------------------
/* Action: garbage collection */
static int
image_gc (lua_State *L)
{
    lua_image *tmp = (lua_image*)get_obj_ptr (L, CLASS_NAME, 1);

    fprintf (stderr, "debug -- unloading %s\n", tmp->fn);
    delete tmp->pli;

    return 0;
}

/* Metatable of Actions */
static const luaL_reg image_meta[] = {
  {"__gc",       image_gc},
  {0, 0}
};
//-------------------------------------------------------


/* Object Creation */
int
register_lua_class_image (lua_State *L)
{
    return register_lua_class (L, CLASS_NAME, image_methods, image_meta);
}
