/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _lua_object_image_h_
#define _lua_object_image_h_

#include "plm_config.h"
#include "plm_image.h"

#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"

typedef struct lua_image_struct lua_image;
struct lua_image_struct {
    const char* fn;
    Plm_image* pli;
    Volume* vol;
};

int register_lua_class_image (lua_State *L);

#endif
