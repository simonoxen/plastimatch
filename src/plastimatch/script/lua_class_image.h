/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _lua_class_image_h_
#define _lua_class_image_h_

#include "plmscript_config.h"
#include "plm_image.h"
#include "plm_path.h"

#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"

typedef struct lua_image_struct lua_image;
struct lua_image_struct {
    char fn[_MAX_PATH];
    Plm_image* pli;
};

int register_lua_class_image (lua_State *L);
void init_image_instance (lua_image* limg);

#endif
