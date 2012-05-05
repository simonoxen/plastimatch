/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _lua_class_xform_h_
#define _lua_class_xform_h_

#include "plmscript_config.h"
#include "plm_path.h"

#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"

class Xform;

typedef struct lua_xform_struct lua_xform;
struct lua_xform_struct {
    char fn[_MAX_PATH];
    Xform* pxf;
};

int register_lua_class_xform (lua_State *L);
void init_xform_instance (lua_xform* lxf);

#endif
