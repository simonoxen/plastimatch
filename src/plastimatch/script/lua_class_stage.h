/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _lua_class_stage_h_
#define _lua_class_stage_h_

#include "plmscript_config.h"
#include "plm_image.h"
#include "plm_path.h"

#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"

#define LUA_CLASS_STAGE    "stage"

typedef struct lua_stage_struct lua_stage;
struct lua_stage_struct {
    bool active;
    int foo;
    int bar;
};

int register_lua_class_stage (lua_State *L);
void init_stage_instance (lua_stage* lstage);

#endif
