/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _lua_class_structs_h_
#define _lua_class_structs_h_

#include "plmscript_config.h"
#include "plm_path.h"
#include "rtss.h"

#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"

typedef struct lua_ss_struct lua_ss;
struct lua_ss_struct {
    char fn[_MAX_PATH];
    Plm_image* ss_img;
};

int register_lua_class_ss (lua_State *L);
void init_ss_instance (lua_ss* lss);

#endif
