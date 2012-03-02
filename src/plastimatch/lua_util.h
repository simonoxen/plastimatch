/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _lua_util_h_
#define _lua_util_h_

#include "plm_config.h"

#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"

int lua_check_type (lua_State *L, const char* class_name, int index);
void* lua_new_instance (lua_State *L, const char* class_name, size_t size);
void* get_obj_ptr (lua_State *L, const char* class_name, int index);
int register_lua_class (
    lua_State *L,
    const char* class_name,
    const luaL_reg* methods,
    const luaL_reg* metatable
);



#endif
