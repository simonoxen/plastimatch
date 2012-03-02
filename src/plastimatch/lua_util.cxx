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
#include "lua_util.h"

/* utility functions */
int
lua_check_type (lua_State *L, const char* class_name, int index)
{
    void *p = lua_touserdata (L, index);
    if (p) {
        if (lua_getmetatable (L, index)) {
            lua_getfield (L, LUA_REGISTRYINDEX, class_name);
            if (lua_rawequal(L, -1, -2)) {
                lua_pop(L, 2);
                return 1;
            }
        }
    }
    return 0;
}

void*
lua_new_instance (lua_State *L, const char* class_name, size_t size)
{
    void *tmp = (void*)lua_newuserdata (L, size);
    luaL_getmetatable (L, class_name);
    lua_setmetatable (L, -2);
    return tmp;
}

void*
get_obj_ptr (lua_State *L, const char* class_name, int index)
{
    void* ptr;
    luaL_checktype (L, index, LUA_TUSERDATA);
    ptr = (void*)luaL_checkudata (L, index, class_name);
    if (ptr == NULL) {
        luaL_typerror (L, index, class_name);
    }
    return ptr;
}

int
register_lua_class (
    lua_State *L,
    const char* class_name,
    const luaL_reg* methods,
    const luaL_reg* metatable
)
{
    /* create methods table, add it to the globals */
    luaL_openlib (L, class_name, methods, 0);

    /* create metatable and add it to the Lua registry */
    luaL_newmetatable (L, class_name);

    /* fill metatable */
    luaL_openlib (L, 0, metatable, 0);

    /* direct calls to non-existant members to methods table */
    lua_pushliteral (L, "__index");
    lua_pushvalue (L, -3);               /* push methods table */
    lua_rawset (L, -3);                  /* metatable.__index = methods */

    /* protect our metatable */
    lua_pushliteral (L, "__metatable");
    lua_pushvalue (L, -3);               /* push methods table */
    lua_rawset (L, -3);                  /* metatable.__metatable = methods */

    lua_pop (L, 1);                      /* pop metatable */
    lua_pop (L, 1);                      /* pop methods */
    return 0;
}
