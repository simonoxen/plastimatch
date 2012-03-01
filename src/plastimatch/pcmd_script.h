/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_script_h_
#define _pcmd_script_h_

#include "plm_config.h"

#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"

/* exposed class names */
#define LUA_CLASS_IMAGE "image"
#define LUA_CLASS_XFORM "xform"


/* Old CLI Glue */
void lua_cli_glue_init (lua_State* L, int* argc, char*** argv);
void lua_cli_glue_grow (lua_State* L, int n, int* argc, char*** argv);
void lua_cli_glue_add (lua_State* L, char* arg, char** argv);
void lua_cli_glue_solvent (lua_State* L, char** argv, int argn);

/* Misc */
void replace_char (char from, char to, char* str);

/* Class Stuff */
int lua_check_type (lua_State *L, const char* class_name, int index);
void* lua_new_instance (lua_State *L, const char* class_name, size_t size);
void* get_obj_ptr (lua_State *L, const char* class_name, int index);
int register_lua_class (
    lua_State *L,
    const char* class_name,
    const luaL_reg* methods,
    const luaL_reg* metatable
);

/* Helpers */
int from_lua_count_struct_members (lua_State* L);
int from_lua_getstring (lua_State* L, char* dest, const char* var_name);
int from_lua_getint (lua_State* L, int* dest, const char* var_name);
int from_lua_getfloat (lua_State* L, float* dest, const char* var_name);
int from_lua_getint3 (lua_State* L, int* dest, const char* var_name);
int from_lua_getfloat3 (lua_State* L, float* dest, const char* var_name);

void do_command_script (int argc, char *argv[]);

#endif
