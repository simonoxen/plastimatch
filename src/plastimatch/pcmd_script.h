/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_script_h_
#define _pcmd_script_h_

#include "plm_config.h"

#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"


int from_lua_getstring (lua_State* L, char* dest, const char* var_name);
int from_lua_getint (lua_State* L, int* dest, const char* var_name);
int from_lua_getfloat (lua_State* L, float* dest, const char* var_name);
int from_lua_getint3 (lua_State* L, int* dest, const char* var_name);
int from_lua_getfloat3 (lua_State* L, float* dest, const char* var_name);

void do_command_script (int argc, char *argv[]);

#endif
