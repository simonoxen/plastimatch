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
#include "pcmd_script.h"
#include "lua_iface_register.h"


/* LUA Helpers */
int
from_lua_getstring (lua_State* L, char* dest, const char* var_name)
{
    lua_pushstring (L, var_name);
    lua_gettable (L, -2);

    if (lua_tostring (L, -1)) {
        strcpy (dest, lua_tostring (L, -1));
        lua_pop (L, 1);
        return 1;
    } else {
        lua_pop (L, 1);
        return 0;
    }
}

int
from_lua_getint (lua_State* L, int* dest, const char* var_name)
{
    lua_pushstring (L, var_name);
    lua_gettable (L, -2);

    if (lua_tointeger (L, -1)) {
        *dest = lua_tointeger (L, -1);
        lua_pop (L, 1);
        return 1;
    } else {
        lua_pop (L, 1);
        return 0;
    }
}

int
from_lua_getfloat (lua_State* L, float* dest, const char* var_name)
{
    lua_pushstring (L, var_name);
    lua_gettable (L, -2);

    if (lua_isnumber (L, -1)) {
        *dest = lua_tonumber (L, -1);
        lua_pop (L, 1);
        return 1;
    } else {
        lua_pop (L, 1);
        return 0;
    }
}


int
from_lua_getint3 (lua_State* L, int* dest, const char* var_name)
{
    /* Get nested table by outer table key */
    lua_pushstring (L, var_name);
    lua_gettable (L, -2);   /* outer table */

    /* Does nested table exist? */
    if (!lua_istable (L, -1)) {
        lua_pop (L, 1);     /* outer table */
        return 0;
    }

    /* Iterate through nested table */
    for (int i=0; i<3; i++) {
        lua_pushnumber (L, i+1);
        lua_gettable (L, -2);   /* inner table */
        if (lua_isnumber (L, -1)) {
            dest[i] = lua_tointeger (L, -1);
        } else {
            lua_pop (L, 1); /* inner table */
            lua_pop (L, 1); /* outer table */
            return 0;
        }
        lua_pop (L, 1);  /* inner table */
    }
    lua_pop (L, 1); /* outer table */

    return 1;
}

int
from_lua_getfloat3 (lua_State* L, float* dest, const char* var_name)
{
    /* Get nested table by outer table key */
    lua_pushstring (L, var_name);
    lua_gettable (L, -2);   /* outer table */

    /* Does nested table exist? */
    if (!lua_istable (L, -1)) {
        lua_pop (L, 1);     /* outer table */
        return 0;
    }

    /* Iterate through nested table */
    for (int i=0; i<3; i++) {
        lua_pushnumber (L, i+1);
        lua_gettable (L, -2);   /* inner table */
        if (lua_isnumber (L, -1)) {
            dest[i] = lua_tonumber (L, -1);
        } else {
            lua_pop (L, 1); /* inner table */
            lua_pop (L, 1); /* outer table */
            return 0;
        }
        lua_pop (L, 1);  /* inner table */
    }
    lua_pop (L, 1); /* outer table */

    return 1;
}




/* Register your LUA interface here */
static void
register_lua_interfaces (lua_State* L)
{
    lua_register (L, "register", LUAIFACE_register);
}


/* Hook into plastmatch commandline */
void
do_command_script (int argc, char *argv[])
{
    lua_State *L;
    char *script_fn = NULL;

    if (!strcmp (argv[1], "script")) {
        if (argc > 2) {
            script_fn = argv[2];
        } else {
            printf ("Usage: plastimatch script script_file\n");
            exit (1);
        }
    }

//    printf ("Opening: %s\n", script_fn);
    L = lua_open();
    luaL_openlibs(L);
    register_lua_interfaces (L);
    luaL_dofile (L, script_fn);
    lua_close (L);
    printf ("\nDone.\n");
}
