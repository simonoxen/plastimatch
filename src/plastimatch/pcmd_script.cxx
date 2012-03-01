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
#include "lua_iface_add.h"
#include "lua_iface_crop.h"
#include "lua_iface_mask.h"     /* also contains fill() */
#include "lua_iface_register.h"
#include "lua_iface_resample.h"
#include "lua_iface_synth.h"
#include "lua_class_image.h"


// JAS 2012.02.2 - Planned for removal in favor of
//                 more elegant Lua objects and classes
//-------------------------------------------------------------------
/* Command Line Glue */
void
lua_cli_glue_init (lua_State* L, int* argc, char*** argv)
{
    /* # of parameters passed via lua stack + 2 */
    /*     1 for argv[0] = "plastimatch"        */
    /*     1 for argv[1] = specified pcmd       */
    luaL_checktype(L, 1, LUA_TTABLE);
    *argc = 2*from_lua_count_struct_members (L) + 2;
    *argv = (char**)malloc (*argc * sizeof(char*));

    for (int i=0; i<*argc; i++) {
        (*argv)[i] = NULL;
    }
    
    (*argv)[0] = (char*)malloc (strlen("plastimatch") * sizeof(char)); 
    strcpy ((*argv)[0], "plastimatch");
}

void
lua_cli_glue_grow (lua_State* L, int n, int* argc, char*** argv)
{
    char** p = (char**)realloc (*argv, (*argc+n) * sizeof(char*));
    if (p) { *argv = p; }

    for (int i=*argc; i<(*argc)+n; i++) {
        (*argv)[i] = NULL;
    }
    *argc += n;
}

void
lua_cli_glue_add (lua_State* L, char* arg, char** argv)
{
    if (*argv == NULL) {
        *argv = (char*)malloc (strlen(arg) * sizeof(char)); 
    } else {
        char* p = (char*)realloc (*argv, strlen(arg) * sizeof(char));
        if (p) { *argv = p; }
    }
    strcpy (*argv, arg);
}

void
lua_cli_glue_solvent (lua_State* L, char** argv, int argn)
{
    for (int i=0; i<argn; i++) {
        free (argv[i]);
    }
    free (argv);
}
//-------------------------------------------------------------------



//-------------------------------------------------------------------
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
//-------------------------------------------------------------------


/* LUA Helpers */
/* Returns the # of members for a table organized
 * like a C struct.  Must be on the top of the lua stack */
int
from_lua_count_struct_members (lua_State* L)
{
    int n=0;
    lua_pushnil (L);
    while (lua_next (L, -2)) {
        luaL_checktype (L, 1, LUA_TTABLE);
        n++;
        lua_pop (L, 1);
    }
    return n;
}

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

void
replace_char (char from, char to, char* str)
{
    int n=0;
    while (str[n] != '\0') {
        if (str[n] == from) {
            str[n] = to;
        }
        n++;
    }
}



/* Register your LUA interface here */
static void
register_lua_interfaces (lua_State* L)
{
    lua_register (L, "add",      LUAIFACE_add);
    lua_register (L, "crop",     LUAIFACE_crop);
    lua_register (L, "mask",     LUAIFACE_mask);
    lua_register (L, "fill",     LUAIFACE_fill);
    lua_register (L, "register", LUAIFACE_register);
    lua_register (L, "resample", LUAIFACE_resample);
    lua_register (L, "synth",    LUAIFACE_synth);
}

static void
register_lua_objects (lua_State* L)
{
    register_lua_class_image (L);
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
    register_lua_objects (L);
    luaL_dofile (L, script_fn);
    lua_close (L);
    printf ("\n[Script Terminated]\n\n");
}
