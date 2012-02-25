#include <stdlib.h>
#include <stdio.h>
#include <string.h>

extern "C"
{
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
}

#include "plm_path.h"
#include "pcmd_script.h"
#include "pcmd_add.h"

// USAGE INSIDE LUA:
//    
//    parms = {
//       output = "out.mha",
//       weight = "0.4 0.3 0.3"
//    }
//
//    vols = {
//       "input_a.mha",
//       "input_b.mha",
//       "input_c.mha"
//    }
//
//    add (vols, parms)
//
int
LUAIFACE_add (lua_State* L)
{
    char** argv;
    int argc;
    int argn;  /* # of struct entries (!necessarily = argc) */

    lua_cli_glue_init (L, &argn, &argv);
    lua_cli_glue_add  (L, "add", &argv[1]);

    char arg[_MAX_PATH];
    char buf[_MAX_PATH];

    char* mask_opt[] = {
        "output",
        "weight"
    };
    int num_mask_opt = sizeof (mask_opt)/sizeof (char*);

    // Process parms
    argc=2;
    for (int i=0; i<num_mask_opt; i++) {
        if (from_lua_getstring (L, arg, mask_opt[i])) {
            sprintf (buf, "--%s", mask_opt[i]);
            lua_cli_glue_add (L, buf, &argv[argc++]);
            lua_cli_glue_add (L, arg, &argv[argc++]);
        }
    }

    // Pop the parms and get the input volumes
    lua_pop (L, 1);
    int num_input = from_lua_count_struct_members (L);
    lua_cli_glue_grow (L, num_input, &argn, &argv);

    lua_pushnil (L);
    while (lua_next (L, -2)) {
        luaL_checktype (L, 1, LUA_TTABLE);
        strcpy (buf, lua_tostring (L, -1));
        lua_cli_glue_add (L, buf, &argv[argc++]);
        lua_pop (L, 1);
    }

    do_command_add (argc, argv);

    lua_cli_glue_solvent (L, argv, argn);

    lua_pushnumber (L, 0);
    return 1; // # of return values
}
