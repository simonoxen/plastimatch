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
#include "pcmd_mask.h"

static int
mask_or_fill (lua_State* L, char* mode)
{
    char** argv;
    int argc;
    int argn;  /* # of struct entries (!necessarily = argc) */

    lua_cli_glue_init (L, &argn, &argv);
    lua_cli_glue_add  (L, mode, &argv[1]);

    char arg[_MAX_PATH];
    char buf[_MAX_PATH];

    char* opt[] = {
        "output",
        "input",
        "mask",
        "output_format",
        "output_type",
        "mask_value"
    };
    int num_opt = sizeof (opt)/sizeof (char*);

    argc=2;
    for (int i=0; i<num_opt; i++) {
        if (from_lua_getstring (L, arg, opt[i])) {
            sprintf (buf, "--%s", opt[i]);
            replace_char ('_', '-', buf);
            lua_cli_glue_add (L, buf, &argv[argc++]);
            lua_cli_glue_add (L, arg, &argv[argc++]);
        }
    }
#if 0
    for (int i=0; i<argc; i++) {
        printf ("argv[%i] = %s\n", i, argv[i]);
    }
#endif

    do_command_mask (argc, argv);

    lua_cli_glue_solvent (L, argv, argn);

    lua_pushnumber (L, 0);
    return 1; // # of return values
}






// USAGE INSIDE LUA:
//    
//    parms = {
//       output = "out.mha",
//       input  = "in.mha",
//             .
//             .
//             .
//    }
//
//    mask (parms)
//
int
LUAIFACE_mask (lua_State* L)
{
    return mask_or_fill (L, "mask");
}


// USAGE INSIDE LUA:
//    
//    parms = {
//       output = "out.mha",
//       input  = "in.mha",
//             .
//             .
//             .
//    }
//
//    fill(parms)
//
int
LUAIFACE_fill (lua_State* L)
{
    return mask_or_fill (L, "fill");
}

