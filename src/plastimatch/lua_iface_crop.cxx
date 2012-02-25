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
#include "pcmd_crop.h"


// USAGE INSIDE LUA:
//    
//    parms = {
//       input  = "in.mha",
//       output = "out.mha",
//       voxels = "x-min x-max y-min y-max z-min z-max"
//    }
//
//    crop (parms)
//
int
LUAIFACE_crop (lua_State* L)
{
    char** argv;
    int argc;
    int argn;  /* # of struct entries (!necessarily = argc) */

    lua_cli_glue_init (L, &argn, &argv);
    lua_cli_glue_add  (L, "crop", &argv[1]);

    char arg[_MAX_PATH];
    char buf[_MAX_PATH];

    char* opt[] = {
        "output",
        "input",
        "voxels"
    };
    int num_opt = sizeof (opt)/sizeof (char*);

    argc=2;
    for (int i=0; i<num_opt; i++) {
        if (from_lua_getstring (L, arg, opt[i])) {
            sprintf (buf, "--%s", opt[i]);
            lua_cli_glue_add (L, buf, &argv[argc++]);
            lua_cli_glue_add (L, arg, &argv[argc++]);
        }
    }
#if 0
    for (int i=0; i<argc; i++) {
        printf ("argv[%i] = %s\n", i, argv[i]);
    }
#endif

    do_command_crop (argc, argv);

    lua_cli_glue_solvent (L, argv, argn);

    lua_pushnumber (L, 0);
    return 1; // # of return values
}
