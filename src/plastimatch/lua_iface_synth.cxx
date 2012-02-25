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
#include "pcmd_synth.h"

// JAS 2012.02.25
//   This is currently an issue with variables containing
//   dashing not being valid in LUA.  One option is to
//   filter all - to _ and then convert back be before
//   constructing the command line parms.

// USAGE INSIDE LUA:
//    
//    parms = {
//           .
//           .
//    }
//
//    synth (parms)
//
int
LUAIFACE_synth (lua_State* L)
{
    char** argv;
    int argc;
    int argn;  /* # of struct entries (!necessarily = argc) */

    lua_cli_glue_init (L, &argn, &argv);
    lua_cli_glue_add  (L, "synth", &argv[1]);

    char arg[_MAX_PATH];
    char buf[_MAX_PATH];

    char* opt[] = {
        "output",
        "output-dicom",
        "output-dose-img",
        "output-ss-img",
        "output-type",
        "pattern",
        "origin",
        "dim",
        "spacing",
        "direction-cosines",
        "volume-size",
        "background",
        "foreground",
        "donut-center",
        "donut-radius",
        "donut-rings",
        "gauss-center",
        "rect-size",
        "sphere-center",
        "grid-pattern",
        "lung-tumor-pos"
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

    do_command_synth (argc, argv);

    lua_cli_glue_solvent (L, argv, argn);

    lua_pushnumber (L, 0);
    return 1; // # of return values
}
