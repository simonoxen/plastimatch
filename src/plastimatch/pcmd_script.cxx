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
#include "pcmd_script.h"
#include "lua_class_image.h"
#include "lua_class_xform.h"
#include "lua_cli_glue.h"
#include "lua_iface_add.h"
#include "lua_iface_crop.h"
#include "lua_iface_mask.h"     /* also contains fill() */
#include "lua_iface_register.h"
#include "lua_iface_resample.h"
#include "lua_iface_synth.h"
#include "lua_util.h"


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
    register_lua_class_xform (L);
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
