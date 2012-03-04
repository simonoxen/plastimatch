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

#include "lua_class_register.h"
#include "lua_class_stage.h"
#include "lua_util.h"
#include "plm_parms.h"
#include "plm_stages.h"

/* Name of class as exposed to Lua */
#define THIS_CLASS LUA_CLASS_REGISTER


/* Helpers */
void
init_register_instance (lua_register* lregister)
{
    memset (lregister->fn, '\0', sizeof (lregister->fn));
    lregister->regp = NULL;
    lregister->moving = NULL;
    lregister->fixed = NULL;
}

/*******************************************************/
/* Object Methods                                      */
/*******************************************************/
static int
register_load (lua_State *L)
{
    const char* fn = luaL_optlstring (L, 1, NULL, NULL);

    if (!fn) {
        fprintf (stderr, "error -- register.load() -- no file specified\n");
        return 0;
    }

    Registration_parms *regp = new Registration_parms; //(Registration_parms*)malloc (sizeof(Registration_parms));
    if (plm_parms_parse_command_file (regp, fn) < 0) {
        fprintf (stderr, "error -- register.load() -- unable to load %s\n", fn);
        return 0;
    }

    lua_register *lreg = (lua_register*)lua_new_instance (L,
                                    THIS_CLASS,
                                    sizeof(lua_register));

    init_register_instance (lreg);

    strcpy (lreg->fn, fn);
    lreg->regp = regp;

    return 1;
}

static int
register_update (lua_State *L)
{
    // We will have to pull file loads outside of the registration
    // algorithms before this can go any further.
    //    They will need to be altered to receive Plm_images
    //    instead of file names... or something similar.

//    lua_register *lreg = (lua_register*)get_obj_ptr (L, THIS_CLASS, 1);
//    lreg->regp->moving = lreg->moving;
//    lreg->regp->fixed  = lreg->fixed;   /* Can't do this ...yet */

    return 0;
}

/*******************************************************/


/*+++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/* Object Actions                                      */
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++*/

/* Action: garbage collection */
static int
register_action_gc (lua_State *L)
{
    return 0;
}
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++*/






/* Super Class Creation */

/* methods table for object */
static const luaL_reg
register_methods[] = {
    {"load",   register_load},
    {"update", register_update},
    {0, 0}
};

/* metatable of actions */
static const luaL_reg
register_meta[] = {
    {"__gc",       register_action_gc},
    {0, 0}
};

/* glue to C struct */
static const lua_sc_glue
getters[] = {
    {"moving",   sc_get_string,  offsetof (lua_register, moving)    },
    {"fixed",    sc_get_string,  offsetof (lua_register, fixed)    },
    {0, 0}
};

static const lua_sc_glue
setters[] = {
    {"moving",   sc_set_string,  offsetof (lua_register, moving)    },
    {"fixed",    sc_set_string,  offsetof (lua_register, fixed)    },
    {0, 0}
};


int
register_lua_class_register (lua_State *L)
{
    return register_lua_super_class (L, THIS_CLASS, register_methods, register_meta, getters, setters);
}
