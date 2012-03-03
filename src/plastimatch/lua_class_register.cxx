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

/* Name of class as exposed to Lua */
#define THIS_CLASS LUA_CLASS_REGISTER


/* Helpers */
void
init_register_instance (lua_register* lregister)
{

    lregister->stages = (lua_stage**)malloc (_MAX_STAGES*sizeof(lua_stage*));

}

/*******************************************************/
/* Object Methods                                      */
/*******************************************************/
static int
register_new (lua_State *L)
{
    lua_register *lregister =
        (lua_register*)lua_new_instance (L, THIS_CLASS, sizeof(lua_register));

    init_register_instance (lregister);

    /* build & initialize the stages array */
    lua_stage** lstages = lregister->stages;
    for (int i=0; i<_MAX_STAGES; i++) {
        lstages[i] = NULL;
    }

#if 0
    /* Need to build a special pointer handler for something this cool... */
    lua_stage** lstages = lregister->stages;
    for (int i=0; i<_MAX_STAGES; i++) {
        lstages[i] =
            (lua_stage*)lua_new_instance (L, LUA_CLASS_STAGE, sizeof(lua_stage));
        lua_pop (L, 1);  /* don't return lstage userdata to Lua */
        fprintf (stderr, "Allocating [%i] %p\n", i, lstages[i]);
        init_stage_instance (lstages[i]);
    }
#endif

    return 1;
}

static int
register_select_stage (lua_State *L)
{
    lua_register *lregister = (lua_register*)get_obj_ptr (L, THIS_CLASS, 1);
    int idx = luaL_checkint (L, 2);

    lregister->stage_idx = idx;

    return 1;
}

static int
register_set_foo (lua_State *L)
{
    lua_register *lregister = (lua_register*)get_obj_ptr (L, THIS_CLASS, 1);
    int foo = luaL_checkint (L, 2);

    lregister->stages[lregister->stage_idx]->foo = foo;

    return 1;
}

static int
register_get_foo (lua_State *L)
{
    lua_register *lregister = (lua_register*)get_obj_ptr (L, THIS_CLASS, 1);

    int foo = lregister->stages[lregister->stage_idx]->foo;

    lua_pushnumber (L, foo);

    return 1;
}
/*******************************************************/


/*+++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/* Object Actions                                      */
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++*/

/* Action: garbage collection */
static int
register_action_gc (lua_State *L)
{
    // TODO
    return 0;
}
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++*/






/* Class Creation */

/* methods table for object */
static const luaL_reg
register_methods[] = {
    {"new",   register_new},
    {"stage", register_select_stage},
    {"set",   register_set_foo},
    {"get",   register_get_foo},
    {0, 0}
};

/* metatable of actions */
static const luaL_reg
register_meta[] = {
    {"__gc",       register_action_gc},
    {0, 0}
};


int
register_lua_class_register (lua_State *L)
{
    return register_lua_class (L, THIS_CLASS, register_methods, register_meta);
}
