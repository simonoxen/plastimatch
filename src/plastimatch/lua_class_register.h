/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _lua_class_register_h_
#define _lua_class_register_h_

#include "plm_config.h"
#include "plm_image.h"
#include "plm_parms.h"
#include "plm_path.h"

#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"

#include "lua_class_stage.h"

#define LUA_CLASS_REGISTER "register"
#define _MAX_STAGES 20

typedef struct lua_register_struct lua_register;
struct lua_register_struct {
    char fn[_MAX_PATH];
    Registration_parms *regp;
    Plm_image *moving;    
    Plm_image *fixed;    
};


int register_lua_class_register (lua_State *L);
void init_register_instance (lua_register* lregister);

#endif
