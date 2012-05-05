/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _lua_class_register_h_
#define _lua_class_register_h_

#include "plmscript_config.h"
#include "plm_image.h"
#include "plm_parms.h"
#include "plm_path.h"

#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"

#include "lua_class_image.h"
#include "lua_class_structs.h"

#define _MAX_STAGES 20

typedef struct lua_register_struct lua_register;
struct lua_register_struct {
    char fn[_MAX_PATH];
    Registration_parms *regp;
    lua_image *moving;    
    lua_image *fixed;    
    lua_ss *moving_mask;
    lua_ss *fixed_mask;
};


int register_lua_class_register (lua_State *L);
void init_register_instance (lua_register* lregister);

#endif
