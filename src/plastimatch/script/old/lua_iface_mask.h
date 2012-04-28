/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _lua_iface_mask_h_
#define _lua_iface_mask_h_

#include "plm_config.h"

#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"

int LUAIFACE_mask (lua_State* L);
int LUAIFACE_fill (lua_State* L);

#endif
