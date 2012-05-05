/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _lua_tty_commands_util_h_
#define _lua_tty_commands_util_h_

#include "plmscript_config.h"

extern "C" {
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
}

#if defined __cplusplus
extern "C" {
#endif

void
print_command_table (
    const char* cmds[], /* command array   */
    unsigned int nc,    /* # of commands   */
    unsigned int tw,    /* table width     */
    unsigned int sp     /* minimum spacing */
);

void build_args (int* argc, char*** argv, char* cmd);
void list_vars_of_class (lua_State* L, const char* class_name);
void sort_list (char** c, int n);
void* get_obj_ptr_from_name (lua_State* L, const char* name);

#if defined __cplusplus
}
#endif

#endif
