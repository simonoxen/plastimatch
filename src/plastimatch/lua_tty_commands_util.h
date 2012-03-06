/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _lua_tty_commands_util_h_
#define _lua_tty_commands_util_h_

#include "plm_config.h"

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
void list_vars_of_class (lua_State* L, char* class_name);

#if defined __cplusplus
}
#endif

#endif
