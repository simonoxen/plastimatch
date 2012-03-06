/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

extern "C" {
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
}

#include "file_util.h"
#include "lua_class_image.h"
#include "lua_class_xform.h"
#include "lua_tty.h"
#include "lua_tty_commands.h"
#include "lua_tty_commands_pcmd.h"
#include "lua_tty_commands_util.h"
#include "lua_util.h"


static void
do_tty_command_help (lua_State* L, int argc, char** argv)
{
    if (argc == 1) {
        print_command_table (tty_cmds, num_tty_cmds, 60, 3);
    } else {
        if (!strcmp (argv[1], TTY_CMD_PCMD)) {
            print_command_table (pcmds, num_pcmds, 60, 3);
        }
        else if (!strcmp (argv[1], TTY_CMD_RUN)) {
            fprintf (stdout, "execute a script from disk\n");
            fprintf (stdout, "  Usage: " TTY_CMD_RUN " script_name\n");
        }
        else if (!strcmp (argv[1], TTY_CMD_DIR)) {
            fprintf (stdout, "get listing for current directory\n");
        }
        else if (!strcmp (argv[1], TTY_CMD_LS)) {
            fprintf (stdout, "get listing for current directory\n");
        }
        else if (!strcmp (argv[1], TTY_CMD_LIST)) {
            fprintf (stdout, "display allocated Plastimatch objects\n");
            fprintf (stdout, "  (images, xforms, structures, etc)\n");
        }
    }

    fprintf (stdout, "\n");
}

static void
do_tty_command_run (lua_State* L, int argc, char** argv)
{
    /* run lua script */
    if (argc < 2) return;
    if (file_exists (argv[1])) {
        printf ("-- running script : %s\n\n", argv[1]);
        luaL_dofile (L, argv[1]);
    } else {
        printf ("unable to load script: %s\n", argv[1]);
    }
}

static void
do_tty_command_list (lua_State* L, int argc, char** argv)
{
    fprintf (stdout, "[IMAGES] - ");
    list_vars_of_class (L, LUA_CLASS_IMAGE);
    printf ("\n");
    fprintf (stdout, "[TRANSFORMS] - ");
    list_vars_of_class (L, LUA_CLASS_XFORM);
    printf ("\n");
}

void
do_tty_command (lua_State *L)
{
    char cmd[TTY_MAXINPUT];
    char** argv;
    int argc;

    memset (cmd, '\0', TTY_MAXINPUT * sizeof(char));
    strcpy (cmd, lua_tolstring (L, 1, NULL));
    lua_pop (L, 1);

    build_args (&argc, &argv, cmd);

    if (!strcmp (argv[0], TTY_CMD_HELP)) {
        do_tty_command_help (L, argc, argv);
    }
    else if (!strcmp (argv[0], TTY_CMD_RUN)) {
        do_tty_command_run (L, argc, argv);
    }
    else if (!strcmp (argv[0], TTY_CMD_LIST)) {
        do_tty_command_list (L, argc, argv);
    }
    else if (!strcmp (argv[0], TTY_CMD_DIR) ||
             !strcmp (argv[0], TTY_CMD_LS)) {
        /* run some OS dependent code */
    }
    else if (!strcmp (argv[0], TTY_CMD_CD)) {
        /* run some OS dependent code */
    }
    else if (!strcmp (argv[0], TTY_CMD_PCMD)) {
        do_tty_command_pcmd (argc, argv);
    }
}
