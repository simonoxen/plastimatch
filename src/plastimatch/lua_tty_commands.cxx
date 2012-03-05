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
#include "lua_tty.h"
#include "lua_tty_commands.h"
#include "lua_tty_commands_pcmd.h"


static void
build_args (int* argc, char*** argv, char* cmd)
{
    int i;
    char tmp[TTY_MAXINPUT];
    char* token;

    /* get argc */
    strcpy (tmp, cmd);
    token = strtok (tmp, " ");
    *argc=0;
    while (token) {
        token = strtok (NULL, " ");
        (*argc)++;
    }
    *argv = (char**)malloc (*argc * sizeof(char*));

    /* populate argv */
    token = strtok (cmd, " ");
    (*argv)[0] = (char*)malloc (strlen(token)*sizeof(char));
    strcpy ((*argv)[0], token);
    for (i=1; i<*argc; i++) {
        token = strtok (NULL, " ");
        (*argv)[i] = (char*)malloc (strlen(token)*sizeof(char));
        strcpy ((*argv)[i], token);
    }
}

static void
do_tty_command_help (lua_State* L, int argc, char** argv)
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

    if (!strcmp (argv[0], TTY_CMD_RUN)) {
        do_tty_command_help (L, argc, argv);
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
