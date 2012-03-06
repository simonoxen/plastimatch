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
print_command_table (
    const char* cmds[], /* command array   */
    unsigned int nc,    /* # of commands   */
    unsigned int tw,    /* table width     */
    unsigned int sp     /* minimum spacing */
)
{
    unsigned int i;
    unsigned int c,w,n;     /* col, word, command # */
    unsigned int lc;        /* longest command      */

    lc = 0;
    for (i=0; i<nc; i++) {
        if (strlen (cmds[i]) > lc) {
            lc = strlen (cmds[i]);
        }
    }

    c=0;w=0;n=0;
    while (n<nc) {
        while (c<tw) {
            while (w<strlen(cmds[n])) {
                fprintf (stdout, "%c", cmds[n][w++]);
                c++;
            }
            w=0;
            while (w<(lc-strlen(cmds[n])+sp)) {
                fprintf (stdout, " ");
                w++;c++;
            }
            n++; w=0;
            if (n>=nc) break;
        }
        printf ("\n"); c=0;
    }

}

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
do_tty_command_help (lua_State* L, int argc, char** argv)
{
    if (argc == 1) {
        print_command_table (tty_cmds, num_tty_cmds, 75, 3);
    } else {
        if (!strcmp (argv[1], TTY_CMD_PCMD)) {
            print_command_table (pcmds, num_pcmds, 75, 3);
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
    }

    fprintf (stdout, "\n");
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
