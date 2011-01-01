/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */

/* Shamelessly stolen and modified from RTK */
#define GGO(ggo_filename, args_info, skip_args)				\
    args_info_##ggo_filename args_info;					\
    {									\
	int ggo_rc = 0;							\
	char *tmp;							\
									\
	if (0) {\
	/* Skip arguments.  For example in the following command: */	\
	/*   "plastimatch segment body [options]                  */	\
	/* we would set skip_args to 2 to skip "segment body"     */	\
	tmp = argv[0];							\
	argv[0] = argv[skip_args];					\
	argv[skip_args] = tmp;						\
	argv += skip_args;						\
	argc -= skip_args;						\
	}\
									\
	ggo_rc = cmdline_parser_##ggo_filename##2 (			\
	    argc, argv, &args_info, 1, 1, 0);				\
	if (ggo_rc) {							\
	    fprintf (stderr, "Run \"" #ggo_filename " --help\" to see the list of options\n"); \
	    exit (-1);							\
	}								\
	if (args_info.config_given) {					\
	    ggo_rc = cmdline_parser_##ggo_filename##_configfile (args_info.config_arg, &args_info, 0, 0, 1); \
	} else {							\
	    ggo_rc = cmdline_parser_##ggo_filename(argc, argv, &args_info); \
	}								\
	if (ggo_rc) {							\
	    fprintf (stderr, "Run \"" #ggo_filename " --help\" to see the list of options\n"); \
	    exit (-1);							\
	}								\
    }

#define GGO_FREE(ggo_filename, args_info, skip_args)			\
    {									\
	char *tmp;							\
	if (0) {\
	/* Restore argc, argv */					\
	argv -= skip_args;						\
	tmp = argv[0];							\
	argv[0] = argv[skip_args];					\
	argv[skip_args] = tmp;						\
	argc += skip_args;						\
	}\
									\
	cmdline_parser_##ggo_filename##_free (&args_info);		\
    }
    

