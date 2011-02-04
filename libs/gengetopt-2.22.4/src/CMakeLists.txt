PROJECT (gengetopt_src)

SET (GENGETOPT_SRC
  argsdef.c  gengetopt.cc  getopt.c  parser.cc   yyerror.cc
  cmdline.c  getopt1.c     gm.cc     scanner.cc
  acceptedvalues.cpp
  fileutils.cpp
  ggos.cpp
  gm_utils.cpp
skels/check_modes.cc         skels/given_field.cc
skels/clear_arg.cc           skels/group_counter.cc
skels/clear_given.cc         skels/group_option.cc
skels/copyright.cc           skels/handle_help.cc
skels/c_source.cc            skels/handle_version.cc
skels/custom_getopt_gen.cc   skels/header.cc
skels/dependant_option.cc    skels/init_args_info.cc
skels/exit_failure.cc        skels/multiple_fill_array.cc
skels/file_save.cc           skels/multiple_opt_list.cc
skels/file_save_multiple.cc  skels/option_arg.cc
skels/free_list.cc           skels/print_help_string.cc
skels/free_multiple.cc       skels/required_option.cc
skels/free_string.cc         skels/reset_group.cc
skels/generic_option.cc      skels/update_given.cc
  )

ADD_EXECUTABLE (gengetopt ${GENGETOPT_SRC})
