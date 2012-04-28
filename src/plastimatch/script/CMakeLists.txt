##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_script)

configure_file (
    ${CMAKE_CURRENT_SOURCE_DIR}/plmscript_config.h.in
    ${PLM_BUILD_ROOT}/plmscript_config.h
)

include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

set (PLMSCRIPT_LIBRARY_SRC
  lua_class_image.cxx lua_class_image.h
  lua_class_register.cxx lua_class_register.h
  lua_class_stage.cxx lua_class_stage.h
  lua_class_structs.cxx lua_class_structs.h
  lua_class_xform.cxx lua_class_xform.h
#  lua_cli_glue.cxx lua_cli_glue.h
#  lua_iface_add.cxx lua_iface_add.h
#  lua_iface_crop.cxx lua_iface_crop.h
#  lua_iface_mask.cxx lua_iface_mask.h
#  lua_iface_register.cxx lua_iface_register.h
#  lua_iface_resample.cxx lua_iface_resample.h
#  lua_iface_synth.cxx lua_iface_synth.h
  lua_tty.cxx lua_tty.h
  lua_tty_commands.cxx lua_tty_commands.h
#  lua_tty_commands_pcmd.cxx lua_tty_commands_pcmd.h
  lua_tty_commands_util.cxx lua_tty_commands_util.h
  lua_util.cxx lua_util.h
  )

# JAS 2012.04.27
#   This Qt4 mess here is simply horrible
#   ...note to self, remember to fix this
#   once we have all the qt code building
#   as plmqt
if (QT4_FOUND)
    qt4_wrap_cpp (PORTAL_MOC_SRC_HACK
      ../cview_portal.h
    )
    set (PLMSCRIPT_LIBRARY_SRC
      ${PLMSCRIPT_LIBRARY_SRC}
      ${PORTAL_MOC_SRC_HACK}
      ../cview_portal.cxx
      lua_tty_preview.cxx
    )
endif ()

plm_add_library (
  plmscript
  "${PLMSCRIPT_LIBRARY_SRC}" 
  "${PLMSCRIPT_LIBRARY_DEPENDENCIES}"
  "")

if (SLICER_FOUND)
  if (WIN32)
    plm_slicer_copy_dll (plastimatch_slicer_copy_plmscript
      "${INDIR}/plmscript.dll"
      "${OUTDIR}/plmscript.dll"
      "${QTOUTDIR}/plmscript.dll"
      plmscript
    )
  endif ()
endif ()



set (LUA_INTERFACE_SRC

)
