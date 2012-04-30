##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_qt)

configure_file (
    ${CMAKE_CURRENT_SOURCE_DIR}/plmqt_config.h.in
    ${PLM_BUILD_ROOT}/plmqt_config.h
)

include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
include_directories (BEFORE ${CMAKE_CURRENT_BINARY_DIR})

set (PLASTIMATCH_QT_SRC
  # pqt_application.cxx pqt_application.h
  pqt_data_source_dialog.cxx pqt_data_source_dialog.h
  pqt_data_source_list_model.cxx pqt_data_source_list_model.h
  pqt_database.cxx pqt_database.h
  pqt_findscu.cxx pqt_findscu.h
  pqt_main.cxx
  pqt_main_window.cxx pqt_main_window.h
  pqt_patient_list_model.cxx pqt_patient_list_model.h
  )

set (CRYSTALVIEW_SRC
  cview_portal.cxx cview_portal.h
  cview_main.cxx cview_main.h
  )


if (QT4_FOUND)
  qt4_wrap_cpp (PORTAL_MOC_SRC
    cview_portal.h
    )
  qt4_wrap_cpp (CRYSTALVIEW_MOC_SRC
    cview_main.h
    )
  set (CRYSTALVIEW_SRC
    ${CRYSTALVIEW_SRC}
    ${CRYSTALVIEW_MOC_SRC}
    ${PORTAL_MOC_SRC}
    )

  if (QT_QTSQL_FOUND)
    qt4_wrap_cpp (PLASTIMATCH_QT_SRC 
      #    pqt_application.h
      pqt_data_source_dialog.h
      pqt_data_source_list_model.h
      pqt_main_window.h 
      pqt_patient_list_model.h
      )
    qt4_wrap_ui (PLASTIMATCH_QT_SRC 
      pqt_data_source_dialog.ui
      pqt_main_window.ui
      )
  endif ()
endif ()

if (FULL_PLASTIMATCH_BUILD)
  if (QT4_FOUND)
    set (CRYSTALVIEW_LIBRARIES ${PLASTIMATCH_LIBS} 
      ${QT_LIBRARIES} ${QT_QTSQL_LIBRARIES})
    plm_add_executable (cview "${CRYSTALVIEW_SRC}"
      "${CRYSTALVIEW_LIBRARIES}" "${GPUIT_EXE_LDFLAGS}"
      ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
  endif ()
  if (QT4_FOUND AND QT_QTSQL_FOUND)
    set (PLASTIMATCH_QT_LIBRARIES ${PLASTIMATCH_LIBS} 
      ${QT_LIBRARIES} ${QT_QTSQL_LIBRARIES})
    plm_add_executable (plastimatch_qt "${PLASTIMATCH_QT_SRC}"
      "${PLASTIMATCH_QT_LIBRARIES}" "${GPUIT_EXE_LDFLAGS}" 
      ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
  endif ()
endif ()
