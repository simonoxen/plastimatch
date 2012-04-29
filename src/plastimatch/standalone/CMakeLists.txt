##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_standalone)

include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

set (BRAGG_CURVE_SRC
  bragg_curve_main.c 
  bragg_curve_opts.c bragg_curve_opts.h
  )
set (BSPLINE_SRC
  bspline_main.cxx 
  bspline_opts.cxx bspline_opts.h
  )
set (CHECK_GRAD_SRC
  check_grad.cxx
  check_grad_opts.cxx check_grad_opts.h 
  )
set (COMPUTE_DISTANCE_SRC
  compute_distance.cxx compute_distance.h
  closest_point.cxx 
  read_obj.cxx
  )
set (DEMONS_SRC
  demons_main.cxx 
  demons_opts.cxx demons_opts.h
  )
set (DICE_STATS_SRC
  dice_stats_main.cxx
  dice_statistics.cxx dice_statistics.h
  hausdorff_statistics.cxx hausdorff_statistics.h
  )
set (DISTANCE_MAP_SRC
  distance_map.cxx
  )


if (FULL_PLASTIMATCH_BUILD)
  plm_add_executable (compute_distance "${COMPUTE_DISTANCE_SRC}" 
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
  plm_add_executable (dice_stats "${DICE_STATS_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_IF_NOT_DEBIAN})
  plm_add_executable (distance_map "${DISTANCE_MAP_SRC}"
    "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
endif ()


plm_add_executable (bspline "${BSPLINE_SRC}" 
  "${GPUIT_LIBRARIES}" "${PLMCUDA_EXE_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
plm_add_executable (check_grad "${CHECK_GRAD_SRC}" 
  "${GPUIT_LIBRARIES}" "${GPUIT_EXE_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
plm_add_executable (demons "${DEMONS_SRC}" 
  "${GPUIT_LIBRARIES}" "${PLMCUDA_EXE_LDFLAGS}" 
  ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})


if (NOT BUILD_DEBIAN)
  plm_add_executable (bragg_curve "${BRAGG_CURVE_SRC}" 
    "${GPUIT_LIBRARIES}" "${GPUIT_EXE_LDFLAGS}" 
    ${BUILD_IF_NOT_SLICER_EXT} ${INSTALL_NEVER})
endif ()
