project (PLM_Testing)

## -------------------------------------------------------------------------
## TODO: create targets like "make check plm-reg-bspline-a"
## http://stackoverflow.com/questions/733475/cmake-ctest-make-test-doesnt-build-tests
## -------------------------------------------------------------------------

## -------------------------------------------------------------------------
## Testing macros
## -------------------------------------------------------------------------
macro (plmtest_check_interval 
    test_name stdout_file regex_string lower_thresh upper_thresh)
  string (REPLACE "\\" "\\\\" regex_escaped "${regex_string}")
  add_test (${test_name}
    ${CMAKE_COMMAND} 
    "-DINFILE=${stdout_file}"
    "-DREGEX=${regex_escaped}"
    "-DLOWER_THRESH=${lower_thresh}"
    "-DUPPER_THRESH=${upper_thresh}"
    -P "${PLM_TESTING_SOURCE_DIR}/PlmCheckInterval.cmake"
    )
endmacro ()

macro (plmtest_check_string
    test_name stdout_file regex_string match_string)
  string (REPLACE "\\" "\\\\" regex_escaped "${regex_string}")
  add_test (${test_name}
    ${CMAKE_COMMAND} 
    "-DINFILE=${stdout_file}"
    "-DREGEX=${regex_escaped}"
    "-DMATCH_STRING=${match_string}"
    -P "${PLM_TESTING_SOURCE_DIR}/PlmCheckString.cmake"
    )
endmacro ()

## -------------------------------------------------------------------------
## Download data sets
## -------------------------------------------------------------------------
macro (plm_download_and_extract DOWNLOAD_URL DOWNLOAD_FILENAME DOWNLOAD_DIR)
  if (NOT IS_DIRECTORY "${PLM_TESTING_DOWNLOAD_DATA_DIR}")
    file (MAKE_DIRECTORY "${PLM_TESTING_DOWNLOAD_DATA_DIR}")
  endif ()
  if (NOT EXISTS ${DOWNLOAD_FILENAME})
    message (STATUS "Downloading test data")
    file (DOWNLOAD "${DOWNLOAD_URL}" "${DOWNLOAD_FILENAME}")
  endif ()
  if (NOT EXISTS ${DOWNLOAD_DIR})
    message (STATUS "Untarring test data")
    execute_process (
      COMMAND 
      ${CMAKE_COMMAND} "-E" "tar" "xvzf" "${DOWNLOAD_FILENAME}"
      WORKING_DIRECTORY ${PLM_TESTING_BUILD_DIR}
      RESULT_VARIABLE untar_result
      OUTPUT_VARIABLE untar_output
      ERROR_VARIABLE untar_error)
    message (STATUS "Untar command returned: ${untar_result}\n${untar_output}\n${untar_error}")
  endif ()
endmacro ()

if (PLM_DOWNLOAD_HEADPHANTOM)
  plm_download_and_extract (
    "http://forge.abcd.harvard.edu/gf/download/frsrelease/85/1929/headphantom.tar.gz"
    "${PLM_TESTING_DOWNLOAD_DATA_DIR}/headphantom.tar.gz"
    "${PLM_TESTING_BUILD_DIR}/headphantom")
endif ()

if (PLM_DOWNLOAD_DICOMRT_AW)
  plm_download_and_extract (
    "http://forge.abcd.harvard.edu/gf/download/frsrelease/85/1613/foot-dicomrt-aw-4.4.tar.gz"
    "${PLM_TESTING_DOWNLOAD_DATA_DIR}/foot-dicomrt-aw-4.4.tar.gz"
    "${PLM_TESTING_BUILD_DIR}/foot-dicomrt-aw-4.4")
endif ()

if (PLM_DOWNLOAD_DICOMRT_CERR)
  plm_download_and_extract (
    "http://forge.abcd.harvard.edu/gf/download/frsrelease/85/1632/chest-phantom-dicomrt-CERR4pt0beta2_25_Jan_2011.tar.gz"
    "${PLM_TESTING_DOWNLOAD_DATA_DIR}/chest-phantom-dicomrt-CERR4pt0beta2_25_Jan_2011.tar.gz"
    "${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-CERR4pt0beta2_25_Jan_2011")
endif ()

if (PLM_DOWNLOAD_DICOMRT_IRREGULAR)
  plm_download_and_extract (
    "http://forge.abcd.harvard.edu/gf/download/frsrelease/85/1827/xio-4.60-irregular-spacing.tar.gz"
    "${PLM_TESTING_DOWNLOAD_DATA_DIR}/xio-4.60-irregular-spacing.tar.gz"
    "${PLM_TESTING_BUILD_DIR}/xio-4.60-irregular-spacing")
endif ()

if (PLM_DOWNLOAD_DICOMRT_PINNACLE)
  plm_download_and_extract (
    "http://forge.abcd.harvard.edu/gf/download/frsrelease/85/1187/rando-dicomrt-pinnacle3-8.2g.tar.gz"
    "${PLM_TESTING_DOWNLOAD_DATA_DIR}/rando-dicomrt-pinnacle3-8.2g.tar.gz"
    "${PLM_TESTING_BUILD_DIR}/rando-dicomrt-pinnacle3-8.2g")
endif ()

if (PLM_DOWNLOAD_DICOMRT_XIO)
  plm_download_and_extract (
    "http://forge.abcd.harvard.edu/gf/download/frsrelease/85/934/chest-phantom-dicomrt-xio-4.33.02.tar.gz"
    "${PLM_TESTING_DOWNLOAD_DATA_DIR}/chest-phantom-dicomrt-xio-4.33.02.tar.gz"
    "${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02")
endif ()

if (PLM_DOWNLOAD_FDK_VARIAN)
  plm_download_and_extract (
    "http://forge.abcd.harvard.edu/gf/download/frsrelease/85/1127/varian-catphan.tar.gz"
    "${PLM_TESTING_DOWNLOAD_DATA_DIR}/varian-catphan.tar.gz"
    "${PLM_TESTING_BUILD_DIR}/varian-catphan")
endif ()

if (PLM_DOWNLOAD_XIO)
  plm_download_and_extract (
    "http://forge.abcd.harvard.edu/gf/download/frsrelease/85/939/chest-phantom-xio-4.33.02.tar.gz"
    "${PLM_TESTING_DOWNLOAD_DATA_DIR}/chest-phantom-xio-4.33.02.tar.gz"
    "${PLM_TESTING_BUILD_DIR}/chest-phantom-xio-4.33.02")
endif ()


## -------------------------------------------------------------------------
## Copy plugin library
##   GCS: N.b. I don't think this is necessary.  There is no need to 
##   copy gpuit or plastimatch1 libraries.  Instead we set the extra path 
##   as an argument to plm_add_test.
## -------------------------------------------------------------------------
# JAS 2010.11.23
# Tests will fail if the test binaries cannot find the cuda plugin.
# So, we copy it into the testing directory.  This seems to work well.
if (PLM_USE_GPU_PLUGINS)
  # Because MSVC places stuff in Release/ or Debug/, etc
  # CMAKE_CFG_INTDIR is "Release/" or "Debug/" for Windows or "/" for Linux
  # In the Linux case, the extra "/" doesn't hurt anything.
  set (GPU_PLUGIN_BIN_DIR
    "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/")

  if (CUDA_FOUND)
    # CMAKE_SHARED_LIBRARY_PREFIX is "" under Windows and "lib" under Linux
    # CMAKE_SHARE_LIBRARY_SUFFIX is ".dll" under Windows and ".so" under Linux
    set (CUDA_PLUGIN_FILE
      "${CMAKE_SHARED_LIBRARY_PREFIX}plmcuda${CMAKE_SHARED_LIBRARY_SUFFIX}")

    # This tells CMake how to handle a target depending on
    # "Testing/[lib]plmcuda[.dll/.so]"
    add_custom_command (
      OUTPUT "${PLM_TESTING_BUILD_DIR}/${CUDA_PLUGIN_FILE}"
      COMMAND
      ${CMAKE_COMMAND} "-E" "copy"
      "${GPU_PLUGIN_BIN_DIR}/${CUDA_PLUGIN_FILE}"
      "${PLM_TESTING_BUILD_DIR}/${CUDA_PLUGIN_FILE}"
      DEPENDS plmcuda
      )

    # The Phony target
    # This is the Makefile equivalent of
    # Testing/plmcuda.dll : plmcuda.dll      (Under Windows)  *OR*
    # Testing/libplmcuda.so : libplmcuda.so  (Under Linux)
    #
    # The "rule" for handling this is the above custom command 
    add_custom_target (plmcuda_tcpy
      DEPENDS "${PLM_TESTING_BUILD_DIR}/${CUDA_PLUGIN_FILE}"
      )

    # When using the plmcuda plugin, GPUIT depends on it for GPU acceleration.
    # So, we add it as a dependency here so that plmcuda plugin will be copied
    # to Testing whenever the user builds GPUIT (or any target that depends on
    # GPUIT, which is most targets).
    add_dependencies (gpuit plmcuda_tcpy)
  endif ()

  if (OPENCL_FOUND)
    set (OPENCL_PLUGIN_FILE
      "${CMAKE_SHARED_LIBRARY_PREFIX}plmopencl${CMAKE_SHARED_LIBRARY_SUFFIX}")

    add_custom_command (
      OUTPUT "${PLM_TESTING_BUILD_DIR}/${OPENCL_PLUGIN_FILE}"
      COMMAND
      ${CMAKE_COMMAND} "-E" "copy"
      "${GPU_PLUGIN_BIN_DIR}/${OPENCL_PLUGIN_FILE}"
      "${PLM_TESTING_BUILD_DIR}/${OPENCL_PLUGIN_FILE}"
      DEPENDS plmopencl
      )

    add_custom_target (plmopencl_tcpy
      DEPENDS "${PLM_TESTING_BUILD_DIR}/${OPENCL_PLUGIN_FILE}"
      )
    add_dependencies (gpuit plmopencl_tcpy)
  endif ()
endif ()

## -------------------------------------------------------------------------
## synth-test - create synthetic mha files
##   black-1.mha   All black, slightly different geometry
##   donut-1.mha   Centered donut
##   gauss-1.mha   Centered gauss
##   gauss-2.mha   Off-center gauss
##   gauss-3.mha   Inverted gauss
##   gauss-4.mha   Gauss with inverted pixel spacing
##   gauss-5.mha   Even more off-center gauss
##   gauss-6.mha   Off-center gauss, in off-center world coordinates
##   gauss-ushort-1.mha   Centered gauss, ushort
##   gauss-ushort-2.mha   Off-center gauss, ushort
##   rect-1.mha    Inverted rect, centered
##   rect-2.mha    Standard rect, off-center
##   rect-3.mha    Standard rect, centered
##   rect-4.mha    Uchar rect, centered
##   rect-5.mha    Standard rect, centered, direction cosines rotated
##   sphere-1.mha  Uchar sphere, bg=0, fg=255
##   sphere-2.mha  Standard sphere, centered
##   rtds-ct-1.mha    Synthetic ct
##   rtds-dose-1.mha  Synthetic dose
##   rtds-s-1.mha     Synthetic ss
## -------------------------------------------------------------------------
set (SYNTH_MHA_SIZE 38)

plm_add_test (
  "black-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "synth;--output;${PLM_TESTING_BUILD_DIR}/black-1.mha;--output-type;float;--pattern;rect;--origin;-25 -25 -25;--dim;40 40 40;--volume-size;50 50 50;--background;-1000;--foreground;-1000"
  )
plm_add_test (
  "donut-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "synth;--output;${PLM_TESTING_BUILD_DIR}/donut-1.mha;--output-ss-img;${PLM_TESTING_BUILD_DIR}/donut-ss-1.mha;--output-type;float;--pattern;donut;--origin;-25 -25 -25;--dim;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--donut-radius;10 10 10;--background;-1000;--foreground;0;--output-dicom;${PLM_TESTING_BUILD_DIR}/donut-1-dicom"
  )
plm_add_test (
  "gauss-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "synth;--output;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;--output-type;float;--pattern;gauss;--origin;-25 -25 -25;--dim;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;-1000;--foreground;0;--gauss-center;0 0 0;--gauss-std;10 10 10"
  )
plm_add_test (
  "gauss-2"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "synth;--output;${PLM_TESTING_BUILD_DIR}/gauss-2.mha;--output-type;float;--pattern;gauss;--origin;-25 -25 -25;--dim;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;-1000;--foreground;0;--gauss-center;10 0 0;--gauss-std;10 10 10"
  )
plm_add_test (
  "gauss-3"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "synth;--output;${PLM_TESTING_BUILD_DIR}/gauss-3.mha;--output-type;float;--pattern;gauss;--origin;-25 -25 -25;--dim;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;0;--foreground;-1000;--gauss-center;10 0 0;--gauss-std;10 10 10"
  )
plm_add_test (
  "gauss-4"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "synth;--output;${PLM_TESTING_BUILD_DIR}/gauss-4.mha;--output-type;float;--pattern;gauss;--origin;+25 +25 -25;--dim;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--spacing;-1.31579 -1.31579 2;--background;-1000;--foreground;0;--gauss-center;0 0 0;--gauss-std;10 10 10"
  )
plm_add_test (
  "gauss-5"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "synth;--output;${PLM_TESTING_BUILD_DIR}/gauss-5.mha;--output-type;float;--pattern;gauss;--origin;-25 -25 -25;--dim;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;-1000;--foreground;0;--gauss-center;20 0 0;--gauss-std;10 10 10"
  )
plm_add_test (
  "gauss-6"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "synth;--output;${PLM_TESTING_BUILD_DIR}/gauss-6.mha;--output-type;float;--pattern;gauss;--origin;-15 -25 -25;--dim;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;-1000;--foreground;0;--gauss-center;10 0 0;--gauss-std;10 10 10"
  )
plm_add_test (
  "gauss-ushort-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "synth;--output;${PLM_TESTING_BUILD_DIR}/gauss-ushort-1.mha;--output-type;ushort;--pattern;gauss;--origin;-25 -25 -25;--dim;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;0;--foreground;1000;--gauss-center;0 0 0;--gauss-std;10 10 10"
  )
plm_add_test (
  "gauss-ushort-2"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "synth;--output;${PLM_TESTING_BUILD_DIR}/gauss-ushort-2.mha;--output-type;ushort;--pattern;gauss;--origin;-25 -25 -25;--dim;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;0;--foreground;1000;--gauss-center;10 0 0;--gauss-std;10 10 10"
  )
plm_add_test (
  "gauss-double-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "synth;--output;${PLM_TESTING_BUILD_DIR}/gauss-double-1.mha;--output-type;double;--pattern;gauss;--origin;-25 -25 -25;--dim;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;0;--foreground;1000;--gauss-center;0 0 0;--gauss-std;10 10 10"
  )
plm_add_test (
  "gauss-double-2"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "synth;--output;${PLM_TESTING_BUILD_DIR}/gauss-double-2.mha;--output-type;double;--pattern;gauss;--origin;-25 -25 -25;--dim;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;0;--foreground;1000;--gauss-center;10 0 0;--gauss-std;10 10 10"
  )
plm_add_test (
  "rect-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "synth;--output;${PLM_TESTING_BUILD_DIR}/rect-1.mha;--output-type;float;--output-dose-img;${PLM_TESTING_BUILD_DIR}/rect-1-dose-img.mha;--output-ss-img;${PLM_TESTING_BUILD_DIR}/rect-1-ss-img.mha;--output-dicom;${PLM_TESTING_BUILD_DIR}/rect-1-dicom;--pattern;rect;--dim;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--background;0;--foreground;-1000"
  )
plm_add_test (
  "rect-2"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "synth;--output;${PLM_TESTING_BUILD_DIR}/rect-2.mha;--output-type;float;--output-dicom;${PLM_TESTING_BUILD_DIR}/rect-2-dicom;--pattern;rect;--dim;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--background;-1000;--foreground;0;--rect-size;-30 70 -50 50 -50 50"
  )
plm_add_test (
  "rect-3"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "synth;--output;${PLM_TESTING_BUILD_DIR}/rect-3.mha;--output-type;float;--output-dicom;${PLM_TESTING_BUILD_DIR}/rect-3-dicom;--pattern;rect;--dim;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--background;-1000;--foreground;0;--rect-size;-50 50 -50 50 -50 50"
  )
plm_add_test (
  "rect-4"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "synth;--output;${PLM_TESTING_BUILD_DIR}/rect-4.mha;--output-type;uchar;--pattern;rect;--dim;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--background;0;--foreground;1;--rect-size;-50 50 -50 50 -50 50"
  )
plm_add_test (
  "rect-5"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "synth;--output;${PLM_TESTING_BUILD_DIR}/rect-5.mha;--output-type;float;--pattern;rect;--dim;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--background;-1000;--foreground;0;--rect-size;-50 50 -50 50 -50 50;--origin;0 -353.553 -250;--direction-cosines;rotated"
  )
plm_add_test (
  "sphere-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "synth;--output;${PLM_TESTING_BUILD_DIR}/sphere-1.mha;--output-type;uchar;--pattern;sphere;--origin;-25 -25 -25;--dim;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--volume-size;50 50 50;--background;0;--foreground;255;--sphere-center;0 0 0;--sphere-radius;10 10 10"
  )
plm_add_test (
  "sphere-2"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "synth;--output;${PLM_TESTING_BUILD_DIR}/sphere-2.mha;--output-type;float;--pattern;sphere;--dim;${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE} ${SYNTH_MHA_SIZE};--background;-1000;--foreground;0;--sphere-center;0 0 0;--sphere-radius;50 50 50"
  )

## -------------------------------------------------------------------------
## bspline - bspline algorithm flavors
## -------------------------------------------------------------------------
plm_add_test (
  "bspline-c"
  ""
  ${PLM_PLASTIMATCH_PATH}/bspline
  "-a;steepest;-m;10;-f;c;-x;${PLM_TESTING_BUILD_DIR}/bspline_c_xf.txt;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;${PLM_TESTING_BUILD_DIR}/gauss-2.mha"
  )
plmtest_check_interval (bspline-c-check
  "${PLM_TESTING_BUILD_DIR}/bspline-c.stdout.txt"
  "^\\\\[.*, 10] MSE *([0-9.]*)"
  "1168.25"
  "1168.35"
  )
set_tests_properties (bspline-c PROPERTIES DEPENDS "gauss-1;gauss-2")
set_tests_properties (bspline-c-check PROPERTIES DEPENDS bspline-c)

plm_add_test (
  "bspline-g"
  ""
  ${PLM_PLASTIMATCH_PATH}/bspline
  "-a;steepest;-m;10;-f;g;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;${PLM_TESTING_BUILD_DIR}/gauss-2.mha"
  )
plmtest_check_interval (bspline-g-check
  "${PLM_TESTING_BUILD_DIR}/bspline-g.stdout.txt"
  "^\\\\[.*, 10] MSE *([0-9.]*)"
  "1168.25"
  "1168.35"
  )
set_tests_properties (bspline-g PROPERTIES DEPENDS "gauss-1;gauss-2")
set_tests_properties (bspline-g-check PROPERTIES DEPENDS bspline-g)

plm_add_test (
  "bspline-h"
  ""
  ${PLM_PLASTIMATCH_PATH}/bspline
  "-a;steepest;-m;10;-f;h;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;${PLM_TESTING_BUILD_DIR}/gauss-2.mha"
  )
plmtest_check_interval (bspline-h-check
  "${PLM_TESTING_BUILD_DIR}/bspline-h.stdout.txt"
  "^\\\\[.*, 10] MSE *([0-9.]*)"
  "1168.25"
  "1168.35"
  )
set_tests_properties (bspline-h PROPERTIES DEPENDS "gauss-1;gauss-2")
set_tests_properties (bspline-h-check PROPERTIES DEPENDS bspline-g)

plm_add_test (
  "bspline-mi-c-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/bspline
  "-a;steepest;-m;10;-f;c;-M;mi;-B;20 20;-O;${PLM_TESTING_BUILD_DIR}/bspline-mi-c-1.mha;-x;${PLM_TESTING_BUILD_DIR}/bspline-mi-c-1.txt;-V;${PLM_TESTING_BUILD_DIR}/bspline-mi-c-1-vf.mha;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;${PLM_TESTING_BUILD_DIR}/gauss-3.mha"
  )
plmtest_check_interval (bspline-mi-c-1-check
  "${PLM_TESTING_BUILD_DIR}/bspline-mi-c-1.stdout.txt"
  "^\\\\[.*, 10] *MI *([-0-9.]*)"
  "-0.800"
  "-0.796"
  )
set_tests_properties (bspline-mi-c-1 PROPERTIES DEPENDS "gauss-1;gauss-3")
set_tests_properties (bspline-mi-c-1-check PROPERTIES DEPENDS bspline-mi-c-1)

plm_add_test (
  "bspline-mi-c-2"
  ""
  ${PLM_PLASTIMATCH_PATH}/bspline
  "-a;steepest;-m;10;-f;c;-M;mi;-O;${PLM_TESTING_BUILD_DIR}/bspline-mi-c-2.mha;${PLM_TESTING_BUILD_DIR}/rect-1.mha;${PLM_TESTING_BUILD_DIR}/rect-2.mha"
  )
## Without DOUBLE_HIST defined, output is 0.031
## With DOUBLE_HIST defined, output is 0.032
plmtest_check_interval (bspline-mi-c-2-check
  "${PLM_TESTING_BUILD_DIR}/bspline-mi-c-2.stdout.txt"
  "^\\\\[.*, 10] *MI *([-0-9.]*)"
  "-0.035"
  "-0.031"
  )
set_tests_properties (bspline-mi-c-2 PROPERTIES DEPENDS "rect-1;rect-2")
set_tests_properties (bspline-mi-c-2-check PROPERTIES DEPENDS bspline-mi-c-2)

## -------------------------------------------------------------------------
## demons
##   demons-a          gauss-1.mha, gauss-2.mha
##   demons-cuda-a     gauss-1.mha, gauss-2.mha
## -------------------------------------------------------------------------
plm_add_test (
  "demons-a"
  ""
  ${PLM_PLASTIMATCH_PATH}/demons
  "-m;50;-s;3;-f;5 5 5;-O;${PLM_TESTING_BUILD_DIR}/demons-a.mha;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;${PLM_TESTING_BUILD_DIR}/gauss-2.mha"
  )
plmtest_check_interval (demons-a-check
  "${PLM_TESTING_BUILD_DIR}/demons-a.stdout.txt"
  "^Mean: *([0-9.]*)"
  "2.1"
  "2.2"
  )
set_tests_properties (demons-a PROPERTIES DEPENDS "gauss-1;gauss-2")
set_tests_properties (demons-a-check PROPERTIES DEPENDS demons-a)

plm_add_test (
  "demons-cuda-a"
  ""
  ${PLM_PLASTIMATCH_PATH}/demons
  "-A;cuda;-m;50;-s;3;-f;5 5 5;-O;${PLM_TESTING_BUILD_DIR}/demons-cuda-a.mha;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;${PLM_TESTING_BUILD_DIR}/gauss-2.mha"
  )
plmtest_check_interval (demons-cuda-a-check
  "${PLM_TESTING_BUILD_DIR}/demons-cuda-a.stdout.txt"
  "^Mean: *([0-9.]*)"
  "2.1"
  "2.2"
  )
set_tests_properties (demons-cuda-a PROPERTIES DEPENDS "gauss-1;gauss-2")
set_tests_properties (demons-cuda-a-check PROPERTIES DEPENDS demons-cuda-a)

## -------------------------------------------------------------------------
## drr
##   drr-a       gauss-1.mha (normal image), exact
##   drr-b       gauss-1.mha (normal image), uniform
##   drr-c       gauss-4.mha (negative spacing), exact
##   drr-d       gauss-4.mha (negative spacing), uniform
##   drr-cuda    gauss-1.mha, uniform
##   drr-opencl  gauss-1.mha, uniform
## -------------------------------------------------------------------------
plm_add_test (
  "drr-a"
  ""
  ${PLM_PLASTIMATCH_PATH}/drr
  "-a;20;-O;${PLM_TESTING_BUILD_DIR}/drr-a/out_;${PLM_TESTING_BUILD_DIR}/gauss-1.mha"
  )
plm_add_test (
  "drr-a-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/drr-a/out_0010.pfm"
  )
plmtest_check_interval ("drr-a-check"
  "${PLM_TESTING_BUILD_DIR}/drr-a-stats.stdout.txt"
  "MAX *([-0-9.]*)"
  "0.045"
  "0.055"
  )
set_tests_properties (drr-a PROPERTIES DEPENDS gauss-1)
set_tests_properties (drr-a-stats PROPERTIES DEPENDS drr-a)
set_tests_properties (drr-a-check PROPERTIES DEPENDS drr-a-stats)

plm_add_test (
  "drr-b"
  ""
  ${PLM_PLASTIMATCH_PATH}/drr
  "-a;20;-O;${PLM_TESTING_BUILD_DIR}/drr-b/out_;-i;uniform;${PLM_TESTING_BUILD_DIR}/gauss-1.mha"
  )
plm_add_test (
  "drr-b-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;drr-b/out_0010.pfm"
  )
plmtest_check_interval ("drr-b-check"
  "${PLM_TESTING_BUILD_DIR}/drr-b-stats.stdout.txt"
  "MAX *([-0-9.]*)"
  "0.045"
  "0.055"
  )
set_tests_properties (drr-b PROPERTIES DEPENDS gauss-1)
set_tests_properties (drr-b-stats PROPERTIES DEPENDS drr-b)
set_tests_properties (drr-b-check PROPERTIES DEPENDS drr-b-stats)

plm_add_test (
  "drr-c"
  ""
  ${PLM_PLASTIMATCH_PATH}/drr
  "-a;20;-O;${PLM_TESTING_BUILD_DIR}/drr-c/out_;${PLM_TESTING_BUILD_DIR}/gauss-4.mha"
  )
plm_add_test (
  "drr-c-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;drr-c/out_0010.pfm"
  )
plmtest_check_interval ("drr-c-check"
  "${PLM_TESTING_BUILD_DIR}/drr-c-stats.stdout.txt"
  "MAX *([-0-9.]*)"
  "0.045"
  "0.055"
  )
set_tests_properties (drr-c PROPERTIES DEPENDS gauss-4)
set_tests_properties (drr-c-stats PROPERTIES DEPENDS drr-c)
set_tests_properties (drr-c-check PROPERTIES DEPENDS drr-c-stats)

plm_add_test (
  "drr-d"
  ""
  ${PLM_PLASTIMATCH_PATH}/drr
  "-a;20;-O;${PLM_TESTING_BUILD_DIR}/drr-d/out_;-i;uniform;${PLM_TESTING_BUILD_DIR}/gauss-4.mha"
  )
plm_add_test (
  "drr-d-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;drr-d/out_0010.pfm"
  )
plmtest_check_interval ("drr-d-check"
  "${PLM_TESTING_BUILD_DIR}/drr-d-stats.stdout.txt"
  "MAX *([-0-9.]*)"
  "0.045"
  "0.055"
  )
set_tests_properties (drr-d PROPERTIES DEPENDS gauss-1)
set_tests_properties (drr-d-stats PROPERTIES DEPENDS drr-d)
set_tests_properties (drr-d-check PROPERTIES DEPENDS drr-d-stats)

plm_add_test (
  "drr-cuda"
  ""
  ${PLM_PLASTIMATCH_PATH}/drr
  "-A;cuda;-a;20;-O;${PLM_TESTING_BUILD_DIR}/drr-cuda/out_;${PLM_TESTING_BUILD_DIR}/gauss-1.mha"
  )
plm_add_test (
  "drr-cuda-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/drr-cuda/out_0010.pfm"
  )
plmtest_check_interval ("drr-cuda-check"
  "${PLM_TESTING_BUILD_DIR}/drr-cuda-stats.stdout.txt"
  "MAX *([-0-9.]*)"
  "0.045"
  "0.055"
  )
set_tests_properties (drr-cuda PROPERTIES DEPENDS gauss-1)
set_tests_properties (drr-cuda-stats PROPERTIES DEPENDS drr-cuda)
set_tests_properties (drr-cuda-check PROPERTIES DEPENDS drr-cuda-stats)

plm_add_test (
  "drr-opencl"
  ""
  ${PLM_PLASTIMATCH_PATH}/drr
  "-A;opencl;-a;20;-O;${PLM_TESTING_BUILD_DIR}/drr-opencl/out_;${PLM_TESTING_BUILD_DIR}/gauss-1.mha"
  )
plm_add_test (
  "drr-opencl-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/drr-opencl/out_0010.pfm"
  )
plmtest_check_interval ("drr-opencl-check"
  "${PLM_TESTING_BUILD_DIR}/drr-opencl-stats.stdout.txt"
  "MAX *([-0-9.]*)"
  "0.045"
  "0.055"
  )
set_tests_properties (drr-opencl PROPERTIES DEPENDS gauss-1)
set_tests_properties (drr-opencl-stats PROPERTIES DEPENDS drr-opencl)
set_tests_properties (drr-opencl-check PROPERTIES DEPENDS drr-opencl-stats)

## -------------------------------------------------------------------------
## fdk - all tests using fdk filtered backprojection
##   fdk-cpu-a     CPU reconstruction of synthetic data
##   fdk-cpu-b     CPU reconstruction of Varian data, no filtering
##   fdk-cpu-c     CPU reconstruction of Varian data, fftw filtering
##   fdk-brook     GPU reconstruction of synthetic data using BROOK
##   fdk-cuda-a    GPU reconstruction of synthetic data using CUDA
##    [fdk-cuda-b  GPU reconstruction of Varian data using CUDA
##   fdk-opencl-a  GPU reconstruction of synthetic data using OpenCL
## -------------------------------------------------------------------------
plm_add_test (
  "fdk-cpu-a"
  ""
  ${PLM_PLASTIMATCH_PATH}/fdk
  "-I;${PLM_TESTING_BUILD_DIR}/drr-a;-f;none;-a;0 19;-O;${PLM_TESTING_BUILD_DIR}/fdk-cpu-a.mha"
  )
plm_add_test (
  "fdk-cpu-a-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/fdk-cpu-a.mha"
  )
plmtest_check_interval ("fdk-cpu-a-check"
  "${PLM_TESTING_BUILD_DIR}/fdk-cpu-a-stats.stdout.txt"
  "MAX *([-0-9.]*)"
  "-987.5"
  "-986.5"
  )
set_tests_properties (fdk-cpu-a PROPERTIES DEPENDS drr-a)
set_tests_properties (fdk-cpu-a-stats PROPERTIES DEPENDS fdk-cpu-a)
set_tests_properties (fdk-cpu-a-check PROPERTIES DEPENDS fdk-cpu-a-stats)

plm_add_test (
  "fdk-cpu-b"
  ""
  ${PLM_PLASTIMATCH_PATH}/fdk
  "-I;${PLM_TESTING_BUILD_DIR}/varian-catphan;-f;none;-a;0 20 670;-O;${PLM_TESTING_BUILD_DIR}/fdk-cpu-b.mha"
  )
plm_add_test (
  "fdk-cpu-b-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/fdk-cpu-b.mha"
  )
plmtest_check_interval ("fdk-cpu-b-check"
  "${PLM_TESTING_BUILD_DIR}/fdk-cpu-b-stats.stdout.txt"
  "AVE *([-0-9.]*)"
  "-821.5"
  "-820.5"
  )
set_tests_properties (fdk-cpu-b-stats PROPERTIES DEPENDS fdk-cpu-b)
set_tests_properties (fdk-cpu-b-check PROPERTIES DEPENDS fdk-cpu-b-stats)

plm_add_test (
  "fdk-cpu-c"
  ""
  ${PLM_PLASTIMATCH_PATH}/fdk
  "-I;${PLM_TESTING_BUILD_DIR}/varian-catphan;-f;ramp;-a;0 50 670;-O;${PLM_TESTING_BUILD_DIR}/fdk-cpu-c.mha"
  )
plm_add_test (
  "fdk-cpu-c-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/fdk-cpu-c.mha"
  )
plmtest_check_interval ("fdk-cpu-c-check"
  "${PLM_TESTING_BUILD_DIR}/fdk-cpu-c-stats.stdout.txt"
  "AVE *([-0-9.]*)"
  "-990.0"
  "-989.0"
  )
set_tests_properties (fdk-cpu-c-stats PROPERTIES DEPENDS fdk-cpu-c)
set_tests_properties (fdk-cpu-c-check PROPERTIES DEPENDS fdk-cpu-c-stats)

plm_add_test (
  "fdk-cuda-a"
  ""
  ${PLM_PLASTIMATCH_PATH}/fdk
  "-I;${PLM_TESTING_BUILD_DIR}/drr-a;-f;none;-A;cuda;-a;0 19;-O;${PLM_TESTING_BUILD_DIR}/fdk-cuda-a.mha"
  )
plm_add_test (
  "fdk-cuda-a-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/fdk-cuda-a.mha"
  )
plmtest_check_interval ("fdk-cuda-a-check"
  "${PLM_TESTING_BUILD_DIR}/fdk-cuda-a-stats.stdout.txt"
  "MAX *([-0-9.]*)"
  "-987.5"
  "-986.5"
  )
set_tests_properties (fdk-cuda-a PROPERTIES DEPENDS drr-a)
set_tests_properties (fdk-cuda-a-stats PROPERTIES DEPENDS fdk-cuda-a)
set_tests_properties (fdk-cuda-a-check PROPERTIES DEPENDS fdk-cuda-a-stats)

plm_add_test (
  "fdk-opencl-a"
  ""
  ${PLM_PLASTIMATCH_PATH}/fdk
  "-I;${PLM_TESTING_BUILD_DIR}/drr-a;-f;none;-A;opencl;-a;0 19;-O;${PLM_TESTING_BUILD_DIR}/fdk-opencl-a.mha"
  )
plm_add_test (
  "fdk-opencl-a-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/fdk-opencl-a.mha"
  )
plmtest_check_interval ("fdk-opencl-a-check"
  "${PLM_TESTING_BUILD_DIR}/fdk-opencl-a-stats.stdout.txt"
  "MAX *([-0-9.]*)"
  "-987.5"
  "-986.5"
  )
set_tests_properties (fdk-opencl-a PROPERTIES DEPENDS drr-a)
set_tests_properties (fdk-opencl-a-stats PROPERTIES DEPENDS fdk-opencl-a)
set_tests_properties (fdk-opencl-a-check PROPERTIES DEPENDS fdk-opencl-a-stats)

## -------------------------------------------------------------------------
## fdk_tutorial - run the fdk tutorial
## -------------------------------------------------------------------------
plm_add_test (
  "fdk-tutorial-a"
  ""
  ${PLM_PLASTIMATCH_PATH}/drr
  "-t;pfm;-a;60;-N;3;-g;1000 1500;-r;100 100;-z;300 300;-I;${PLM_TESTING_BUILD_DIR}/headphantom/headphantom.mha;-O;${PLM_TESTING_BUILD_DIR}/fdk-tutorial-a/"
  )
plm_add_test (
  "fdk-tutorial-b"
  ""
  ${PLM_PLASTIMATCH_PATH}/fdk
  "-f;none;-r;100 100 100;-I;${PLM_TESTING_BUILD_DIR}/fdk-tutorial-a;-O;${PLM_TESTING_BUILD_DIR}/fdk-tutorial-b.mha"
  )
plm_add_test (
  "fdk-tutorial-b-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/fdk-tutorial-b.mha"
  )
plmtest_check_interval ("fdk-tutorial-b-check-1"
  "${PLM_TESTING_BUILD_DIR}/fdk-tutorial-b-stats-1.stdout.txt"
  "AVE *([-0-9.]*)"
  "-955.8"
  "-954.8"
  )
plm_add_test (
  "fdk-tutorial-c"
  ""
  ${PLM_PLASTIMATCH_PATH}/fdk
  "-f;ramp;-r;100 100 100;-I;${PLM_TESTING_BUILD_DIR}/fdk-tutorial-a;-O;${PLM_TESTING_BUILD_DIR}/fdk-tutorial-c.mha"
  )
plm_add_test (
  "fdk-tutorial-c-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/fdk-tutorial-c.mha"
  )
plmtest_check_interval ("fdk-tutorial-c-check-1"
  "${PLM_TESTING_BUILD_DIR}/fdk-tutorial-c-stats-1.stdout.txt"
  "AVE *([-0-9.]*)"
  "-999.0"
  "-998.0"
  )
set_tests_properties (fdk-tutorial-b PROPERTIES DEPENDS fdk-tutorial-a)
set_tests_properties (fdk-tutorial-b-stats-1 PROPERTIES DEPENDS fdk-tutorial-b)
set_tests_properties (fdk-tutorial-b-check-1 PROPERTIES 
  DEPENDS fdk-tutorial-b-stats-1)
set_tests_properties (fdk-tutorial-c PROPERTIES DEPENDS fdk-tutorial-a)
set_tests_properties (fdk-tutorial-c-stats-1 PROPERTIES DEPENDS fdk-tutorial-c)
set_tests_properties (fdk-tutorial-c-check-1 PROPERTIES 
  DEPENDS fdk-tutorial-c-stats-1)

## -------------------------------------------------------------------------
## landmark_warp a: itk tps, slicer fcsv
## landmark_warp b: wendland rbf, slicer fcsv
## landmark_warp c: nsh gauss rbf, slicer fcsv
## -------------------------------------------------------------------------
plm_add_test (
  "landmark-warp-a" 
  ""
  ${PLM_PLASTIMATCH_PATH}/landmark_warp
  "-a;tps;-f;${PLM_TESTING_DATA_DIR}/fiducials-rect-2.fcsv;-m;${PLM_TESTING_DATA_DIR}/fiducials-rect-3.fcsv;-I;${PLM_TESTING_BUILD_DIR}/rect-3.mha;-O;${PLM_TESTING_BUILD_DIR}/landmark-warp-a-img.mha;-V;${PLM_TESTING_BUILD_DIR}/landmark-warp-a-vf.mha"
  )
plm_add_test (
  "landmark-warp-a-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/landmark-warp-a-img.mha"
  )
plmtest_check_interval ("landmark-warp-a-check-1"
  "${PLM_TESTING_BUILD_DIR}/landmark-warp-a-stats-1.stdout.txt"
  "AVE *([-0-9.]*)"
  "-992.0"
  "-991.5"
  )
plm_add_test (
  "landmark-warp-a-stats-2"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/landmark-warp-a-vf.mha"
  )
plmtest_check_interval ("landmark-warp-a-check-2"
  "${PLM_TESTING_BUILD_DIR}/landmark-warp-a-stats-2.stdout.txt"
  "Mean: *([-0-9.]*)"
  "-23.2"
  "-22.7"
  )
set_tests_properties (landmark-warp-a PROPERTIES DEPENDS rect-3)
set_tests_properties (landmark-warp-a-stats-1 PROPERTIES 
  DEPENDS landmark-warp-a)
set_tests_properties (landmark-warp-a-check-1 PROPERTIES 
  DEPENDS landmark-warp-a-stats-1)
set_tests_properties (landmark-warp-a-stats-2 PROPERTIES 
  DEPENDS landmark-warp-a)
set_tests_properties (landmark-warp-a-check-1 PROPERTIES 
  DEPENDS landmark-warp-a-stats-2)

plm_add_test (
  "landmark-warp-b" 
  ""
  ${PLM_PLASTIMATCH_PATH}/landmark_warp
  "-a;wendland;-f;${PLM_TESTING_DATA_DIR}/fiducials-rect-2.fcsv;-m;${PLM_TESTING_DATA_DIR}/fiducials-rect-3.fcsv;-I;${PLM_TESTING_BUILD_DIR}/rect-3.mha;-O;${PLM_TESTING_BUILD_DIR}/landmark-warp-b-img.mha;-V;${PLM_TESTING_BUILD_DIR}/landmark-warp-b-vf.mha;-r;90;-d;0;-Y;0.0"
  )
plm_add_test (
  "landmark-warp-b-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/landmark-warp-b-img.mha"
  )
plmtest_check_interval ("landmark-warp-b-check-1"
  "${PLM_TESTING_BUILD_DIR}/landmark-warp-b-stats-1.stdout.txt"
  "AVE *([-0-9.]*)"
  "-992.2"
  "-991.8"
  )
plm_add_test (
  "landmark-warp-b-stats-2"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/landmark-warp-b-vf.mha"
  )
plmtest_check_interval ("landmark-warp-b-check-2"
  "${PLM_TESTING_BUILD_DIR}/landmark-warp-b-stats-2.stdout.txt"
  "Mean: *([-0-9.]*)"
  "-1.55"
  "-1.4"
  )
set_tests_properties (landmark-warp-b PROPERTIES DEPENDS rect-3)
set_tests_properties (landmark-warp-b-stats-1 PROPERTIES 
  DEPENDS landmark-warp-b)
set_tests_properties (landmark-warp-b-check-1 PROPERTIES 
  DEPENDS landmark-warp-b-stats-1)
set_tests_properties (landmark-warp-b-stats-2 PROPERTIES 
  DEPENDS landmark-warp-b)
set_tests_properties (landmark-warp-b-check-2 PROPERTIES 
  DEPENDS landmark-warp-b-stats-2)

plm_add_test (
  "landmark-warp-c" 
  ""
  ${PLM_PLASTIMATCH_PATH}/landmark_warp
  "-a;gauss;-f;${PLM_TESTING_DATA_DIR}/fiducials-rect-2.fcsv;-m;${PLM_TESTING_DATA_DIR}/fiducials-rect-3.fcsv;-I;${PLM_TESTING_BUILD_DIR}/rect-3.mha;-O;${PLM_TESTING_BUILD_DIR}/landmark-warp-c-img.mha;-V;${PLM_TESTING_BUILD_DIR}/landmark-warp-c-vf.mha;-r;74;-d;0;-Y;1.0"
  )
plm_add_test (
  "landmark-warp-c-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/landmark-warp-c-img.mha"
  )
plmtest_check_interval ("landmark-warp-c-check-1"
  "${PLM_TESTING_BUILD_DIR}/landmark-warp-c-stats-1.stdout.txt"
  "AVE *([-0-9.]*)"
  "-992.0"
  "-991.5"
  )
plm_add_test (
  "landmark-warp-c-stats-2"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/landmark-warp-c-vf.mha"
  )
plmtest_check_interval ("landmark-warp-c-check-2"
  "${PLM_TESTING_BUILD_DIR}/landmark-warp-c-stats-2.stdout.txt"
  "Mean: *([-0-9.]*)"
  "-1.6"
  "-1.5"
  )
set_tests_properties (landmark-warp-c PROPERTIES DEPENDS rect-3)
set_tests_properties (landmark-warp-c-stats-1 PROPERTIES 
  DEPENDS landmark-warp-c)
set_tests_properties (landmark-warp-c-check-1 PROPERTIES 
  DEPENDS landmark-warp-c-stats-1)
set_tests_properties (landmark-warp-c-stats-2 PROPERTIES 
  DEPENDS landmark-warp-c)
set_tests_properties (landmark-warp-c-check-2 PROPERTIES 
  DEPENDS landmark-warp-c-stats-2)

## -------------------------------------------------------------------------
## plastimatch convert dicom (and dicom rt)
##   plm convert dicom a: ct only
##   plm convert dicom b: rtss only, with referenced-ct
##   plm convert dicom c: rtss only, with fixed image reference
##   plm convert dicom d: dose only, native sampling rate
##   plm convert dicom e: input xio dicom, output plastimatch dicom
##   plm convert dicom f: input plastimatch dicom, output plastimatch dicom
##   plm convert dicom g: input rtss, no fixed image, no ct reference
##   plm convert dicom h: set metadata
## -------------------------------------------------------------------------
plm_add_test (
  "plm-convert-dicom-a"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02;--output-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-a.mha"
  )
plm_add_test (
  "plm-convert-dicom-a-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-a.mha"
  )
plmtest_check_interval ("plm-convert-dicom-a-check"
  "${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-a-stats.stdout.txt"
  "AVE *([-0-9.]*)"
  "-667.5"
  "-666.6"
  )
set_tests_properties (plm-convert-dicom-a-stats PROPERTIES 
  DEPENDS plm-convert-dicom-a)
set_tests_properties (plm-convert-dicom-a-check PROPERTIES 
  DEPENDS plm-convert-dicom-a-stats)

plm_add_test (
  "plm-convert-dicom-b"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02/SS.rtp1.12.20080627A.5.dcm;--output-ss-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-b-ss-img.mha;--referenced-ct;${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02"
  )
plm_add_test (
  "plm-convert-dicom-b-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-b-ss-img.mha"
  )
if (PLM_USE_SS_IMAGE_VEC)
plmtest_check_interval ("plm-convert-dicom-b-check"
  "${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-b-stats.stdout.txt"
  "S *1 *NVOX *([-0-9.]*)"
  "62767"
  "62767"
  )
ELSE ()
plmtest_check_interval ("plm-convert-dicom-b-check"
  "${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-b-stats.stdout.txt"
  "AVE *([-0-9.]*)"
  "1.05"
  "1.06"
  )
endif ()
set_tests_properties (plm-convert-dicom-b-stats PROPERTIES 
  DEPENDS plm-convert-dicom-b)
set_tests_properties (plm-convert-dicom-b-check PROPERTIES 
  DEPENDS plm-convert-dicom-b-stats)

plm_add_test (
  "plm-convert-dicom-c"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02/SS.rtp1.12.20080627A.5.dcm;--output-ss-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-c-ss-img.mha;--fixed;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-a.mha;--output-cxt;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-c.cxt"
  )
plm_add_test (
  "plm-convert-dicom-c-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-c-ss-img.mha"
  )
IF (PLM_USE_SS_IMAGE_VEC)
# To be resolved.  Why do dicom-b and dicom-c give different answers?
plmtest_check_interval ("plm-convert-dicom-c-check"
  "${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-c-stats.stdout.txt"
  "S *1 *NVOX *([-0-9.]*)"
  "62769"
  "62769"
  )
ELSE ()
plmtest_check_interval ("plm-convert-dicom-c-check"
  "${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-c-stats.stdout.txt"
  "AVE *([-0-9.]*)"
  "1.05"
  "1.06"
  )
ENDIF ()
set_tests_properties (plm-convert-dicom-c PROPERTIES 
  DEPENDS plm-convert-dicom-a)
set_tests_properties (plm-convert-dicom-c-stats PROPERTIES 
  DEPENDS plm-convert-dicom-c)
set_tests_properties (plm-convert-dicom-c-check PROPERTIES 
  DEPENDS plm-convert-dicom-c-stats)

plm_add_test (
  "plm-convert-dicom-d"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02/DOSE.20080627A.TRAINING4FLD.dcm;--output-dose-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-d-dose-img.mha"
  )
plm_add_test (
  "plm-convert-dicom-d-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-d-dose-img.mha"
  )
plmtest_check_interval ("plm-convert-dicom-d-check"
  "${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-d-stats.stdout.txt"
  "MAX *([-0-9.]*)"
  "42.1"
  "42.2"
  )
set_tests_properties (plm-convert-dicom-d-stats PROPERTIES 
  DEPENDS plm-convert-dicom-d)
set_tests_properties (plm-convert-dicom-d-check PROPERTIES 
  DEPENDS plm-convert-dicom-d-stats)

plm_add_test (
  "plm-convert-dicom-e" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02;--output-dicom;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-e-dicom;--output-dose-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-e-dose-img.mha"
  )

plm_add_test (
  "plm-convert-dicom-f" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-e-dicom;--output-dicom;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-f-dicom;--output-dose-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-f-dose-img.mha"
  )
plm_add_test (
  "plm-convert-dicom-f-stats" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "compare;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-e-dose-img.mha;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-f-dose-img.mha"
  )
plmtest_check_interval ("plm-convert-dicom-f-check"
  "${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-f-stats.stdout.txt"
  "MAE *([-0-9.]*)"
  "0.000"
  "0.005"
  )
set_tests_properties (plm-convert-dicom-f PROPERTIES 
  DEPENDS plm-convert-dicom-e)
set_tests_properties (plm-convert-dicom-f-stats PROPERTIES 
  DEPENDS plm-convert-dicom-f)
set_tests_properties (plm-convert-dicom-f-check PROPERTIES 
  DEPENDS plm-convert-dicom-f-stats)

plm_add_test (
  "plm-convert-dicom-g" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/rect-1-dicom/ss.dcm;--output-ss-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-g.mha;--output-ss-list;plm-convert-dicom-g.txt;--output-cxt;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-g.cxt"
  )
set_tests_properties (plm-convert-dicom-g PROPERTIES DEPENDS rect-1)

plm_add_test (
  "plm-convert-dicom-h"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/rect-1-dicom;--output-dicom;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-h-dicom;--metadata;0018,5100=FFS"
  )
plm_add_test (
  "plm-convert-dicom-h-stats" 
  ""
  ${PLM_PLASTIMATCH_PATH}/dicom_info
  "--input;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-h-dicom/image000.dcm"
  )
plmtest_check_string ("plm-convert-dicom-h-check" 
  "${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-h-stats.stdout.txt"
  "Patient Position . (.*[^ ]) *$"
  "FFS"
  )
set_tests_properties (plm-convert-dicom-h PROPERTIES DEPENDS rect-1)
set_tests_properties (plm-convert-dicom-h-stats PROPERTIES 
  DEPENDS plm-convert-dicom-h)
set_tests_properties (plm-convert-dicom-h-check PROPERTIES 
  DEPENDS plm-convert-dicom-h-stats)

## -------------------------------------------------------------------------
## plastimatch-convert-dicom-donut
## -------------------------------------------------------------------------
plm_add_test (
  "plm-convert-dicom-donut"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/donut-1-dicom;--output-ss-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-donut-ss.nrrd"
  )
plm_add_test (
  "plm-convert-dicom-donut-stats" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-donut-ss.nrrd"
  )
set_tests_properties (plm-convert-dicom-donut PROPERTIES DEPENDS donut-1)
set_tests_properties (plm-convert-dicom-donut-stats PROPERTIES 
  DEPENDS plm-convert-dicom-donut)

## -------------------------------------------------------------------------
## plastimatch convert dicom-rt (aw)
## -------------------------------------------------------------------------
plm_add_test (
  "plm-convert-dicom-aw-a" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/foot-dicomrt-aw-4.4/20101202HFS;--output-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-aw-a.mha;--output-ss-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-aw-a-ss.mha;--output-ss-list;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-aw-a-ss.txt;--output-dose-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-aw-a-dose.mha;--output-dicom;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-aw-a-dicom"
  )
plm_add_test (
  "plm-convert-dicom-aw-a-stats" 
  ""
  ${PLM_PLASTIMATCH_PATH}/dicom_info
  "--input;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-aw-a-dicom/image000.dcm"
  )
plmtest_check_string ("plm-convert-dicom-aw-a-check" 
  "${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-aw-a-stats.stdout.txt"
  "^Patient's Name . (.*[^ ]) *$"
  "PHANTOM^HFS"
  )
plm_add_test (
  "plm-convert-dicom-aw-b" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/foot-dicomrt-aw-4.4/20101202FFS;--output-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-aw-b.mha;--output-ss-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-aw-b-ss.mha;--output-ss-list;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-aw-b-ss.txt;--output-dose-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-aw-b-dose.mha"
  )
set_tests_properties (plm-convert-dicom-aw-a-stats PROPERTIES 
  DEPENDS plm-convert-dicom-aw-a)
set_tests_properties (plm-convert-dicom-aw-a-check PROPERTIES 
  DEPENDS plm-convert-dicom-aw-a-stats)

## -------------------------------------------------------------------------
## plastimatch convert dicom-rt (cerr)
## -------------------------------------------------------------------------
plm_add_test (
  "plm-convert-dicom-cerr-a" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-CERR4pt0beta2_25_Jan_2011;--output-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-cerr-a.mha;--output-ss-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-cerr-a-ss.mha;--output-ss-list;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-cerr-a-ss.txt;--output-dose-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-cerr-a-dose.mha"
  )

## -------------------------------------------------------------------------
## plastimatch convert dicom-rt (xio, irregular)
## -------------------------------------------------------------------------
plm_add_test (
  "plm-convert-dicom-irregular-a"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/xio-4.60-irregular-spacing;--output-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-irregular-a.mha;--output-ss-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-irregular-a-ss.mha;--output-ss-list;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-irregular-a-ss.txt;--output-dose-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-irregular-a-dose.mha"
  )

## -------------------------------------------------------------------------
## plastimatch convert dicom-rt (pinnacle)
## -------------------------------------------------------------------------
plm_add_test (
  "plm-convert-dicom-pinnacle-a" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/rando-dicomrt-pinnacle3-8.2g;--output-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-pinnacle-a.mha;--output-ss-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-pinnacle-a-ss.mha;--output-ss-list;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-pinnacle-a-ss.txt;--output-dose-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-pinnacle-a-dose.mha"
  )

## -------------------------------------------------------------------------
## plastimatch convert dicom-rt (xio)
## -------------------------------------------------------------------------
plm_add_test (
  "plm-convert-dicom-xio-a"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02;--output-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-xio-a.mha;--output-ss-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-xio-a-ss.mha;--output-ss-list;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-xio-a-ss.txt;--output-dose-img;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-xio-a-dose.mha"
  )

## -------------------------------------------------------------------------
## plastimatch-convert-cxt
## -------------------------------------------------------------------------
plm_add_test (
  "plm-convert-cxt"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/plm-convert-dicom-c.cxt;--output-ss-img;${PLM_TESTING_BUILD_DIR}/plm-convert-cxt.mha;--output-ss-list;${PLM_TESTING_BUILD_DIR}/plm-convert-cxt.txt"
  )
plm_add_test (
  "plm-convert-cxt-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-convert-cxt.mha"
  )
IF (PLM_USE_SS_IMAGE_VEC)
plmtest_check_interval ("plm-convert-cxt-check"
  "${PLM_TESTING_BUILD_DIR}/plm-convert-cxt-stats.stdout.txt"
  "S *1 *NVOX *([-0-9.]*)"
  "62769"
  "62769"
  )
ELSE ()
plmtest_check_interval ("plm-convert-cxt-check"
  "${PLM_TESTING_BUILD_DIR}/plm-convert-cxt-stats.stdout.txt"
  "AVE *([-0-9.]*)"
  "1.05"
  "1.06"
  )
ENDIF ()
set_tests_properties (plm-convert-cxt PROPERTIES DEPENDS plm-convert-dicom-c)
set_tests_properties (plm-convert-cxt-stats PROPERTIES DEPENDS plm-convert-cxt)
set_tests_properties (plm-convert-cxt-check PROPERTIES 
  DEPENDS plm-convert-cxt-stats)

## -------------------------------------------------------------------------
## plastimatch convert xio
## -------------------------------------------------------------------------
plm_add_test (
  "plm-convert-xio"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-xio-4.33.02;--output-img;${PLM_TESTING_BUILD_DIR}/chest-phantom-xio.mha;--output-xio;${PLM_TESTING_BUILD_DIR}/plm-convert-xio-xio-output;--output-dicom;${PLM_TESTING_BUILD_DIR}/plm-convert-xio-dicom"
  )
plm_add_test (
  "plm-convert-xio-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-xio.mha"
  )
plmtest_check_interval ("plm-convert-xio-check"
  "${PLM_TESTING_BUILD_DIR}/plm-convert-xio-stats.stdout.txt"
  "AVE *([-0-9.]*)"
  "-667.5"
  "-666.6"
  )
set_tests_properties (plm-convert-xio-stats PROPERTIES DEPENDS plm-convert-xio)
set_tests_properties (plm-convert-xio-check PROPERTIES 
  DEPENDS plm-convert-xio-stats)

## -------------------------------------------------------------------------
## plastimatch-convert-prefix-fcsv
## -------------------------------------------------------------------------
plm_add_test (
  "plm-convert-prefix-fcsv"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "convert;--input-ss-img;${PLM_TESTING_BUILD_DIR}/donut-ss-1.mha;--output-prefix-fcsv;${PLM_TESTING_BUILD_DIR}/plm-convert-prefix-fcsv"
  )
set_tests_properties (plm-convert-prefix-fcsv PROPERTIES DEPENDS donut-1)

## -------------------------------------------------------------------------
## plastimatch fill/mask
## -------------------------------------------------------------------------
plm_add_test (
  "plm-fill-a"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "fill;--input;${PLM_TESTING_BUILD_DIR}/rect-1.mha;--output;${PLM_TESTING_BUILD_DIR}/plm-fill-a.mha;--mask;${PLM_TESTING_BUILD_DIR}/rect-4.mha"
  )
plm_add_test (
  "plm-fill-a-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-fill-a.mha"
  )
plmtest_check_interval ("plm-fill-a-check"
  "${PLM_TESTING_BUILD_DIR}/plm-fill-a-stats.stdout.txt"
  "AVE *([-0-9.]*)"
  "-0.1"
  "+0.1"
  )
set_tests_properties (plm-fill-a PROPERTIES DEPENDS "rect-1;rect-4")
set_tests_properties (plm-fill-a-stats PROPERTIES DEPENDS plm-fill-a)
set_tests_properties (plm-fill-a-check PROPERTIES DEPENDS plm-fill-a-stats)

plm_add_test (
  "plm-mask-a"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "mask;--input;${PLM_TESTING_BUILD_DIR}/rect-1.mha;--output;${PLM_TESTING_BUILD_DIR}/plm-mask-a.mha;--mask;${PLM_TESTING_BUILD_DIR}/rect-4.mha;--mask-value;-1000"
  )
plm_add_test (
  "plm-mask-a-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-mask-a.mha"
  )
plmtest_check_interval ("plm-mask-a-check"
  "${PLM_TESTING_BUILD_DIR}/plm-mask-a-stats.stdout.txt"
  "AVE *([-0-9.]*)"
  "-1000.1"
  "-999.9"
  )
set_tests_properties (plm-mask-a PROPERTIES DEPENDS "rect-1;rect-4")
set_tests_properties (plm-mask-a-stats PROPERTIES DEPENDS plm-mask-a)
set_tests_properties (plm-mask-a-check PROPERTIES DEPENDS plm-mask-a-stats)

## -------------------------------------------------------------------------
## plastimatch usage
## -------------------------------------------------------------------------
plm_add_test (
  "plm-usage" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  ""
  )

## -------------------------------------------------------------------------
## plastimatch register (group 1)
##   align-center
##   itk-translation
##   itk-rigid
##   itk-bspline
##   itk-demons
## -------------------------------------------------------------------------
if (CMAKE_Fortran_COMPILER_WORKS OR F2C_LIBRARY)
  # For Nocedal optimizer
  set (PLM_BSPLINE_TEST_LOWER_THRESH "-746.0")
  set (PLM_BSPLINE_TEST_UPPER_THRESH "-744.5")
else ()
  # For Liblbfgs optimizer
  set (PLM_BSPLINE_TEST_LOWER_THRESH "-706.0")
  set (PLM_BSPLINE_TEST_UPPER_THRESH "-704.5")
endif ()

plm_add_test (
  "plm-reg-align-center"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-reg-align-center.txt"
  )
plm_add_test (
  "plm-reg-align-center-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "compare;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;${PLM_TESTING_BUILD_DIR}/plm-reg-align-center-img.mha"
  )
plmtest_check_interval ("plm-reg-align-center-check"
  "${PLM_TESTING_BUILD_DIR}/plm-reg-align-center-stats.stdout.txt"
  "MAE *([-0-9.]*)"
  "-0.005"
  "0.005"
  )
set_tests_properties (plm-reg-align-center PROPERTIES 
  DEPENDS "gauss-1;gauss-6")
set_tests_properties (plm-reg-align-center-stats PROPERTIES 
  DEPENDS plm-reg-align-center)
set_tests_properties (plm-reg-align-center-check PROPERTIES 
  DEPENDS plm-reg-align-center-stats)

plm_add_test (
  "plm-reg-itk-translation"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-reg-itk-translation.txt"
  )
plm_add_test (
  "plm-reg-itk-translation-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-reg-itk-translation-img.mha"
  )
plmtest_check_interval ("plm-reg-itk-translation-check"
  "${PLM_TESTING_BUILD_DIR}/plm-reg-itk-translation-stats.stdout.txt"
  "AVE *([-0-9.]*)"
  "-637.0"
  "-636.9"
  )
set_tests_properties (plm-reg-itk-translation PROPERTIES 
  DEPENDS "gauss-1;gauss-2")
set_tests_properties (plm-reg-itk-translation-stats PROPERTIES 
  DEPENDS plm-reg-itk-translation)
set_tests_properties (plm-reg-itk-translation-check PROPERTIES 
  DEPENDS plm-reg-itk-translation-stats)

plm_add_test (
  "plm-reg-itk-rigid"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-reg-itk-rigid.txt"
  )
plm_add_test (
  "plm-reg-itk-rigid-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-reg-itk-rigid-intermediate-img.mha"
  )
plmtest_check_interval ("plm-reg-itk-rigid-check-1"
  "${PLM_TESTING_BUILD_DIR}/plm-reg-itk-rigid-stats-1.stdout.txt"
  "AVE *([-0-9.]*)"
  "-965.7"
  "-965.4"
  )
plm_add_test (
  "plm-reg-itk-rigid-stats-2"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-reg-itk-rigid-img.mha"
  )
plmtest_check_interval ("plm-reg-itk-rigid-check-2"
  "${PLM_TESTING_BUILD_DIR}/plm-reg-itk-rigid-stats-2.stdout.txt"
  "AVE *([-0-9.]*)"
  "-965.7"
  "-965.4"
  )
set_tests_properties (plm-reg-itk-rigid PROPERTIES 
  DEPENDS "rect-2;rect-3")
set_tests_properties (plm-reg-itk-rigid-stats-1 PROPERTIES 
  DEPENDS plm-reg-itk-rigid)
set_tests_properties (plm-reg-itk-rigid-check-1 PROPERTIES 
  DEPENDS plm-reg-itk-rigid-stats-1)
set_tests_properties (plm-reg-itk-rigid-stats-2 PROPERTIES 
  DEPENDS plm-reg-itk-rigid)
set_tests_properties (plm-reg-itk-rigid-check-2 PROPERTIES 
  DEPENDS plm-reg-itk-rigid-stats-2)

plm_add_test (
  "plm-reg-itk-bspline"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-reg-itk-bspline.txt"
  )
plm_add_test (
  "plm-reg-itk-bspline-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-reg-itk-bspline-img.mha"
  )
plmtest_check_interval ("plm-reg-itk-bspline-check-1"
  "${PLM_TESTING_BUILD_DIR}/plm-reg-itk-bspline-stats-1.stdout.txt"
  "AVE *([-0-9.]*)"
  "-763.85"
  "-763.75"
  )
set_tests_properties (plm-reg-itk-bspline PROPERTIES 
  DEPENDS "gauss-1;gauss-2")
set_tests_properties (plm-reg-itk-bspline-stats-1 PROPERTIES 
  DEPENDS plm-reg-itk-bspline)
set_tests_properties (plm-reg-itk-bspline-check-1 PROPERTIES 
  DEPENDS plm-reg-itk-bspline-stats-1)

plm_add_test (
  "plm-reg-itk-demons"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-reg-itk-demons.txt"
  )
plmtest_check_interval ("plm-reg-itk-demons-check-1"
  "${PLM_TESTING_BUILD_DIR}/plm-reg-itk-demons.stdout.txt"
  "VF_AVG +[^ ] +([-0-9.]*)"
  "8.55"
  "8.65"
  )
plm_add_test (
  "plm-reg-itk-demons-stats-2"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-reg-itk-demons-img.mha"
  )
plmtest_check_interval ("plm-reg-itk-demons-check-2"
  "${PLM_TESTING_BUILD_DIR}/plm-reg-itk-demons-stats-2.stdout.txt"
  "AVE *([-0-9.]*)"
  "-853.0"
  "-852.6"
  )
set_tests_properties (plm-reg-itk-demons PROPERTIES 
  DEPENDS "gauss-1;gauss-2")
set_tests_properties (plm-reg-itk-demons-check-1 PROPERTIES 
  DEPENDS plm-reg-itk-demons)
set_tests_properties (plm-reg-itk-demons-stats-2 PROPERTIES 
  DEPENDS plm-reg-itk-demons)
set_tests_properties (plm-reg-itk-demons-check-2 PROPERTIES 
  DEPENDS plm-reg-itk-demons-stats-2)

## -------------------------------------------------------------------------
## plastimatch register (group 2)
##   plm-bspline-single-c
##   plm-bspline-single-h
##   plm-bspline-single-openmp
##   plm-bspline-single-cuda
##   plm-reg-bspline-ushort
##   plm-reg-bspline-double
##   plm-reg-bspline-itk-output
##   plm-reg-bspline-char-output
## -------------------------------------------------------------------------
plm_add_test (
  "plm-bspline-single-c" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-bspline-single-c.txt"
  )
plm_add_test (
  "plm-bspline-single-c-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-bspline-single-c-img.mha"
  )
plmtest_check_interval ("plm-bspline-single-c-check"
  "${PLM_TESTING_BUILD_DIR}/plm-bspline-single-c-stats.stdout.txt"
  "AVE *([-0-9.]*)"
  "${PLM_BSPLINE_TEST_LOWER_THRESH}"
  "${PLM_BSPLINE_TEST_UPPER_THRESH}"
  )
set_tests_properties (plm-bspline-single-c PROPERTIES 
  DEPENDS "gauss-1;gauss-2")
set_tests_properties (plm-bspline-single-c-stats PROPERTIES 
  DEPENDS plm-bspline-single-c)
set_tests_properties (plm-bspline-single-c-check PROPERTIES 
  DEPENDS plm-bspline-single-c-stats)

plm_add_test (
  "plm-bspline-single-h" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-bspline-single-h.txt"
  )
plm_add_test (
  "plm-bspline-single-h-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-bspline-single-h-img.mha"
  )
plmtest_check_interval ("plm-bspline-single-h-check"
  "${PLM_TESTING_BUILD_DIR}/plm-bspline-single-h-stats.stdout.txt"
  "AVE *([-0-9.]*)"
  "${PLM_BSPLINE_TEST_LOWER_THRESH}"
  "${PLM_BSPLINE_TEST_UPPER_THRESH}"
  )
set_tests_properties (plm-bspline-single-h PROPERTIES 
  DEPENDS "gauss-1;gauss-2")
set_tests_properties (plm-bspline-single-h-stats PROPERTIES 
  DEPENDS plm-bspline-single-h)
set_tests_properties (plm-bspline-single-h-check PROPERTIES 
  DEPENDS plm-bspline-single-h-stats)

plm_add_test (
  "plm-bspline-openmp" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-bspline-openmp.txt"
  )
plm_add_test (
  "plm-bspline-openmp-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-bspline-openmp-img.mha"
  )
plmtest_check_interval ("plm-bspline-openmp-check"
  "${PLM_TESTING_BUILD_DIR}/plm-bspline-openmp-stats.stdout.txt"
  "AVE *([-0-9.]*)"
  "${PLM_BSPLINE_TEST_LOWER_THRESH}"
  "${PLM_BSPLINE_TEST_UPPER_THRESH}"
  )
set_tests_properties (plm-bspline-openmp PROPERTIES 
  DEPENDS "gauss-1;gauss-2")
set_tests_properties (plm-bspline-openmp-stats PROPERTIES 
  DEPENDS plm-bspline-openmp)
set_tests_properties (plm-bspline-openmp-check PROPERTIES 
  DEPENDS plm-bspline-openmp-stats)

plm_add_test (
  "plm-bspline-cuda" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-bspline-cuda.txt"
  )
plm_add_test (
  "plm-bspline-cuda-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-bspline-cuda-img.mha"
  )
plmtest_check_interval ("plm-bspline-cuda-check"
  "${PLM_TESTING_BUILD_DIR}/plm-bspline-cuda-stats.stdout.txt"
  "AVE *([-0-9.]*)"
  "${PLM_BSPLINE_TEST_LOWER_THRESH}"
  "${PLM_BSPLINE_TEST_UPPER_THRESH}"
  )
set_tests_properties (plm-bspline-cuda PROPERTIES 
  DEPENDS "gauss-1;gauss-2")
set_tests_properties (plm-bspline-cuda-stats PROPERTIES 
  DEPENDS plm-bspline-cuda)
set_tests_properties (plm-bspline-cuda-check PROPERTIES 
  DEPENDS plm-bspline-cuda-stats)

plm_add_test (
  "plm-reg-bspline-ushort"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-reg-bspline-ushort.txt"
  )
plm_add_test (
  "plm-reg-bspline-ushort-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-reg-bspline-ushort-img.mha"
  )
plmtest_check_interval ("plm-reg-bspline-ushort-check"
  "${PLM_TESTING_BUILD_DIR}/plm-reg-bspline-ushort-stats.stdout.txt"
  "AVE *([-0-9.]*)"
  "106.6"
  "107.6"
  )
set_tests_properties (plm-reg-bspline-ushort PROPERTIES 
  DEPENDS "gauss-ushort-1;gauss-ushort-2")
set_tests_properties (plm-reg-bspline-ushort-stats PROPERTIES 
  DEPENDS plm-reg-bspline-ushort)
set_tests_properties (plm-reg-bspline-ushort-check PROPERTIES 
  DEPENDS plm-reg-bspline-ushort-stats)

plm_add_test (
  "plm-reg-bspline-double"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-reg-bspline-double.txt"
  )
plm_add_test (
  "plm-reg-bspline-double-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-reg-bspline-double-img.mha"
  )
plmtest_check_interval ("plm-reg-bspline-double-check"
  "${PLM_TESTING_BUILD_DIR}/plm-reg-bspline-double-stats.stdout.txt"
  "AVE *([-0-9.]*)"
  "107.6"
  "108.6"
  )
set_tests_properties (plm-reg-bspline-double PROPERTIES 
  DEPENDS "gauss-double-1;gauss-double-2")
set_tests_properties (plm-reg-bspline-double-stats PROPERTIES 
  DEPENDS plm-reg-bspline-double)
set_tests_properties (plm-reg-bspline-double-check PROPERTIES 
  DEPENDS plm-reg-bspline-double-stats)

plm_add_test (
  "plm-reg-bspline-itk-output"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-reg-bspline-itk-output.txt"
  )
set_tests_properties (plm-reg-bspline-itk-output PROPERTIES 
  DEPENDS "gauss-1;gauss-2")

plm_add_test (
  "plm-reg-bspline-char-output"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-reg-bspline-char-output.txt"
  )
plm_add_test (
  "plm-reg-bspline-char-output-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;${PLM_TESTING_BUILD_DIR}/plm-reg-bspline-char-output.mha"
  )
plmtest_check_interval (
  "plm-reg-bspline-char-output-check"
  "${PLM_TESTING_BUILD_DIR}/plm-reg-bspline-char-output-stats.stdout.txt"
  "MIN *([-0-9.]*)"
  "-128.1"
  "-127.9"
  )
set_tests_properties (plm-reg-bspline-char-output PROPERTIES 
  DEPENDS "rect-2;rect-3")
set_tests_properties (plm-reg-bspline-char-output-stats PROPERTIES 
  DEPENDS plm-reg-bspline-char-output)
set_tests_properties (plm-reg-bspline-char-output-check PROPERTIES 
  DEPENDS plm-reg-bspline-char-output-stats)

## -------------------------------------------------------------------------
## plastimatch register (group 3)
##   register-bsp-rect
##   register-bsp-regularize-none
##   register-bsp-regularize-analytic
##   register-bsp-regularize-numeric
## -------------------------------------------------------------------------
plm_add_test (
  "plm-bsp-rect"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-bsp-rect.txt"
  )
set_tests_properties (plm-bsp-rect PROPERTIES DEPENDS "rect-2;rect-3")

plm_add_test (
  "plm-bsp-regularize-none" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-bsp-regularize-none.txt"
  )
plm_add_test (
  "plm-bsp-regularize-none-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-bsp-regularize-none-vf.mha"
  )
plmtest_check_interval (
  "plm-bsp-regularize-none-check"
  "${PLM_TESTING_BUILD_DIR}/plm-bsp-regularize-none-stats.stdout.txt"
  "MINJAC *([-0-9.]*)"
  "0.31"
  "0.32"
  )
set_tests_properties (plm-bsp-regularize-none PROPERTIES 
  DEPENDS "rect-3;sphere-2")
set_tests_properties (plm-bsp-regularize-none-stats PROPERTIES 
  DEPENDS plm-bsp-regularize-none)
set_tests_properties (plm-bsp-regularize-none-check PROPERTIES 
  DEPENDS plm-bsp-regularize-none-stats)

plm_add_test (
  "plm-bsp-regularize-analytic" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-bsp-regularize-analytic.txt"
  )
plm_add_test (
  "plm-bsp-regularize-analytic-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-bsp-regularize-analytic-vf.mha"
  )
plmtest_check_interval (
  "plm-bsp-regularize-analytic-check"
  "${PLM_TESTING_BUILD_DIR}/plm-bsp-regularize-analytic-stats.stdout.txt"
  "MINJAC *([-0-9.]*)"
  "0.415"
  "0.425"
  )
set_tests_properties (plm-bsp-regularize-analytic PROPERTIES 
  DEPENDS "rect-3;sphere-2")
set_tests_properties (plm-bsp-regularize-analytic-stats PROPERTIES 
  DEPENDS plm-bsp-regularize-analytic)
set_tests_properties (plm-bsp-regularize-analytic-check PROPERTIES 
  DEPENDS plm-bsp-regularize-analytic-stats)

plm_add_test (
  "plm-bsp-regularize-numeric" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-bsp-regularize-numeric.txt"
  )
plm_add_test (
  "plm-bsp-regularize-numeric-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-bsp-regularize-numeric-vf.mha"
  )
plmtest_check_interval (
  "plm-bsp-regularize-numeric-check"
  "${PLM_TESTING_BUILD_DIR}/plm-bsp-regularize-numeric-stats.stdout.txt"
  "MINJAC *([-0-9.]*)"
  "0.440"
  "0.445"
  )
set_tests_properties (plm-bsp-regularize-numeric PROPERTIES 
  DEPENDS "rect-3;sphere-2")
set_tests_properties (plm-bsp-regularize-numeric-stats PROPERTIES 
  DEPENDS plm-bsp-regularize-numeric)
set_tests_properties (plm-bsp-regularize-numeric-check PROPERTIES 
  DEPENDS plm-bsp-regularize-numeric-stats)

## -------------------------------------------------------------------------
## plastimatch compose
##   This test uses registers from gauss-1 to gauss-2 which are done above, 
##   and composes with results from an registration from gauss-2 to 
##   gauss-5.  The composed vf should be correctly align gauss-1 to gauss-5.
##   
##   plm-reg-compose    Make vf for testing composition
##   plm-compose-a           Compose translation and vector field
##   plm-compose-b           Compose vector field and vector field
##   plm-compose-c           Compose bspline and vector field
##   
##   Remark: these tests are correct, in the sense that they work as 
##   implemented.  Transformation vectors which are not defined are 
##   set to zero in plastimatch, and this causes artifacts in the composed 
##   vector field.
## -------------------------------------------------------------------------
plm_add_test (
  "plm-reg-compose" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "${PLM_TESTING_DATA_DIR}/plm-reg-compose.txt"
  )
set_tests_properties (plm-reg-compose PROPERTIES DEPENDS "gauss-2;gauss-5")

plm_add_test (
  "plm-compose-a" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "compose;${PLM_TESTING_BUILD_DIR}/plm-reg-itk-translation-xf.txt;${PLM_TESTING_BUILD_DIR}/plm-reg-compose-vf.mha;${PLM_TESTING_BUILD_DIR}/plm-compose-a-vf.mha"
  )
plm_add_test (
  "plm-compose-a-warp" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--input;${PLM_TESTING_BUILD_DIR}/gauss-5.mha;--xf;${PLM_TESTING_BUILD_DIR}/plm-compose-a-vf.mha;--output-img;${PLM_TESTING_BUILD_DIR}/plm-compose-a-warp.mha"
  )
plm_add_test (
  "plm-compose-a-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-compose-a-vf.mha"
  )
plmtest_check_interval ("plm-compose-a-check"
  "${PLM_TESTING_BUILD_DIR}/plm-compose-a-stats.stdout.txt"
  "Mean: *([-0-9.]*)"
  "13.1"
  "13.3"
  )
set_tests_properties (plm-compose-a PROPERTIES 
  DEPENDS "plm-reg-itk-translation;plm-reg-compose")
set_tests_properties (plm-compose-a-warp PROPERTIES DEPENDS plm-compose-a)
set_tests_properties (plm-compose-a-stats PROPERTIES DEPENDS plm-compose-a)
set_tests_properties (plm-compose-a-check PROPERTIES 
  DEPENDS plm-compose-a-stats)

plm_add_test (
  "plm-compose-b" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "compose;${PLM_TESTING_BUILD_DIR}/plm-reg-itk-translation-vf.mha;${PLM_TESTING_BUILD_DIR}/plm-reg-compose-vf.mha;${PLM_TESTING_BUILD_DIR}/plm-compose-b-vf.mha"
  )
plm_add_test (
  "plm-compose-b-warp" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--input;${PLM_TESTING_BUILD_DIR}/gauss-5.mha;--xf;${PLM_TESTING_BUILD_DIR}/plm-compose-b-vf.mha;--output-img;${PLM_TESTING_BUILD_DIR}/plm-compose-b-warp.mha"
  )
plm_add_test (
  "plm-compose-b-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-compose-b-vf.mha"
  )
plmtest_check_interval ("plm-compose-b-check"
  "${PLM_TESTING_BUILD_DIR}/plm-compose-b-stats.stdout.txt"
  "Mean: *([-0-9.]*)"
  "13.1"
  "13.3"
  )
set_tests_properties (plm-compose-b PROPERTIES 
  DEPENDS "plm-reg-itk-translation;plm-reg-compose")
set_tests_properties (plm-compose-b-warp PROPERTIES DEPENDS plm-compose-b)
set_tests_properties (plm-compose-b-stats PROPERTIES DEPENDS plm-compose-b)
set_tests_properties (plm-compose-b-check PROPERTIES 
  DEPENDS plm-compose-b-stats)

plm_add_test (
  "plm-compose-c" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "compose;plm-bspline-single-h-xf.txt;${PLM_TESTING_BUILD_DIR}/plm-reg-compose-vf.mha;${PLM_TESTING_BUILD_DIR}/plm-compose-c-vf.mha"
  )
plm_add_test (
  "plm-compose-c-warp" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--input;${PLM_TESTING_BUILD_DIR}/gauss-5.mha;--xf;${PLM_TESTING_BUILD_DIR}/plm-compose-c-vf.mha;--output-img;${PLM_TESTING_BUILD_DIR}/plm-compose-c-warp.mha"
  )
plm_add_test (
  "plm-compose-c-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-compose-c-vf.mha"
  )
plmtest_check_interval ("plm-compose-c-check"
  "${PLM_TESTING_BUILD_DIR}/plm-compose-c-stats.stdout.txt"
  "Mean: *([-0-9.]*)"
  "12.5"
  "12.6"
  )
set_tests_properties (plm-compose-c PROPERTIES 
  DEPENDS "plm-reg-itk-translation;plm-reg-compose")
set_tests_properties (plm-compose-c-warp PROPERTIES DEPENDS plm-compose-c)
set_tests_properties (plm-compose-c-stats PROPERTIES DEPENDS plm-compose-c)
set_tests_properties (plm-compose-c-check PROPERTIES 
  DEPENDS plm-compose-c-stats)

## -------------------------------------------------------------------------
## plm-resample-a    Image, subsample
## plm-resample-b    Image, resample to fixed
## plm-resample-c    Vector field, resample to fixed
## -------------------------------------------------------------------------
plm_add_test (
  "plm-resample-a" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "resample;--input;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;--output;${PLM_TESTING_BUILD_DIR}/plm-resample-a.mha;--output-type;float;--subsample;2 2 2"
  )
plm_add_test (
  "plm-resample-a-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-resample-a.mha"
  )
plmtest_check_interval ("plm-resample-a-check"
  "${PLM_TESTING_BUILD_DIR}/plm-resample-a-stats.stdout.txt"
  "AVE *([-0-9.]*)"
  "-879.0"
  "-878.0"
  )
set_tests_properties (plm-resample-a PROPERTIES DEPENDS gauss-1)
set_tests_properties (plm-resample-a-stats PROPERTIES DEPENDS plm-resample-a)
set_tests_properties (plm-resample-a-check PROPERTIES 
  DEPENDS plm-resample-a-stats)

plm_add_test (
  "plm-resample-b" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "resample;--input;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;--output;${PLM_TESTING_BUILD_DIR}/plm-resample-b.mha;--output-type;float;--fixed;${PLM_TESTING_BUILD_DIR}/black-1.mha"
  )
plm_add_test (
  "plm-resample-b-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-resample-b.mha"
  )
plmtest_check_interval ("plm-resample-b-check"
  "${PLM_TESTING_BUILD_DIR}/plm-resample-b-stats.stdout.txt"
  "AVE *([-0-9.]*)"
  "-879.0"
  "-878.0"
  )
set_tests_properties (plm-resample-b PROPERTIES DEPENDS "gauss-1;black-1")
set_tests_properties (plm-resample-b-stats PROPERTIES DEPENDS plm-resample-b)
set_tests_properties (plm-resample-b-check PROPERTIES 
  DEPENDS plm-resample-b-stats)

plm_add_test (
  "plm-resample-c" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "resample;--input;${PLM_TESTING_BUILD_DIR}/bspline-mi-c-1-vf.mha;--output;${PLM_TESTING_BUILD_DIR}/plm-resample-c-vf.mha;--fixed;${PLM_TESTING_BUILD_DIR}/black-1.mha"
  )
plm_add_test (
  "plm-resample-c-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-resample-c-vf.mha"
  )
plmtest_check_interval ("plm-resample-c-check"
  "${PLM_TESTING_BUILD_DIR}/plm-resample-c-stats.stdout.txt"
  "Mean: *([-0-9.]*)"
  "6.73"
  "6.8"
  )
set_tests_properties (plm-resample-c PROPERTIES 
  DEPENDS "bspline-mi-c-1;black-1")
set_tests_properties (plm-resample-c-stats PROPERTIES DEPENDS plm-resample-c)
set_tests_properties (plm-resample-c-check PROPERTIES 
  DEPENDS plm-resample-c-stats)

## -------------------------------------------------------------------------
## plastimatch warp pointset
##   plm-warp-pointset-a:  xf = bspline, output fcsv
##   plm-warp-pointset-b:  xf = bspline, output txt
##   plm-warp-pointset-c:  xf = vf, output fcsv
## -------------------------------------------------------------------------
plm_add_test (
  "plm-warp-pointset-a" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--xf;${PLM_TESTING_BUILD_DIR}/plm-bsp-rect-xf.txt;--input;${PLM_TESTING_DATA_DIR}/fiducials-rect-2.fcsv;--output-pointset;${PLM_TESTING_BUILD_DIR}/plm-warp-pointset-a.fcsv"
  )
plm_add_test (
  "plm-warp-pointset-b" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--xf;${PLM_TESTING_BUILD_DIR}/plm-bsp-rect-xf.txt;--input;${PLM_TESTING_DATA_DIR}/fiducials-rect-2.fcsv;--output-pointset;${PLM_TESTING_BUILD_DIR}/plm-warp-pointset-b.txt"
  )
plm_add_test (
  "plm-warp-pointset-c" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--xf;${PLM_TESTING_BUILD_DIR}/landmark-warp-c-vf.mha;--input;${PLM_TESTING_DATA_DIR}/fiducials-rect-2.fcsv;--output-pointset;${PLM_TESTING_BUILD_DIR}/plm-warp-landmark-c.fcsv"
  )
set_tests_properties (plm-warp-pointset-a PROPERTIES DEPENDS plm-bsp-rect)
set_tests_properties (plm-warp-pointset-b PROPERTIES DEPENDS plm-bsp-rect)
set_tests_properties (plm-warp-pointset-c PROPERTIES DEPENDS landmark-warp-c)

## -------------------------------------------------------------------------
## plastimatch warp
##   plm warp a: xf = bspline, algorithm = itk
##   plm warp b: xf = bspline, algorithm = native
##   plm warp c: xf = bspline, algorithm = itk, interp = nn
##   plm warp d: xf = bspline, algorithm = native, interp = nn
##   plm warp e: input = xio, multiple outputs
##   plm warp f: input = dicom, multiple outputs
##   plm warp g: xf = translation
##   plm warp h: xf = translation, ushort images
##   plm warp i: xf = translation, double images
## -------------------------------------------------------------------------
plm_add_test (
  "plm-warp-a" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--algorithm;itk;--xf;${PLM_TESTING_BUILD_DIR}/bspline_c_xf.txt;--output-vf;${PLM_TESTING_BUILD_DIR}/plm-warp-a-vf.mha;--input;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;--output-img;${PLM_TESTING_BUILD_DIR}/plm-warp-a.mha"
  )
plm_add_test (
  "plm-warp-a-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-warp-a-vf.mha"
  )
plmtest_check_interval ("plm-warp-a-check"
  "${PLM_TESTING_BUILD_DIR}/plm-warp-a-stats.stdout.txt"
  "Mean: *([-0-9.]*)"
  "5.77"
  "5.78"
  )
set_tests_properties (plm-warp-a PROPERTIES DEPENDS "bspline-c;gauss-1")
set_tests_properties (plm-warp-a-stats PROPERTIES DEPENDS plm-warp-a)
set_tests_properties (plm-warp-a-check PROPERTIES DEPENDS plm-warp-a-stats)

plm_add_test (
  "plm-warp-b" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--xf;${PLM_TESTING_BUILD_DIR}/bspline_c_xf.txt;--output-vf;${PLM_TESTING_BUILD_DIR}/plm-warp-b-vf.mha;--input;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;--output-img;${PLM_TESTING_BUILD_DIR}/plm-warp-b.mha;--default-value;0"
  )
plm_add_test (
  "plm-warp-b-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-warp-b-vf.mha"
  )
plmtest_check_interval ("plm-warp-b-check-1"
  "${PLM_TESTING_BUILD_DIR}/plm-warp-b-stats-1.stdout.txt"
  "Mean: *([-0-9.]*)"
  "5.77"
  "5.78"
  )
plm_add_test (
  "plm-warp-b-stats-2"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-warp-b.mha"
  )
plmtest_check_interval ("plm-warp-b-check-2"
  "${PLM_TESTING_BUILD_DIR}/plm-warp-b-stats-2.stdout.txt"
  "AVE *([-0-9.]*)"
  "-775.7"
  "-775.6"
  )
set_tests_properties (plm-warp-b PROPERTIES DEPENDS "bspline-c;gauss-1")
set_tests_properties (plm-warp-b-stats-1 PROPERTIES DEPENDS plm-warp-b-1)
set_tests_properties (plm-warp-b-check-1 PROPERTIES DEPENDS plm-warp-b-stats-1)
set_tests_properties (plm-warp-b-stats-2 PROPERTIES DEPENDS plm-warp-b-2)
set_tests_properties (plm-warp-b-check-2 PROPERTIES DEPENDS plm-warp-b-stats-2)

plm_add_test (
  "plm-warp-c" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--algorithm;itk;--xf;${PLM_TESTING_BUILD_DIR}/bspline_c_xf.txt;--output-vf;${PLM_TESTING_BUILD_DIR}/plm-warp-c-vf.mha;--input;${PLM_TESTING_BUILD_DIR}/sphere-1.mha;--output-img;${PLM_TESTING_BUILD_DIR}/plm-warp-c.mha;--interpolation;nn;--default-value;0"
  )
plm_add_test (
  "plm-warp-c-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-warp-c.mha"
  )
plmtest_check_interval ("plm-warp-c-check"
  "${PLM_TESTING_BUILD_DIR}/plm-warp-c-stats.stdout.txt"
  "AVE *([-0-9.]*)"
  "9.2"
  "9.3"
  )
set_tests_properties (plm-warp-c PROPERTIES DEPENDS "bspline-c;sphere-1")
set_tests_properties (plm-warp-c-stats PROPERTIES DEPENDS plm-warp-c)
set_tests_properties (plm-warp-c-check PROPERTIES DEPENDS plm-warp-c-stats)

plm_add_test (
  "plm-warp-d" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--xf;${PLM_TESTING_BUILD_DIR}/bspline_c_xf.txt;--output-vf;${PLM_TESTING_BUILD_DIR}/plm-warp-d-vf.mha;--input;${PLM_TESTING_BUILD_DIR}/sphere-1.mha;--output-img;${PLM_TESTING_BUILD_DIR}/plm-warp-d.mha;--interpolation;nn;--default-value;0"
  )
plm_add_test (
  "plm-warp-d-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-warp-d.mha"
  )
plmtest_check_interval ("plm-warp-d-check"
  "${PLM_TESTING_BUILD_DIR}/plm-warp-d-stats.stdout.txt"
  "AVE *([-0-9.]*)"
  "9.2"
  "9.3"
  )
set_tests_properties (plm-warp-d PROPERTIES DEPENDS "bspline-d;sphere-1")
set_tests_properties (plm-warp-d-stats PROPERTIES DEPENDS plm-warp-d)
set_tests_properties (plm-warp-d-check PROPERTIES DEPENDS plm-warp-d-stats)

plm_add_test (
  "plm-warp-e" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--xf;${PLM_TESTING_DATA_DIR}/xf-bspline-chest-phantom.txt;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-xio-4.33.02;--output-xio;${PLM_TESTING_BUILD_DIR}/plm-warp-e-xio;--output-ss-img;${PLM_TESTING_BUILD_DIR}/plm-warp-e-ss-img.nrrd;--output-ss-list;${PLM_TESTING_BUILD_DIR}/plm-warp-e-ss-list.txt;--output-labelmap;${PLM_TESTING_BUILD_DIR}/plm-warp-e-labelmap.nrrd;--output-img;${PLM_TESTING_BUILD_DIR}/plm-warp-e-img.nrrd;--output-prefix;${PLM_TESTING_BUILD_DIR}/plm-warp-e;--output-cxt;${PLM_TESTING_BUILD_DIR}/plm-warp-e.cxt;--output-colormap;${PLM_TESTING_BUILD_DIR}/plm-warp-e.ctbl;--default-value;0"
  )
plm_add_test (
  "plm-warp-e-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-warp-e-img.nrrd"
  )
plmtest_check_interval ("plm-warp-e-check-1"
  "${PLM_TESTING_BUILD_DIR}/plm-warp-e-stats-1.stdout.txt"
  "AVE *([-0-9.]*)"
  "-661.5"
  "-661.4"
  )
plm_add_test (
  "plm-warp-e-stats-2"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-warp-e/Patient.mha"
  )
plmtest_check_interval ("plm-warp-e-check-2"
  "${PLM_TESTING_BUILD_DIR}/plm-warp-e-stats-2.stdout.txt"
  "AVE *([-0-9.]*)"
  "0.385"
  "0.395"
  )
plm_add_test (
  "plm-warp-e-stats-3"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-warp-e-labelmap.nrrd"
  )
plmtest_check_interval ("plm-warp-e-check-3"
  "${PLM_TESTING_BUILD_DIR}/plm-warp-e-stats-3.stdout.txt"
  "AVE *([-0-9.]*)"
  "0.67"
  "0.68"
  )
plm_add_test (
  "plm-warp-e-stats-4"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-warp-e-ss-img.nrrd"
  )
if (PLM_USE_SS_IMAGE_VEC)
  plmtest_check_interval ("plm-warp-e-check-4"
    "${PLM_TESTING_BUILD_DIR}/plm-warp-e-stats-4.stdout.txt"
    "S *1 *NVOX *([-0-9.]*)"
    "1971"
    "1971"
    )
else ()
  plmtest_check_interval ("plm-warp-e-check-4"
    "${PLM_TESTING_BUILD_DIR}/plm-warp-e-stats-4.stdout.txt"
    "AVE *([-0-9.]*)"
    "1.05"
    "1.15"
    )
endif ()
add_test ("plm-warp-e-check-5" ${CMAKE_COMMAND} -E compare_files
  "${PLM_TESTING_DATA_DIR}/plm-warp-e-ss-list.txt"
  "${PLM_TESTING_BUILD_DIR}/plm-warp-e-ss-list.txt")
add_test ("plm-warp-e-check-6" ${CMAKE_COMMAND} -E compare_files
  "${PLM_TESTING_DATA_DIR}/plm-warp-e.ctbl"
  "${PLM_TESTING_BUILD_DIR}/plm-warp-e.ctbl")
set_tests_properties (plm-warp-e-stats-1 PROPERTIES DEPENDS plm-warp-e)
set_tests_properties (plm-warp-e-check-1 PROPERTIES DEPENDS plm-warp-e-stats-1)
set_tests_properties (plm-warp-e-stats-2 PROPERTIES DEPENDS plm-warp-e)
set_tests_properties (plm-warp-e-check-2 PROPERTIES DEPENDS plm-warp-e-stats-2)
set_tests_properties (plm-warp-e-stats-3 PROPERTIES DEPENDS plm-warp-e)
set_tests_properties (plm-warp-e-check-3 PROPERTIES DEPENDS plm-warp-e-stats-3)
set_tests_properties (plm-warp-e-stats-4 PROPERTIES DEPENDS plm-warp-e)
set_tests_properties (plm-warp-e-check-4 PROPERTIES DEPENDS plm-warp-e-stats-4)
set_tests_properties (plm-warp-e-check-5 PROPERTIES DEPENDS plm-warp-e)
set_tests_properties (plm-warp-e-check-6 PROPERTIES DEPENDS plm-warp-e)

plm_add_test (
  "plm-warp-f" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--xf;${PLM_TESTING_DATA_DIR}/xf-bspline-chest-phantom.txt;--input;${PLM_TESTING_BUILD_DIR}/chest-phantom-dicomrt-xio-4.33.02;--output-dicom;${PLM_TESTING_BUILD_DIR}/plm-warp-f-dicom;--output-ss-img;${PLM_TESTING_BUILD_DIR}/plm-warp-f-ss-img.nrrd;--output-ss-list;${PLM_TESTING_BUILD_DIR}/plm-warp-f-ss-list.txt;--output-labelmap;${PLM_TESTING_BUILD_DIR}/plm-warp-f-labelmap.nrrd;--output-img;${PLM_TESTING_BUILD_DIR}/plm-warp-f-img.nrrd;--output-prefix;${PLM_TESTING_BUILD_DIR}/plm-warp-f;--output-cxt;${PLM_TESTING_BUILD_DIR}/plm-warp-f.cxt;--default-value;0"
  )
plm_add_test (
  "plm-warp-f-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-warp-f-img.nrrd"
  )
plmtest_check_interval ("plm-warp-f-check-1"
  "${PLM_TESTING_BUILD_DIR}/plm-warp-f-stats-1.stdout.txt"
  "AVE *([-0-9.]*)"
  "-661.5"
  "-661.4"
  )
if (PLM_USE_SS_IMAGE_VEC)
  plm_add_test (
    "plm-warp-f-stats-2"
    ""
    ${PLM_PLASTIMATCH_PATH}/plastimatch
    "stats;--input;plm-warp-f-ss-img.nrrd"
    )
  plmtest_check_interval ("plm-warp-f-check-2"
    "${PLM_TESTING_BUILD_DIR}/plm-warp-f-stats-2.stdout.txt"
    "S *1 *NVOX *([-0-9.]*)"
    "1971"
    "1971"
    )
set_tests_properties (plm-warp-f-stats-2 PROPERTIES DEPENDS plm-warp-f)
set_tests_properties (plm-warp-f-check-2 PROPERTIES DEPENDS plm-warp-f-stats-2)
else ()
  # Don't bother
endif ()
set_tests_properties (plm-warp-f-stats-1 PROPERTIES DEPENDS plm-warp-f)
set_tests_properties (plm-warp-f-check-1 PROPERTIES DEPENDS plm-warp-f-stats-1)

plm_add_test (
  "plm-warp-g" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--xf;${PLM_TESTING_DATA_DIR}/xf-translation-1.txt;--input;${PLM_TESTING_BUILD_DIR}/gauss-1.mha;--output-img;${PLM_TESTING_BUILD_DIR}/plm-warp-g-img.nrrd;--default-value;0"
  )
plm_add_test (
  "plm-warp-g-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-warp-g-img.nrrd"
  )
plmtest_check_interval ("plm-warp-g-check-1"
  "${PLM_TESTING_BUILD_DIR}/plm-warp-g-stats-1.stdout.txt"
  "AVE *([-0-9.]*)"
  "-675.5"
  "-674.5"
  )
set_tests_properties (plm-warp-g PROPERTIES DEPENDS gauss-1)
set_tests_properties (plm-warp-g-stats-1 PROPERTIES DEPENDS plm-warp-g)
set_tests_properties (plm-warp-g-check-1 PROPERTIES DEPENDS plm-warp-g-stats-1)

plm_add_test (
  "plm-warp-h" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--xf;${PLM_TESTING_DATA_DIR}/xf-translation-1.txt;--input;${PLM_TESTING_BUILD_DIR}/gauss-ushort-1.mha;--output-img;${PLM_TESTING_BUILD_DIR}/plm-warp-h-img.nrrd;--default-value;0"
  )
plm_add_test (
  "plm-warp-h-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-warp-h-img.nrrd"
  )
plmtest_check_interval ("plm-warp-h-check-1"
  "${PLM_TESTING_BUILD_DIR}/plm-warp-h-stats-1.stdout.txt"
  "AVE *([-0-9.]*)"
  "113.5"
  "114.0"
  )
set_tests_properties (plm-warp-h PROPERTIES DEPENDS gauss-ushort-1)
set_tests_properties (plm-warp-h-stats-1 PROPERTIES DEPENDS plm-warp-h)
set_tests_properties (plm-warp-h-check-1 PROPERTIES DEPENDS plm-warp-h-stats-1)

plm_add_test (
  "plm-warp-i" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--xf;${PLM_TESTING_DATA_DIR}/xf-translation-1.txt;--input;${PLM_TESTING_BUILD_DIR}/gauss-double-1.mha;--output-img;${PLM_TESTING_BUILD_DIR}/plm-warp-i-img.nrrd;--default-value;0"
  )
plm_add_test (
  "plm-warp-i-stats-1"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;plm-warp-i-img.nrrd"
  )
plmtest_check_interval ("plm-warp-i-check-1"
  "${PLM_TESTING_BUILD_DIR}/plm-warp-i-stats-1.stdout.txt"
  "AVE *([-0-9.]*)"
  "114.4"
  "114.7"
  )
set_tests_properties (plm-warp-i PROPERTIES DEPENDS gauss-double-1)
set_tests_properties (plm-warp-i-stats-1 PROPERTIES DEPENDS plm-warp-i)
set_tests_properties (plm-warp-i-check-1 PROPERTIES DEPENDS plm-warp-i-stats-1)

## -------------------------------------------------------------------------
## plm-xf-convert-a    plm bspline to vf
## plm-xf-convert-b    vf to plm bspline
## plm-xf-convert-c    plm bspline to vf
## plm-xf-convert-d    plm bspline to itk bspline
## -------------------------------------------------------------------------
plm_add_test (
  "plm-xf-convert-a" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "xf-convert;--input;${PLM_TESTING_DATA_DIR}/xf-bspline-1.txt;--output;${PLM_TESTING_BUILD_DIR}/plm-xf-convert-a-vf.mha;--output-type;vf;--dim;4 4 4;--origin;-187.5 -187.5 -187.5;--spacing;125 125 125;"
  )

plm_add_test (
  "plm-xf-convert-b"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "xf-convert;--input;${PLM_TESTING_BUILD_DIR}/plm-xf-convert-a-vf.mha;--output;${PLM_TESTING_BUILD_DIR}/plm-xf-convert-b.txt;--output-type;bspline;--dim;4 4 4;--origin;-187.5 -187.5 -187.5;--spacing;125 125 125;--grid-spacing;10 10 10"
  )
set_tests_properties (plm-xf-convert-b PROPERTIES DEPENDS plm-xf-convert-a)

plm_add_test (
  "plm-xf-convert-c"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "xf-convert;--input;${PLM_TESTING_BUILD_DIR}/plm-xf-convert-b.txt;--output;${PLM_TESTING_BUILD_DIR}/plm-xf-convert-c-vf.mha;--output-type;vf;--dim;4 4 4;--origin;-187.5 -187.5 -187.5;--spacing;125 125 125"
  )
set_tests_properties (plm-xf-convert-c PROPERTIES DEPENDS plm-xf-convert-b)

plm_add_test (
  "plm-xf-convert-d" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "xf-convert;--input;${PLM_TESTING_DATA_DIR}/plm-xf-convert.txt;--output;${PLM_TESTING_BUILD_DIR}/plm-xf-convert-d-xf.tfm;--output-type;itk_bspline"
  )
plm_add_test (
  "plm-xf-convert-d-warp-1" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--input;${PLM_TESTING_BUILD_DIR}/gauss-2.mha;--xf;${PLM_TESTING_DATA_DIR}/plm-xf-convert.txt;--output-img;${PLM_TESTING_BUILD_DIR}/plm-xf-convert-d-warp-1.nrrd"
  )
plm_add_test (
  "plm-xf-convert-d-warp-1-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-xf-convert-d-warp-1.nrrd"
  )
plmtest_check_interval (
  "plm-xf-convert-d-warp-1-check"
  "${PLM_TESTING_BUILD_DIR}/plm-xf-convert-d-warp-1-stats.stdout.txt"
  "AVE *([-0-9.]*)"
  "-745.8"
  "-744.8"
  )
plm_add_test (
  "plm-xf-convert-d-warp-2" 
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "warp;--input;${PLM_TESTING_BUILD_DIR}/gauss-2.mha;--xf;${PLM_TESTING_BUILD_DIR}/plm-xf-convert-d-xf.tfm;--output-img;${PLM_TESTING_BUILD_DIR}/plm-xf-convert-d-warp-2.nrrd"
  )
plm_add_test (
  "plm-xf-convert-d-warp-2-stats"
  ""
  ${PLM_PLASTIMATCH_PATH}/plastimatch
  "stats;--input;${PLM_TESTING_BUILD_DIR}/plm-xf-convert-d-warp-2.nrrd"
  )
plmtest_check_interval (
  "plm-xf-convert-d-warp-2-check"
  "${PLM_TESTING_BUILD_DIR}/plm-xf-convert-d-warp-2-stats.stdout.txt"
  "AVE *([-0-9.]*)"
  # GCS 2011-10-08.  The ITK B-spline resampling method is very poor, and 
  # gives different results on different platforms.  Probably we need 
  # to convert all ITK B-splines to native, and resample using the 
  # native method.
  "-751.0"
  #  "-745.8"
  "-744.8"
  )
set_tests_properties (plm-xf-convert-d-warp-1 PROPERTIES DEPENDS gauss-2)
set_tests_properties (plm-xf-convert-d-warp-1-stats PROPERTIES 
  DEPENDS plm-xf-convert-d-warp-1)
set_tests_properties (plm-xf-convert-d-warp-1-check PROPERTIES 
  DEPENDS plm-xf-convert-d-warp-1-stats)
set_tests_properties (plm-xf-convert-d-warp-2 PROPERTIES 
  DEPENDS "gauss-2;plm-xf-convert-d")
set_tests_properties (plm-xf-convert-d-warp-2-stats PROPERTIES 
  DEPENDS plm-xf-convert-d-warp-2)
set_tests_properties (plm-xf-convert-d-warp-2-check PROPERTIES 
  DEPENDS plm-xf-convert-d-warp-2-stats)

## -------------------------------------------------------------------------
## plm-api - Tests of plastimatch API
##   plm drr api a:
##   plm registration api a:
## -------------------------------------------------------------------------
plm_add_test (
  "plm-drr-api-a"
  ""
  ${PLM_PLASTIMATCH_PATH}/plm_drr_api_a
  ""
  )
plm_add_test (
  "plm-registration-api-a"
  ""
  ${PLM_PLASTIMATCH_PATH}/plm_registration_api_a
  ""
  )

## -------------------------------------------------------------------------
## bragg-curve
## proton-dose
## -------------------------------------------------------------------------
plm_add_test (
  "bragg-curve"
  ""
  ${PLM_PLASTIMATCH_PATH}/bragg_curve
  "-O;${PLM_TESTING_BUILD_DIR}/bragg-curve.txt"
  )
add_test (bragg-curve-check ${CMAKE_COMMAND} -E compare_files
  "${PLM_TESTING_DATA_DIR}/bragg-curve.txt"
  "${PLM_TESTING_BUILD_DIR}/bragg-curve.txt")
set_tests_properties (bragg-curve-check PROPERTIES DEPENDS bragg-curve)

plm_add_test (
  "proton-dose"
  ""
  ${PLM_PLASTIMATCH_PATH}/proton_dose
  "-src;1 0 0;-iso;0 0 0;-vup;0 0 1;-u;1;-p;${PLM_TESTING_DATA_DIR}/bragg-curve.txt;--debug;${PLM_TESTING_BUILD_DIR}/rect-2.mha;${PLM_TESTING_BUILD_DIR}/proton-dose.mha"
  )
set_tests_properties (proton-dose PROPERTIES DEPENDS rect-2)

## -------------------------------------------------------------------------
## slicer tests
## -------------------------------------------------------------------------
if (Slicer3_DIR)
#  slicer3_add_plugins_test(${CLP}Test1 ${CLP} --help)
endif (Slicer3_DIR)
