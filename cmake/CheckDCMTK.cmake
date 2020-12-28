##---------------------------------------------------------------------------
## See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##---------------------------------------------------------------------------

include (CheckCXXSourceCompiles)

file (READ "${PLM_SOURCE_DIR}/cmake/check_dcmtk_ec_invalidvalue.cxx"
  CHECK_DCMTK_EC_INVALIDVALUE_SOURCE)
push_vars (CMAKE_REQUIRED_INCLUDES CMAKE_REQUIRED_LIBRARIES)
set (CMAKE_REQUIRED_INCLUDES ${DCMTK_INCLUDE_DIRS})
set (CMAKE_REQUIRED_LIBRARIES ${DCMTK_LIBRARIES})
check_cxx_source_compiles ("${CHECK_DCMTK_EC_INVALIDVALUE_SOURCE}"
  DCMTK_HAS_EC_INVALIDVALUE)
pop_vars (CMAKE_REQUIRED_INCLUDES CMAKE_REQUIRED_LIBRARIES)
