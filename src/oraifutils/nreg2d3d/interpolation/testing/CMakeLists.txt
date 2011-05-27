
# interpolator test
INCLUDE_DIRECTORIES(
  ${CMAKE_SOURCE_DIR}/${ORAIFUTILS_BASICTESTING_SUBDIR}
  ${CMAKE_SOURCE_DIR}/${ORAIFUTILS_NREG2D3D_INTERPOLATION_SUBDIR}
)
ADD_EXECUTABLE(TestOneToOneInterpolateImageFunction
  
  TestOneToOneInterpolateImageFunction.cxx
  
  ${CMAKE_SOURCE_DIR}/${ORAIFUTILS_NREG2D3D_INTERPOLATION_SUBDIR}/oraOneToOneInterpolateImageFunction.h
  ${CMAKE_SOURCE_DIR}/${ORAIFUTILS_NREG2D3D_INTERPOLATION_SUBDIR}/oraOneToOneInterpolateImageFunction.txx
)
TARGET_LINK_LIBRARIES(TestOneToOneInterpolateImageFunction
  ORAIFNReg2D3DInterpolation
)
ADD_TEST(TestOneToOneInterpolateImageFunction ${EXECUTABLE_OUTPUT_PATH}/TestOneToOneInterpolateImageFunction -v)
