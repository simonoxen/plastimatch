PROJECT(CUDA_tex_test)

IF (PLM_BUILD_CUDA_TEST AND CUDA_FOUND)
  CUDA_COMPILE (CUDA_TEX_TEST_WRAPPERS 
    tex_kernels.cu
    tex_stubs.cu)
  SET (CUDA_TEX_TEST_SRC
    ${CUDA_TEX_TEST_WRAPPERS}
    tex_kernels.h
    tex_stubs.h
    textures.c
    )
  PLM_ADD_EXECUTABLE (cuda_tex_test "${CUDA_TEX_TEST_SRC}" 
    "${CUDA_LIBRARIES}" "" 0)
ENDIF ()
