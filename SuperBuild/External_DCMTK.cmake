##-----------------------------------------------------------------------------
##  Download DCMTK from internet and compile
##-----------------------------------------------------------------------------
set (proj DCMTK)

set (dcmtk_url ftp://dicom.offis.de/pub/dicom/offis/software/dcmtk/dcmtk362/dcmtk-3.6.2.tar.gz)
set (dcmtk_md5sum d219a4152772985191c9b89d75302d12)

ExternalProject_Add (${proj}
  DOWNLOAD_DIR ${proj}-download
  URL ${dcmtk_url}
  URL_MD5 ${dcmtk_md5sum}
  SOURCE_DIR ${proj}
  BINARY_DIR ${proj}-build
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
  -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
  -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
  -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
  -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
  -DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}
  #    -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD}
  -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=${CMAKE_CXX_STANDARD_REQUIRED}
  -DCMAKE_CXX_EXTENSIONS:BOOL=${CMAKE_CXX_EXTENSIONS}
  -DBUILD_APPS:BOOL=OFF
  -DBUILD_SHARED_LIBS:BOOL=ON
  -DDCMTK_OVERWRITE_WIN32_COMPILER_FLAGS:BOOL=OFF
  INSTALL_COMMAND ""
  )

set (DCMTK_DIR ${CMAKE_BINARY_DIR}/${proj}-build)
