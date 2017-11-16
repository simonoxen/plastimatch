cmake \
    -DQt5_DIR:PATH=/usr/lib/x86_64-linux-gnu/cmake/Qt5/ \
    -DSlicer_VTK_VERSION_MAJOR=8 \
    -DCMAKE_CXX_STANDARD=11 \
    -DSlicer_USE_SimpleITK:BOOL=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING:BOOL=OFF \
    ../Slicer

#    -DSlicer_USE_SYSTEM_OpenSSL:BOOL=ON \
#    -DSlicer_USE_SYSTEM_python \
#    -DSlicer_USE_SYSTEM_QT
