@echo off
set BRT_RUNTIME=dx9

@rem ---------- Set directories ------------
@rem set DXSDK_DIR=C:\Program Files\Microsoft DirectX SDK (November 2007)
set DXSDK_DIR=C:\Program Files\Microsoft DirectX SDK (June 2007)

@rem set MSVC_DIR=C:\Program Files\Microsoft Visual Studio 8
set MSVC_DIR=C:\Program Files\Microsoft Visual Studio 9.0

@rem set PSDK_DIR=C:\Program Files\Microsoft Visual Studio 8\VC\PlatformSDK\Lib
set PSDK_DIR=C:\Program Files\Microsoft Platform SDK for Windows Server 2003 R2

@rem ---------- Set LIB ------------
@rem (The following may be needed for vs2005 professional)
@rem set LIB=C:\Program Files\Microsoft Visual Studio 8\SDK\v2.0\Lib

set LIB=%LIB%;%MSVC_DIR%\VC\lib
set LIB=%LIB%;%PSDK_DIR%\Lib
set LIB=%LIB%;%DXSDK_DIR%\Lib

@rem ---------- Set INCLUDE ------------
@rem (The following may be needed for vs2005 professional)
@rem set INCLUDE=C:\Program Files\Microsoft Visual Studio 8\SDK\v2.0\include

set INCLUDE=%INCLUDE%;%MSVC_DIR%\VC\include
set INCLUDE=%INCLUDE%;%PSDK_DIR%\Include
set INCLUDE=%INCLUDE%;%DXSDK_DIR%\Include

@rem ---------- Set PATH ------------
@rem PATH=%DXSDK_DIR%\Utilities;%PATH%
PATH=%DXSDK_DIR%\Utilities\Bin\x86;%PATH%
PATH=C:\Program Files\NVIDIA Corporation\Cg\bin;%PATH%
PATH=%MSVC_DIR%\Common7\IDE;%PATH%
PATH=%MSVC_DIR%\VC\bin;%PATH%
