@echo off

if "%COMPUTERNAME%"=="ROSHAR" (
  echo Setting for ROSHAR
  SET ITK_PATH=%HOME%\build\vs2005\itk-3.16.0\bin\Release
  SET PLASTIMATCH_BASE=%HOME%\build\vs2005\plastimatch-3.16.0
  SET CTTOOLS_PATH=%HOME%\projects\plastimatch\cttools
) else if "%COMPUTERNAME%"=="COTTONTAIL" (
  echo Setting for COTTONTAIL
  SET ITK_PATH=%HOME%\build\itk-3.8.0-vse2005\bin\Release
  SET PLASTIMATCH_BASE=%HOME%\build\plastimatch-vse2005
  SET CTTOOLS_PATH=%HOME%\projects\plastimatch\cttools
@rem  SET DXSDK_UTILS_DIR="C:\Program Files\Microsoft DirectX SDK (June 2007)\Utilities\Bin\x86"
@rem  PATH="%PATH%;%DXSDK_UTILS_DIR%"
@rem  PATH=%PATH%;"C:\Program Files\Microsoft DirectX SDK (June 2007)\Utilities\Bin\x86"
) else if "%COMPUTERNAME%"=="SLUMBER" (
  echo Setting for SLUMBER
  SET ITK_PATH=%HOME%\build\vs2008\itk-3.16.0\bin\Release
  SET PLASTIMATCH_BASE=%HOME%\build\vs2008\plastimatch-3.16.0
  SET CTTOOLS_PATH=%HOME%\projects\plastimatch\cttools
) else if "%COMPUTERNAME%"=="ROFOVIA" (
  echo Setting for ROFOVIA
  SET ITK_PATH=c:\plastimatch\build\itk-3.8.0\bin\Release
  SET PLASTIMATCH_BASE=c:\plastimatch\build\plastimatch
  SET CTTOOLS_PATH=c:\plastimatch\src\plastimatch\cttools
) else if "%COMPUTERNAME%"=="TORTOISE" (
  echo Setting for TORTOISE
  SET ITK_PATH=%HOME%\build\vs2008\itk-3.16.0\bin\Release
  SET PLASTIMATCH_BASE=%HOME%\build\vs2008\plastimatch-3.16.0
  SET CTTOOLS_PATH=%HOME%\projects\plastimatch\cttools
) else (
  echo "Sorry, couldn't recognize host"
  exit /b
)

SET PLASTIMATCH_PATH=%PLASTIMATCH_BASE%\Release

@rem if NOT %DXSDK_UTILS_DIR%=="" PATH="%PATH%;%DXSDK_UTILS_DIR%"
@rem if NOT "%DXSDK_UTILS_DIR%"=="" PATH=%PATH%;%DXSDK_UTILS_DIR%

PATH=%PATH%;%ITK_PATH%
PATH=%PATH%;%PLASTIMATCH_PATH%
PATH=%PATH%;%CTTOOLS_PATH%

echo PATH set to:
PATH
