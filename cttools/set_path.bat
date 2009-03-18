@echo off

if "%COMPUTERNAME%"=="ROSHAR" (
  echo Setting for ROSHAR
@rem  SET ITK_PATH=%HOME%\build\itk-3.8-vs2005\bin\Release
@rem  SET PLASTIMATCH_PATH=%HOME%\build\plastimatch-itk-3.8-vs2005\Release
  SET ITK_PATH=%HOME%\build\itk-3.10.2\bin\Release
  SET PLASTIMATCH_PATH=%HOME%\build\plastimatch-3.10.2\Release
  SET CTTOOLS_PATH=%HOME%\projects\plastimatch\cttools
) else if "%COMPUTERNAME%"=="COTTONTAIL" (
  echo Setting for COTTONTAIL
  SET ITK_PATH=%HOME%\build\itk-3.8.0-vse2005\bin\Release
  SET PLASTIMATCH_PATH=%HOME%\build\plastimatch-vse2005\Release
  SET CTTOOLS_PATH=%HOME%\projects\plastimatch\cttools
@rem  SET DXSDK_UTILS_DIR="C:\Program Files\Microsoft DirectX SDK (June 2007)\Utilities\Bin\x86"
@rem  PATH="%PATH%;%DXSDK_UTILS_DIR%"
@rem  PATH=%PATH%;"C:\Program Files\Microsoft DirectX SDK (June 2007)\Utilities\Bin\x86"
) else if "%COMPUTERNAME%"=="SLUMBER" (
  echo Setting for SLUMBER
@rem  SET ITK_PATH=%HOME%\build\itk-3.8.0\bin\Release
@rem  SET PLASTIMATCH_PATH=%HOME%\build\plastimatch-3.8.0\Release
  SET ITK_PATH=%HOME%\build\itk-3.10.2\bin\Release
  SET PLASTIMATCH_PATH=%HOME%\build\plastimatch-3.10.2\Release
  SET CTTOOLS_PATH=%PLASTIMATCH_PATH%\cttools
) else if "%COMPUTERNAME%"=="ROFOVIA" (
  echo Setting for ROFOVIA
  SET ITK_PATH=c:\plastimatch\build\itk-3.8.0\bin\Release
  SET PLASTIMATCH_PATH=c:\plastimatch\build\plastimatch\Release
  SET CTTOOLS_PATH=c:\plastimatch\src\plastimatch\cttools
) else (
  echo "Sorry, couldn't recognize host"
  exit /b
)


@rem if NOT %DXSDK_UTILS_DIR%=="" PATH="%PATH%;%DXSDK_UTILS_DIR%"
@rem if NOT "%DXSDK_UTILS_DIR%"=="" PATH=%PATH%;%DXSDK_UTILS_DIR%

PATH=%PATH%;%ITK_PATH%
PATH=%PATH%;%PLASTIMATCH_PATH%
PATH=%PATH%;%CTTOOLS_PATH%

echo PATH set to:
PATH
