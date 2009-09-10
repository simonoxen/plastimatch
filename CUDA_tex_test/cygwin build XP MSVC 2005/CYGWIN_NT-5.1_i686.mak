################################################################################
# EXTENSION DEFINITION #########################################################
################################################################################

# Define extensions for this OS
X_OBJEXT=.obj
X_LIBEXT=.lib
X_EXEEXT=.exe



################################################################################
# CFLAGS / LDFLAGS #############################################################
################################################################################

# MSVC Compiler Flags
CFLAGS = /EHsc /W3 /nologo /MT

# MSVC Compiler Flags for CUDA Files
CFLAGS_CUDA = $(CFLAGS)

# NVCC Flags
NVCCFLAGS = --ptxas-options=-v #--device-emulation

# Linker flags for NVCC
LDFLAGS = /INCREMENTAL:NO /NOLOGO /MACHINE:X86 /LTCG /OPT:REF 

################################################################################
# SOFTWARE PATHS ###############################################################
################################################################################

# Location of Visual Studio 8.0
PATH_VC= /cygdrive/c/Program Files/Microsoft Visual Studio 8
PATH_VC_DOS= C:\Program Files\Microsoft Visual Studio 8

# Location of CUDA
PATH_CUDA= /cygdrive/c/CUDA
PATH_CUDA_DOS= C:\CUDA

# Location of CUDA SDK
PATH_CUDA_SDK= /cygdrive/c/Program Files/NVIDIA Corporation/NVIDIA CUDA SDK
PATH_CUDA_SDK_DOS= C:\Program Files\NVIDIA Corporation\NVIDIA CUDA SDK



################################################################################
# EXPORTS ######################################################################
################################################################################
export VSINSTALLDIR=/cygdrive/c/Program Files/Microsoft Visual Studio 8
export VCINSTALLDIR=/cygdrive/c/Program Files/Microsoft Visual Studio 8/VC
export FrameworkDir=v2.0.50727
export FrameworkSDKDir=/cygdrice/c/Program Files/Microsoft Visual Studio 8/SDK/v2.0

export INCLUDE=C:\Program Files\Microsoft Visual Studio 8\VC\ATLMFC\INCLUDE;C:\Program Files\Microsoft Visual Studio 8\VC\INCLUDE;C:\Program Files\Microsoft Visual Studio 8\VC\PlatformSDK\include;C:\Program Files\Microsoft Visual Studio 8\SDK\v2.0\include;%INCLUDE%
export LIB=C:\Program Files\Microsoft Visual Studio 8\VC\ATLMFC\LIB;C:\Program Files\Microsoft Visual Studio 8\VC\LIB;C:\Program Files\Microsoft Visual Studio 8\VC\PlatformSDK\lib;C:\Program Files\Microsoft Visual Studio 8\SDK\v2.0\lib;%LIB%
export LIBPATH=C:\WINDOWS\Microsoft.NET\Framework\v2.0.50727;C:\Program Files\Microsoft Visual Studio 8\VC\ATLMFC\LIB



################################################################################
# DERIVED PATHS ################################################################
################################################################################
VC_BIN=$(PATH_VC)/VC/bin
VC_LIB=$(PATH_VC)/VC/lib
VC_LIB_DOS=$(PATH_VC_DOS)\VC\lib
VC_INC=$(PATH_VC)/VC/include
VC_COMMON=$(PATH_VC)/Common7
VC_TOOLS=$(VC_COMMON)/Tools
CUDA_BIN=$(PATH_CUDA)/bin
CUDA_COMMON=$(PATH_CUDA_SDK)/common
CUDA_COMMON_DOS=$(PATH_CUDA_SDK_DOS)\common
#CUDA_INC=$(CUDA_COMMON_DOS)\inc
CUDA_INC=$(PATH_CUDA_DOS)\include

#STD_LIB="$(VC_LIB)"

STD_LIB_DOS=$(VC_LIB_DOS)
STD_LIB_CUDA_1_DOS=$(PATH_CUDA_DOS)\lib
STD_LIB_CUDA_2_DOS=$(CUDA_COMMON_DOS)\lib
STD_LIBS_CUDA=cudart.lib cutil32.lib
STD_LIBS=

STD_LIB_CUDA_1=$(PATH_CUDA)/lib
STD_LIB_CUDA_2=$(CUDA_COMMON)/lib

################################################################################
# BINARIES #####################################################################
################################################################################
CL = $(VC_BIN)/cl.exe
LINK = $(VC_BIN)/link.exe
NVCC = $(CUDA_BIN)/nvcc.exe
SET_ENV = $(VC_BIN)/vcvars32.bat


################################################################################
# PATH #########################################################################
################################################################################
PATH := ${PATH}:$(PATH_CUDA)/open64/bin
PATH := ${PATH}:$(VC_COMMON)/IDE:$(VC_BIN):$(VC_TOOLS):$(VC_TOOLS)/bin:$(PATH_VC)/VC/PlatformSDK/bin
PATH := ${PATH}:$(PATH_VC)/SDK/v2.0/bin:/cygwin/c/WINDOWS/Microsoft.NET/Framework/v2.0.50727:$(PATH_VC)/VC/VCPackages

