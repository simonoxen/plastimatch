# Added by: James Shackleford
#     Date: Feb 28th, 2012
#
# Based on CMakeLists.txt from http://trac.assembla.com/lua
#  adapted for Plastimatch's need of only a static link lib
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(lua)

INCLUDE(CMakeDependentOption)
SET(CPACK_SET_DESTDIR TRUE)
INCLUDE(CPack)

SET(LUA_MAJOR_VERSION 5)
SET(LUA_MINOR_VERSION 1)
SET(LUA_PATCH_VERSION 4)

SET(LUA_VERSION ${LUA_MAJOR_VERSION}.${LUA_MINOR_VERSION}.${LUA_PATCH_VERSION})
SET(LUA_COMPATIBILITY_VERSION ${LUA_MAJOR_VERSION}.${LUA_MINOR_VERSION}.0)
SET(LUA_SOVERSION ${LUA_MAJOR_VERSION}.${LUA_MINOR_VERSION}.0)

SET(WANTS_BUILD_STATIC_LIBRARY ON)
SET(WANTS_BUILD_FRAMEWORK OFF)
SET(LUA_LIBRARY_NAME lua)

SET(LUA_CORE
    ./src/lapi.c
    ./src/lcode.c
    ./src/ldebug.c
    ./src/ldo.c
    ./src/ldump.c
    ./src/lfunc.c
    ./src/lgc.c
    ./src/llex.c
    ./src/lmem.c
    ./src/lobject.c
    ./src/lopcodes.c
    ./src/lparser.c
    ./src/lstate.c
    ./src/lstring.c
    ./src/ltable.c 
    ./src/ltm.c 
    ./src/lundump.c
    ./src/lvm.c
    ./src/lzio.c
)

SET(LUA_STANDARD_LIBRARY
    ./src/lauxlib.c
    ./src/lbaselib.c
    ./src/ldblib.c
    ./src/liolib.c
    ./src/lmathlib.c
    ./src/loslib.c
    ./src/ltablib.c
    ./src/lstrlib.c
    ./src/loadlib.c
    ./src/linit.c
)

SET(PUBLIC_HEADERS
    ./src/lua.h 
    ./src/lualib.h
    ./src/lauxlib.h
    ./src/luaconf.h
    ./etc/lua.hpp
)
SET(RESOURCE_FILES
    ./etc/strict.lua
    ./etc/lua.ico
)

IF(APPLE)
    # Enable dyld for loading external libraries at runtime.
    #SET(LUA_C_FLAGS "${LUA_C_FLAGS} -DLUA_USE_MACOSX=1")
    # If you don't need pre-10.3 compatibility, DLOPEN is recommended instead.
    SET(LUA_C_FLAGS "${LUA_C_FLAGS} -DLUA_USE_POSIX -DLUA_USE_DLOPEN")
ENDIF(APPLE)

IF(CMAKE_SYSTEM MATCHES "Linux.*")
    # enable dlopen() and mkstemp() support, etc.
    SET(LUA_C_FLAGS "${LUA_C_FLAGS} -DLUA_USE_LINUX")
    # Lua uses libm's math functions and libdl's dlopen.
    SET(LUA_LINKER_FLAGS "m;dl;${LUA_LINKER_FLAGS}")
ENDIF(CMAKE_SYSTEM MATCHES "Linux.*")


ADD_LIBRARY(lua_library_static STATIC ${PUBLIC_HEADERS} ${LUA_STANDARD_LIBRARY} ${LUA_CORE} ${RESOURCE_FILES})

SET_TARGET_PROPERTIES(lua_library_static PROPERTIES OUTPUT_NAME "lua")
SET_TARGET_PROPERTIES(lua_library_static PROPERTIES COMPILE_FLAGS "${LUA_C_FLAGS}")


# Setup properties for static library
IF(APPLE)
    SET_TARGET_PROPERTIES(lua_library_static PROPERTIES
        VERSION ${LUA_VERSION}
        SOVERSION ${LUA_COMPATIBILITY_VERSION}
    )
ELSEIF(UNIX)
    SET_TARGET_PROPERTIES(lua_library_static PROPERTIES
        VERSION ${LUA_VERSION}
        SOVERSION ${LUA_SOVERSION}
    )
ELSE(APPLE)
    SET_TARGET_PROPERTIES(lua_library_static PROPERTIES
        VERSION ${LUA_VERSION}
        SOVERSION ${LUA_COMPATIBILITY_VERSION}
    )
ENDIF(APPLE)


# Set defaults for Universal Binaries. We want 32-bit Intel/PPC on 10.4 
# and 32/64-bit Intel/PPC on >= 10.5. Anything <= 10.3 doesn't support.
IF(APPLE)
    # These are just defaults/recommendations, but how we want to build
    # out of the box. But the user needs to be able to change these options.
    # So we must only set the values the first time CMake is run, or we 
    # will overwrite any changes the user sets.
    # FORCE is used because the options are not reflected in the UI otherwise.
    # Seems like a good place to add version specific compiler flags too.
    IF(NOT LUA_CONFIG_HAS_BEEN_RUN_BEFORE)
        # This is really fragile, but CMake doesn't provide the OS system 
        # version information we need. (Darwin versions can be changed 
        # independently of OS X versions.)
        # It does look like CMake handles the CMAKE_OSX_SYSROOT automatically.
        IF(EXISTS /Developer/SDKs/MacOSX10.5.sdk)
            SET(CMAKE_OSX_ARCHITECTURES "ppc;i386;ppc64;x86_64" CACHE STRING "Build architectures for OSX" FORCE)
            SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mmacosx-version-min=10.5" CACHE STRING "Flags used by the compiler during all build types." FORCE)
        ELSE(EXISTS /Developer/SDKs/MacOSX10.5.sdk)
            IF(EXISTS /Developer/SDKs/MacOSX10.4u.sdk)
                SET(CMAKE_OSX_ARCHITECTURES "ppc;i386" CACHE STRING "Build architectures for OSX" FORCE)
                SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mmacosx-version-min=10.4" CACHE STRING "Flags used by the compiler during all build types." FORCE)
            ELSE(EXISTS /Developer/SDKs/MacOSX10.4u.sdk)
                # No Universal Binary support
                # Should break down further to set the -mmacosx-version-min,
                # but the SDK detection is too unreliable here.
            ENDIF(EXISTS /Developer/SDKs/MacOSX10.4u.sdk)
        ENDIF(EXISTS /Developer/SDKs/MacOSX10.5.sdk)
    ENDIF(NOT LUA_CONFIG_HAS_BEEN_RUN_BEFORE)
ENDIF(APPLE)

# This needs to be run very last so other parts of the scripts can take
# advantage of this.
IF(NOT LUA_CONFIG_HAS_BEEN_RUN_BEFORE)
    SET(LUA_CONFIG_HAS_BEEN_RUN_BEFORE 1 CACHE INTERNAL "Flag to track whether this is the first time running CMake or if CMake has been configured before")
ENDIF(NOT LUA_CONFIG_HAS_BEEN_RUN_BEFORE)

