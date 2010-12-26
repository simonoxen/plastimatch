/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _opencl_utils_nvidia_h_
#define _opencl_utils_nvidia_h_

#include "plm_config.h"
#if (OPENCL_FOUND)
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

//////////////////////////////////////////////////////////////////////////////
//! Borrowed Utility Functions
//////////////////////////////////////////////////////////////////////////////

// OS dependent includes
#ifdef _WIN32
    // Headers needed for Windows
    #include <windows.h>
#else
    // Headers needed for Linux
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <sys/time.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <stdarg.h>
#endif

// Other headers needed for both Windows and Linux
#include <math.h>
#include <assert.h>
#include <string>

using std::string;

// Defines and enum for use with logging functions
// *********************************************************************
#define DEFAULTLOGFILE "SdkConsoleLog.txt"
#define MASTERLOGFILE "SdkMasterLog.csv"
enum LOGMODES 
    {
	LOGCONSOLE = 1, // bit to signal "log to console" 
	LOGFILE =    2, // bit to signal "log to file" 
	LOGBOTH =    3, // convenience union of first 2 bits to signal "log to both"
	APPENDMODE = 4, // bit to set "file append" mode instead of "replace mode" on open
	MASTER =     8, // bit to signal master .csv log output
	ERRORMSG =  16, // bit to signal "pre-pend Error" 
	CLOSELOG =  32  // bit to close log file, if open, after any requested file write
    };
#define HDASHLINE "-----------------------------------------------------------\n"

// Standardized boolean
enum shrBOOL
{
    shrFALSE = 0,
    shrTRUE = 1
};

// Standardized MAX, MIN and CLAMP
#define MAX(a, b) ((a > b) ? a : b)
#define MIN(a, b) ((a < b) ? a : b)
#define CLAMP(a, b, c) MIN(MAX(a, b), c)    // double sided clip of input a
#define TOPCLAMP(a, b) (a < b ? a:b)	    // single top side clip of input a

// Error and Exit Handling Macros... 
// *********************************************************************
// Full error handling macro with Cleanup() callback (if supplied)... 
// (Companion Inline Function lower on page)
#define shrCheckErrorEX(a, b, c) __shrCheckErrorEX(a, b, c, __FILE__ , __LINE__) 

// Short version without Cleanup() callback pointer
// Both Input (a) and Reference (b) are specified as args
#define shrCheckError(a, b) shrCheckErrorEX(a, b, 0) 

// Simple argument checker macro
#define ARGCHECK(a) if((a) != shrTRUE)return shrFALSE 

// Define for user-customized error handling
#define STDERROR "file %s, line %i\n\n" , __FILE__ , __LINE__


// Function to deallocate memory allocated within shrUtils
// *********************************************************************
extern "C" void shrFree(void* ptr);

// *********************************************************************
// Helper function to log standardized information to Console, to File or to both
//! Examples: shrLogEx(LOGBOTH, 0, "Function A\n"); 
//!         : shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
//! 
//! Automatically opens file and stores handle if needed and not done yet
//! Closes file and nulls handle on request
//! 
//! @param 0 iLogMode: LOGCONSOLE, LOGFILE, LOGBOTH, APPENDMODE, MASTER, ERRORMSG, CLOSELOG.  
//!          LOGFILE and LOGBOTH may be | 'd  with APPENDMODE to select file append mode instead of overwrite mode 
//!          LOGFILE and LOGBOTH may be | 'd  with CLOSELOG to "write and close" 
//!          First 3 options may be | 'd  with MASTER to enable independent write to master data log file
//!          First 3 options may be | 'd  with ERRORMSG to start line with standard error message
//! @param 2 dValue:    
//!          Positive val = double value for time in secs to be formatted to 6 decimals. 
//!          Negative val is an error code and this give error preformatting.
//! @param 3 cFormatString: String with formatting specifiers like printf or fprintf.  Supported specifiers are:  
//!             %i, %d, %u, %f, %e, %E, %s,  
//!             %.<dig>i, %.<dig>d, %.<dig>u, %.<dig>f, %.<dig>e, %.<dig>E, , %.<dig>s    (where <dig> is 0-9)
//! @param 4... variable args: like printf or fprintf.  Must match format specifer type above.  
//! @return 0 if OK, negative value on error or if error occurs or was passed in. 
// *********************************************************************
extern "C" gpuit_EXPORT int shrLogEx(int iLogMode, int iErrNum, const char* cFormatString, ...);

// Short version of shrLogEx defaulting to shrLogEx(LOGBOTH, 0, 
// *********************************************************************
extern "C" int gpuit_EXPORT shrLog(const char* cFormatString, ...);

// Optional LogFileNameOverride function
// *********************************************************************
extern "C" gpuit_EXPORT void shrSetLogFileName (const char* cOverRideName);

////////////////////////////////////////////////////////////////////////////
//! Find the path for a filename
//! @return the path if succeeded, otherwise 0
//! @param filename        name of the file
//! @param executablePath  optional absolute path of the executable
////////////////////////////////////////////////////////////////////////////
extern "C" gpuit_EXPORT char* shrFindFilePath(const char* filename, const char* executablePath);

extern "C" gpuit_EXPORT size_t shrRoundUp(int group_size, int global_size);

// companion inline function for error checking and exit on error WITH Cleanup Callback (if supplied)
// *********************************************************************
inline void __shrCheckErrorEX(int iSample, int iReference, void (*pCleanup)(int), const char* cFile, const int iLine)
{
    if (iReference != iSample)
    {
        shrLogEx(LOGBOTH | ERRORMSG, iSample, "line %i , in file %s !!!\n\n" , iLine, cFile); 
        if (pCleanup != NULL)
        {
            pCleanup(EXIT_FAILURE);
        }
        else 
        {
            shrLogEx(LOGBOTH | CLOSELOG, 0, "Exiting...\n");
            exit(EXIT_FAILURE);
        }
    }
}

// Error and Exit Handling Macros... 
// *********************************************************************
// Full error handling macro with Cleanup() callback (if supplied)... 
// (Companion Inline Function lower on page)
#define oclCheckErrorEX(a, b, c) __oclCheckErrorEX(a, b, c, __FILE__ , __LINE__) 

// Short version without Cleanup() callback pointer
// Both Input (a) and Reference (b) are specified as args
#define oclCheckError(a, b) oclCheckErrorEX(a, b, 0) 

//////////////////////////////////////////////////////////////////////////////
//! Gets the platform ID for NVIDIA if available, otherwise default to platform 0
//!
//! @return the id 
//! @param clSelectedPlatformID         OpenCL platform ID
//////////////////////////////////////////////////////////////////////////////
extern "C" gpuit_EXPORT cl_int oclGetPlatformID(cl_platform_id* clSelectedPlatformID);

//////////////////////////////////////////////////////////////////////////////
//! Gets the id of the first device from the context
//!
//! @return the id 
//! @param cxGPUContext         OpenCL context
//////////////////////////////////////////////////////////////////////////////
extern "C" gpuit_EXPORT cl_device_id oclGetFirstDev(cl_context cxGPUContext);

//////////////////////////////////////////////////////////////////////////////
//! Gets the id of the nth device from the context
//!
//! @return the id or -1 when out of range
//! @param cxGPUContext         OpenCL context
//! @param device_idx            index of the device of interest
//////////////////////////////////////////////////////////////////////////////
extern "C" gpuit_EXPORT cl_device_id oclGetDev(cl_context cxGPUContext, unsigned int device_idx);

//////////////////////////////////////////////////////////////////////////////
//! Gets the id of device with maximal FLOPS from the context
//!
//! @return the id 
//! @param cxGPUContext         OpenCL context
//////////////////////////////////////////////////////////////////////////////
extern "C" cl_device_id oclGetMaxFlopsDev(cl_context cxGPUContext);

//////////////////////////////////////////////////////////////////////////////
//! Loads a Program file and prepends the cPreamble to the code.
//!
//! @return the source string if succeeded, 0 otherwise
//! @param cFilename        program filename
//! @param cPreamble        code that is prepended to the loaded file, typically a set of #defines or a header
//! @param szFinalLength    returned length of the code string
//////////////////////////////////////////////////////////////////////////////
extern "C" gpuit_EXPORT char* oclLoadProgSource(const char* cFilename, const char* cPreamble, size_t* szFinalLength);

//////////////////////////////////////////////////////////////////////////////
//! Get and log the binary (PTX) from the OpenCL compiler for the requested program & device
//!
//! @param cpProgram                   OpenCL program
//! @param cdDevice                    device of interest
//! @param const char*  cPtxFileName   optional PTX file name
//////////////////////////////////////////////////////////////////////////////
extern "C" void gpuit_EXPORT oclLogPtx(cl_program cpProgram, cl_device_id cdDevice, const char* cPtxFileName);

//////////////////////////////////////////////////////////////////////////////
//! Get and log the Build Log from the OpenCL compiler for the requested program & device
//!
//! @param cpProgram    OpenCL program
//! @param cdDevice     device of interest
//////////////////////////////////////////////////////////////////////////////
extern "C" void gpuit_EXPORT oclLogBuildInfo(cl_program cpProgram, cl_device_id cdDevice);

// Helper function for De-allocating cl objects
// *********************************************************************
extern "C" void oclDeleteMemObjs(cl_mem* cmMemObjs, int iNumObjs);

// Helper function to get error string
// *********************************************************************
extern "C" gpuit_EXPORT const char* oclErrorString(cl_int error);

// companion inline function for error checking and exit on error WITH Cleanup Callback (if supplied)
// *********************************************************************
inline void __oclCheckErrorEX(cl_int iSample, cl_int iReference, void (*pCleanup)(int), const char* cFile, const int iLine)
{
    if (iReference != iSample)
    {
        shrLog("\n !!! Error # %i (%s) at line %i , in file %s !!!\n\n", iSample, oclErrorString(iSample), iLine, cFile); 
        if (pCleanup != NULL)
        {
            pCleanup(EXIT_FAILURE);
        }
        else 
        {
            shrLogEx(LOGBOTH | CLOSELOG, 0, "Exiting...\n");
            exit(EXIT_FAILURE);
        }
    }
}

#endif /* HAVE_OPENCL */
#endif
