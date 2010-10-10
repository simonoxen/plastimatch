/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include "opencl_util.h"
#include "opencl_util_nvidia.h"
#include "plm_timer.h"
#include "print_and_exit.h"

//////////////////////////////////////////////////////////////////////////////
//! Borrowed Utility Functions
//////////////////////////////////////////////////////////////////////////////

/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

//////////////////////////////////////////////////////////////////////////////
//! Gets the platform ID for NVIDIA if available, otherwise default
//!
//! @return the id 
//! @param clSelectedPlatformID         OpenCL platoform ID
//////////////////////////////////////////////////////////////////////////////
cl_int oclGetPlatformID(cl_platform_id* clSelectedPlatformID)
{
    char chBuffer[1024];
    cl_uint num_platforms; 
    cl_platform_id* clPlatformIDs;
    cl_int ciErrNum;
    *clSelectedPlatformID = NULL;

    // Get OpenCL platform count
    ciErrNum = clGetPlatformIDs (0, NULL, &num_platforms);
    if (ciErrNum != CL_SUCCESS)
    {
        shrLog(" Error %i in clGetPlatformIDs Call !!!\n\n", ciErrNum);
        return -1000;
    }
    else 
    {
        if(num_platforms == 0)
        {
            shrLog("No OpenCL platform found!\n\n");
            return -2000;
        }
        else 
        {
            // if there's a platform or more, make space for ID's
            if ((clPlatformIDs = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id))) == NULL)
            {
                shrLog("Failed to allocate memory for cl_platform ID's!\n\n");
                return -3000;
            }

            // get platform info for each platform and trap the NVIDIA platform if found
            ciErrNum = clGetPlatformIDs (num_platforms, clPlatformIDs, NULL);
            for(cl_uint i = 0; i < num_platforms; ++i)
            {
                ciErrNum = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
                if(ciErrNum == CL_SUCCESS)
                {
                    if(strstr(chBuffer, "NVIDIA") != NULL)
                    {
                        *clSelectedPlatformID = clPlatformIDs[i];
                        break;
                    }
                }
            }

            // default to zeroeth platform if NVIDIA not found
            if(*clSelectedPlatformID == NULL)
            {
                *clSelectedPlatformID = clPlatformIDs[0];
            }

            free(clPlatformIDs);
        }
    }

    return CL_SUCCESS;
}

//////////////////////////////////////////////////////////////////////////////
//! Gets the id of the first device from the context
//!
//! @return the id 
//! @param cxGPUContext         OpenCL context
//////////////////////////////////////////////////////////////////////////////
cl_device_id oclGetFirstDev(cl_context cxGPUContext)
{
    size_t szParmDataBytes;
    cl_device_id* cdDevices;

    // get the list of GPU devices associated with context
    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
    cdDevices = (cl_device_id*) malloc(szParmDataBytes);

    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);

    cl_device_id first = cdDevices[0];
    free(cdDevices);

    return first;
}

//////////////////////////////////////////////////////////////////////////////
//! Gets the id of device with maximal FLOPS from the context
//!
//! @return the id 
//! @param cxGPUContext         OpenCL context
//////////////////////////////////////////////////////////////////////////////
cl_device_id oclGetMaxFlopsDev(cl_context cxGPUContext)
{
    size_t szParmDataBytes;
    cl_device_id* cdDevices;

    // get the list of GPU devices associated with context
    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
    cdDevices = (cl_device_id*) malloc(szParmDataBytes);
    size_t device_count = szParmDataBytes / sizeof(cl_device_id);

    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);

    cl_device_id max_flops_device = cdDevices[0];
    int max_flops = 0;
    
    size_t current_device = 0;
    
    // CL_DEVICE_MAX_COMPUTE_UNITS
    cl_uint compute_units;
    clGetDeviceInfo(cdDevices[current_device], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);

    // CL_DEVICE_MAX_CLOCK_FREQUENCY
    cl_uint clock_frequency;
    clGetDeviceInfo(cdDevices[current_device], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
    
    max_flops = compute_units * clock_frequency;
    ++current_device;

    while( current_device < device_count )
    {
        // CL_DEVICE_MAX_COMPUTE_UNITS
        cl_uint compute_units;
        clGetDeviceInfo(cdDevices[current_device], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);

        // CL_DEVICE_MAX_CLOCK_FREQUENCY
        cl_uint clock_frequency;
        clGetDeviceInfo(cdDevices[current_device], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
        
        int flops = compute_units * clock_frequency;
        if( flops > max_flops )
        {
            max_flops        = flops;
            max_flops_device = cdDevices[current_device];
        }
        ++current_device;
    }

    free(cdDevices);

    return max_flops_device;
}

//////////////////////////////////////////////////////////////////////////////
//! Loads a Program file and prepends the cPreamble to the code.
//!
//! @return the source string if succeeded, 0 otherwise
//! @param cFilename        program filename
//! @param cPreamble        code that is prepended to the loaded file, typically a set of #defines or a header
//! @param szFinalLength    returned length of the code string
//////////////////////////////////////////////////////////////////////////////
char* oclLoadProgSource(const char* cFilename, const char* cPreamble, size_t* szFinalLength)
{
    // locals 
    FILE* pFileStream = NULL;
    size_t szSourceLength;

    // open the OpenCL source code file
    #ifdef _WIN32   // Windows version
        if(fopen_s(&pFileStream, cFilename, "rb") != 0) 
        {       
            return NULL;
        }
    #else           // Linux version
        pFileStream = fopen(cFilename, "rb");
        if(pFileStream == 0) 
        {       
            return NULL;
        }
    #endif

    size_t szPreambleLength = strlen(cPreamble);

    // get the length of the source code
    fseek(pFileStream, 0, SEEK_END); 
    szSourceLength = ftell(pFileStream);
    fseek(pFileStream, 0, SEEK_SET); 

    // allocate a buffer for the source code string and read it in
    char* cSourceString = (char *)malloc(szSourceLength + szPreambleLength + 1); 
    memcpy(cSourceString, cPreamble, szPreambleLength);
    if (fread((cSourceString) + szPreambleLength, szSourceLength, 1, pFileStream) != 1)
    {
        fclose(pFileStream);
        free(cSourceString);
        return 0;
    }

    // close the file and return the total length of the combined (preamble + source) string
    fclose(pFileStream);
    if(szFinalLength != 0)
    {
        *szFinalLength = szSourceLength + szPreambleLength;
    }
    cSourceString[szSourceLength + szPreambleLength] = '\0';

    return cSourceString;
}

//////////////////////////////////////////////////////////////////////////////
//! Gets the id of the nth device from the context
//!
//! @return the id or -1 when out of range
//! @param cxGPUContext         OpenCL context
//! @param device_idx            index of the device of interest
//////////////////////////////////////////////////////////////////////////////
cl_device_id oclGetDev(cl_context cxGPUContext, unsigned int nr)
{
    size_t szParmDataBytes;
    cl_device_id* cdDevices;

    // get the list of GPU devices associated with context
    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
    
    if( szParmDataBytes / sizeof(cl_device_id) < nr ) {
      return (cl_device_id)-1;
    }
    
    cdDevices = (cl_device_id*) malloc(szParmDataBytes);

    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);
    
    cl_device_id device = cdDevices[nr];
    free(cdDevices);

    return device;
}

//////////////////////////////////////////////////////////////////////////////
//! Get and log the binary (PTX) from the OpenCL compiler for the requested program & device
//!
//! @param cpProgram                   OpenCL program
//! @param cdDevice                    device of interest
//! @param const char*  cPtxFileName   optional PTX file name
//////////////////////////////////////////////////////////////////////////////
void oclLogPtx(cl_program cpProgram, cl_device_id cdDevice, const char* cPtxFileName)
{
    // Grab the number of devices associated with the program
    cl_uint num_devices;
    clGetProgramInfo(cpProgram, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &num_devices, NULL);

    // Grab the device ids
    cl_device_id* devices = (cl_device_id*) malloc(num_devices * sizeof(cl_device_id));
    clGetProgramInfo(cpProgram, CL_PROGRAM_DEVICES, num_devices * sizeof(cl_device_id), devices, 0);

    // Grab the sizes of the binaries
    size_t* binary_sizes = (size_t*)malloc(num_devices * sizeof(size_t));    
    clGetProgramInfo(cpProgram, CL_PROGRAM_BINARY_SIZES, num_devices * sizeof(size_t), binary_sizes, NULL);

    // Now get the binaries
    char** ptx_code = (char**)malloc(num_devices * sizeof(char*));
    for( unsigned int i=0; i<num_devices; ++i)
    {
        ptx_code[i] = (char*)malloc(binary_sizes[i]);
    }
    clGetProgramInfo(cpProgram, CL_PROGRAM_BINARIES, 0, ptx_code, NULL);

    // Find the index of the device of interest
    unsigned int idx = 0;
    while((idx < num_devices) && (devices[idx] != cdDevice)) 
    {
        ++idx;
    }
    
    // If the index is associated, log the result
    if(idx < num_devices)
    {
         
        // if a separate filename is supplied, dump ptx there 
        if (NULL != cPtxFileName)
        {
            shrLog("\nWriting ptx to separate file: %s ...\n\n", cPtxFileName);
            FILE* pFileStream = NULL;
            #ifdef _WIN32
                fopen_s(&pFileStream, cPtxFileName, "wb");
            #else
                pFileStream = fopen(cPtxFileName, "wb");
            #endif

            fwrite(ptx_code[idx], binary_sizes[idx], 1, pFileStream);
            fclose(pFileStream);        
        }
        else // log to logfile and console if no ptx file specified
        {
           shrLog("\n%s\nProgram Binary:\n%s\n%s\n", HDASHLINE, ptx_code[idx], HDASHLINE);
        }
    }

    // Cleanup
    free(devices);
    free(binary_sizes);
    for(unsigned int i = 0; i < num_devices; ++i)
    {
        free(ptx_code[i]);
    }
    free( ptx_code );
}

//////////////////////////////////////////////////////////////////////////////
//! Get and log the binary (PTX) from the OpenCL compiler for the requested program & device
//!
//! @param cpProgram    OpenCL program
//! @param cdDevice     device of interest
//////////////////////////////////////////////////////////////////////////////
void oclLogBuildInfo(cl_program cpProgram, cl_device_id cdDevice)
{
    // write out the build log and ptx, then exit
    char cBuildLog[10240];
    clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, 
                          sizeof(cBuildLog), cBuildLog, NULL );
    shrLog("\n%s\nBuild Log:\n%s\n%s\n", HDASHLINE, cBuildLog, HDASHLINE);
}

// Helper function for De-allocating cl objects
// *********************************************************************
void oclDeleteMemObjs(cl_mem* cmMemObjs, int iNumObjs)
{
    int i;
    for (i = 0; i < iNumObjs; i++)
    {
        if (cmMemObjs[i])clReleaseMemObject(cmMemObjs[i]);
    }
}  

// Helper function to get error string
// *********************************************************************
const char* oclErrorString(cl_int error)
{
    static const char* errorString[] = {
        "CL_SUCCESS",
        "CL_DEVICE_NOT_FOUND",
        "CL_DEVICE_NOT_AVAILABLE",
        "CL_COMPILER_NOT_AVAILABLE",
        "CL_MEM_OBJECT_ALLOCATION_FAILURE",
        "CL_OUT_OF_RESOURCES",
        "CL_OUT_OF_HOST_MEMORY",
        "CL_PROFILING_INFO_NOT_AVAILABLE",
        "CL_MEM_COPY_OVERLAP",
        "CL_IMAGE_FORMAT_MISMATCH",
        "CL_IMAGE_FORMAT_NOT_SUPPORTED",
        "CL_BUILD_PROGRAM_FAILURE",
        "CL_MAP_FAILURE",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "CL_INVALID_VALUE",
        "CL_INVALID_DEVICE_TYPE",
        "CL_INVALID_PLATFORM",
        "CL_INVALID_DEVICE",
        "CL_INVALID_CONTEXT",
        "CL_INVALID_QUEUE_PROPERTIES",
        "CL_INVALID_COMMAND_QUEUE",
        "CL_INVALID_HOST_PTR",
        "CL_INVALID_MEM_OBJECT",
        "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
        "CL_INVALID_IMAGE_SIZE",
        "CL_INVALID_SAMPLER",
        "CL_INVALID_BINARY",
        "CL_INVALID_BUILD_OPTIONS",
        "CL_INVALID_PROGRAM",
        "CL_INVALID_PROGRAM_EXECUTABLE",
        "CL_INVALID_KERNEL_NAME",
        "CL_INVALID_KERNEL_DEFINITION",
        "CL_INVALID_KERNEL",
        "CL_INVALID_ARG_INDEX",
        "CL_INVALID_ARG_VALUE",
        "CL_INVALID_ARG_SIZE",
        "CL_INVALID_KERNEL_ARGS",
        "CL_INVALID_WORK_DIMENSION",
        "CL_INVALID_WORK_GROUP_SIZE",
        "CL_INVALID_WORK_ITEM_SIZE",
        "CL_INVALID_GLOBAL_OFFSET",
        "CL_INVALID_EVENT_WAIT_LIST",
        "CL_INVALID_EVENT",
        "CL_INVALID_OPERATION",
        "CL_INVALID_GL_OBJECT",
        "CL_INVALID_BUFFER_SIZE",
        "CL_INVALID_MIP_LEVEL",
        "CL_INVALID_GLOBAL_WORK_SIZE",
    };

    const int errorCount = sizeof(errorString) / sizeof(errorString[0]);

    const int index = -error;

    return (index >= 0 && index < errorCount) ? errorString[index] : "";
}

// Deallocate memory allocated within shrUtils
// *********************************************************************
void shrFree(void* ptr) 
{
  if( NULL != ptr) free( ptr);
}

// Optional LogFileName Override function
// *********************************************************************
char* cLogFilePathAndName = NULL;
void shrSetLogFileName (const char* cOverRideName)
{
    if( cLogFilePathAndName != NULL ) {
        free(cLogFilePathAndName);
    }
    cLogFilePathAndName = (char*) malloc(strlen(cOverRideName) + 1);
    #ifdef WIN32
        strcpy_s(cLogFilePathAndName, strlen(cOverRideName) + 1, cOverRideName);
    #else
        strcpy(cLogFilePathAndName, cOverRideName);
    #endif
    return;
}

// Function to log standardized information to console, file or both
// *********************************************************************
static int shrLogV(int iLogMode, int iErrNum, const char* cFormatString, va_list vaArgList)
{
    static FILE* pFileStream0 = NULL;
    static FILE* pFileStream1 = NULL;
    size_t szNumWritten = 0;
    char cFileMode [3];
    const char*     pStr; 
    const char*     cArg;
    int             iArg;
    double          dArg;
    unsigned int    uiArg;

    // if the sample log file is closed and the call incudes a "write-to-file", open file for writing
    if ((pFileStream0 == NULL) && (iLogMode & LOGFILE))
    {
        // if the default filename has not been overriden, set to default
        if (cLogFilePathAndName == NULL)
        {
            shrSetLogFileName(DEFAULTLOGFILE); 
        }

        #ifdef _WIN32   // Windows version
            // set the file mode
            if (iLogMode & APPENDMODE)  // append to prexisting file contents
            {
                sprintf_s (cFileMode, 3, "a+");  
            }
            else                        // replace prexisting file contents
            {
                sprintf_s (cFileMode, 3, "w"); 
            }

            // open the individual sample log file in the requested mode
            errno_t err = fopen_s(&pFileStream0, cLogFilePathAndName, cFileMode);
            
            // if error on attempt to open, be sure the file is null or close it, then return negative error code            
            if (err != 0)
            {
                if (pFileStream0)
                {
                    fclose (pFileStream0);
                }
                return -err;
            }
        #else           // Linux & Mac version
            // set the file mode
            if (iLogMode & APPENDMODE)  // append to prexisting file contents
            {
                sprintf (cFileMode, "a+");  
            }
            else                        // replace prexisting file contents
            {
                sprintf (cFileMode, "w"); 
            }

            // open the file in the requested mode
            if ((pFileStream0 = fopen(cLogFilePathAndName, cFileMode)) == 0)
            {
                // if error on attempt to open, be sure the file is null or close it, then return negative error code
                if (pFileStream0)
                {
                    fclose (pFileStream0);
                }
                return -1;
            }
        #endif
    }
    
    // if the master log file is closed and the call incudes a "write-to-file" and MASTER, open master logfile file for writing
    if ((pFileStream1 == NULL) && (iLogMode & LOGFILE) && (iLogMode & MASTER))
    {

        #ifdef _WIN32   // Windows version
            // open the master log file in append mode
            errno_t err = fopen_s(&pFileStream1, MASTERLOGFILE, "a+");

            // if error on attempt to open, be sure the file is null or close it, then return negative error code
            if (err != 0)
            {
                if (pFileStream1)
                {
                    fclose (pFileStream1);
                }
                return -err;
            }
        #else           // Linux & Mac version

            // open the file in the requested mode
            if ((pFileStream1 = fopen(MASTERLOGFILE, "a+")) == 0)
            {
                // if error on attempt to open, be sure the file is null or close it, then return negative error code
                if (pFileStream1)
                {
                    fclose (pFileStream1);
                }
                return -1;
            }
        #endif
        
        // If master log file length has become excessive, empty/reopen
        fseek(pFileStream1, 0L, SEEK_END);            
        if (ftell(pFileStream1) > 50000L)
        {
            fclose (pFileStream1);
        #ifdef _WIN32   // Windows version
            fopen_s(&pFileStream1, MASTERLOGFILE, "w");
	    #else
	        pFileStream1 = fopen(MASTERLOGFILE, "w");
	    #endif
        }
    }

    // Handle special Error Message code
    if (iLogMode & ERRORMSG)  
    {   
        // print string to console if flagged
        if (iLogMode & LOGCONSOLE) 
        {
            szNumWritten = printf ("\n !!! Error # %i at ", iErrNum);
        }
        // print string to file if flagged
        if (iLogMode & LOGFILE) 
        {
            szNumWritten = fprintf (pFileStream0, "\n !!! Error # %i at ", iErrNum);
        }
    }

    // Start at the head of the string and scan to the null at the end
    for (pStr = cFormatString; *pStr; ++pStr)
    {
        // Check if the current character is not a formatting specifier ('%') 
        if (*pStr != '%')
        {
            // not "%", so print verbatim to console and/or files as flagged
            if (iLogMode & LOGCONSOLE) // to console if flagged
            {
                szNumWritten = putc(*pStr, stdout);
            }
            if (iLogMode & LOGFILE)    
            {
                szNumWritten  = putc(*pStr, pFileStream0);      // sample log file
                if (iLogMode & MASTER)                          
                {
                    szNumWritten  = putc(*pStr, pFileStream1);  // master log file
                }
            }
        } 
        else 
        {
            // character was %, so handle the next arg according to next character
            switch (*++pStr)
            {
                case 's':   // string with default precision (max length for string)
                {
                    // Set cArg as the next value in list
                    cArg = va_arg(vaArgList, char*);

                    // print string to console and/or files if flagged
                    if (iLogMode & LOGCONSOLE) 
                    {
                        szNumWritten = printf("%s", cArg);
                    }
                    if (iLogMode & LOGFILE)
                    {
                        szNumWritten = fprintf (pFileStream0, "%s", cArg);
                        if (iLogMode & MASTER)                          
                        {
                            szNumWritten  = fprintf(pFileStream1, "%s", cArg);  // master log file
                        }
                    }
                    continue;
                }
                case 'd':   // integer with default precision
                case 'i':   // integer with default precision
                {
                    // set iArg as the next value in list
                    iArg = va_arg(vaArgList, int);

                    // print string to console and/or file if flagged
                    if (iLogMode & LOGCONSOLE) 
                    {
                        szNumWritten = printf("%i", iArg);
                    }
                    if (iLogMode & LOGFILE)
                    {
                        szNumWritten = fprintf (pFileStream0, "%i", iArg);
                        if (iLogMode & MASTER)                          
                        {
                            szNumWritten  = fprintf(pFileStream1, "%i", iArg);  // master log file
                        }
                    }
                    continue;
                }
                case 'u':   // unsigned integer with default precision
                {
                    // set uiArg as the next value in list
                    uiArg = va_arg(vaArgList, unsigned int);

                    // print string to console and/or file if flagged
                    if (iLogMode & LOGCONSOLE) 
                    {
                        szNumWritten = printf("%u", uiArg);
                    }
                    if (iLogMode & LOGFILE)
                    {
                        szNumWritten = fprintf (pFileStream0, "%u", uiArg);
                        if (iLogMode & MASTER)                          
                        {
                            szNumWritten  = fprintf(pFileStream1, "%u", uiArg);  // master log file
                        }
                    }
                    continue;
                }
                case 'e':   // scientific double/float with default precision
                case 'E':   // scientific double/float with default precision
                {
                    // set dArg as the next value in list
                    dArg = va_arg(vaArgList, double);

                    // print string to console and/or file if flagged
                    if (iLogMode & LOGCONSOLE) 
                    {
                        szNumWritten = printf("%e", dArg);
                    }
                    if (iLogMode & LOGFILE)
                    {
                        szNumWritten = fprintf (pFileStream0, "%e", dArg);
                        if (iLogMode & MASTER)                          
                        {
                            szNumWritten  = fprintf(pFileStream1, "%e", dArg);  // master log file
                        }
                    }
                    continue;
                }
                case 'f':  // float/double with default precision
                {
                    // set dArg as the next value in list
                    dArg = va_arg(vaArgList, double);

                    // print string to console and/or file if flagged
                    if (iLogMode & LOGCONSOLE) 
                    {
                        szNumWritten = printf("%f", dArg);  
                    }
                    if (iLogMode & LOGFILE)
                    {
                        szNumWritten = fprintf (pFileStream0, "%f", dArg);
                        if (iLogMode & MASTER)                          
                        {
                            szNumWritten  = fprintf(pFileStream1, "%f", dArg);  // master log file
                        }
                    }

                    continue;
                }
                case '.':  // Type and Precision are specified (f, e, E, i, d, u, s)
                {
                    // format with specified type and precision 
                    char cFormat [5] = "%.3f";
                    cFormat[2] = *++pStr;   // jump past the decimal pt and get the precision width
                    cFormat[3] = *++pStr;   // jump past the precision val and get type field (f, e, E, i, d, u, s)   

                    switch(cFormat[3])
                    {
                        case 'f':
                        case 'e':
                        case 'E':
                        {
                            // set dArg as the next value in list
                            dArg = va_arg(vaArgList, double);

                            // print string to console if flagged
                            if (iLogMode & LOGCONSOLE) 
                            {
                                szNumWritten = printf(cFormat, dArg);
                            }

                            // print string to file if flagged
                            if (iLogMode & LOGFILE)
                            {
                                szNumWritten = fprintf (pFileStream0, cFormat, dArg);
                                if (iLogMode & MASTER)                          
                                {
                                    szNumWritten  = fprintf(pFileStream1, cFormat, dArg);  // master log file
                                }
                            }
                            break;
                        }
                        case 'd':
                        case 'i':
                        {
                            // set iArg as the next value in list
                            iArg = va_arg(vaArgList, int);

                            // print string to console if flagged
                            if (iLogMode & LOGCONSOLE) 
                            {
                                szNumWritten = printf(cFormat, iArg);
                            }

                            // print string to file if flagged
                            if (iLogMode & LOGFILE)
                            {
                                szNumWritten = fprintf (pFileStream0, cFormat, iArg);
                                if (iLogMode & MASTER)                          
                                {
                                    szNumWritten  = fprintf(pFileStream1, cFormat, iArg);  // master log file
                                }
                            }
                            break;
                        }
                        case 'u':
                        {
                            // set uiArg as the next value in list
                            uiArg = va_arg(vaArgList, unsigned int);

                            // print string to console if flagged
                            if (iLogMode & LOGCONSOLE) 
                            {
                                szNumWritten = printf(cFormat, uiArg);
                            }

                            // print string to file if flagged
                            if (iLogMode & LOGFILE)
                            {
                                szNumWritten = fprintf (pFileStream0, cFormat, uiArg);
                                if (iLogMode & MASTER)                          
                                {
                                    szNumWritten  = fprintf(pFileStream1, cFormat, uiArg);  // master log file
                                }
                            }
                            break;
                        }
                        case 's':
                        {
                            // Set cArg as the next value in list
                            cArg = va_arg(vaArgList, char*);

                            // print string to console if flagged
                            if (iLogMode & LOGCONSOLE) 
                            {
                                szNumWritten = printf(cFormat, cArg);
                            }

                            // print string to file if flagged
                            if (iLogMode & LOGFILE)
                            {
                                szNumWritten = fprintf (pFileStream0, cFormat, cArg);
                                if (iLogMode & MASTER)                          
                                {
                                    szNumWritten  = fprintf(pFileStream1, cFormat, cArg);  // master log file
                                }
                            }
                            break;
                        }
                    }
                    continue;
                }
                default: 
                {
                    if (iLogMode & LOGCONSOLE) // to console if flagged
                    {
                        szNumWritten = putc(*pStr, stdout);
                    }
                    if (iLogMode & LOGFILE)    
                    {
                        szNumWritten  = putc(*pStr, pFileStream0);      // sample log file
                        if (iLogMode & MASTER)                          
                        {
                            szNumWritten  = putc(*pStr, pFileStream1);  // master log file
                        }
                    }
                }
            }
        }
    }

    // end the sample log(s) with a horizontal line if closing
    if (iLogMode & CLOSELOG) 
    {
        if (iLogMode & LOGCONSOLE) 
        {
            printf(HDASHLINE);
        }
        if (iLogMode & LOGFILE)
        {
            fprintf(pFileStream0, HDASHLINE);
        }
    }

    // flush console and/or file buffers if updated
    if (iLogMode & LOGCONSOLE) 
    {
        fflush(stdout);
    }
    if (iLogMode & LOGFILE)
    {
        fflush (pFileStream0);

        // if the master log file has been updated, flush it too
        if (iLogMode & MASTER)
        {
            fflush (pFileStream1);
        }
    }

    // If the log file is open and the caller requests "close file", then close and NULL file handle
    if ((pFileStream0) && (iLogMode & CLOSELOG))
    {
        fclose (pFileStream0);
        pFileStream0 = NULL;
    }
    if ((pFileStream1) && (iLogMode & CLOSELOG))
    {
        fclose (pFileStream1);
        pFileStream1 = NULL;
    }

    // return error code or OK 
    if (iLogMode & ERRORMSG)
    {
        return iErrNum;
    }
    else 
    {
        return 0;
    }
}

// Function to log standardized information to console, file or both
// *********************************************************************
int shrLogEx(int iLogMode = LOGCONSOLE, int iErrNum = 0, const char* cFormatString = "", ...)
{
    va_list vaArgList;

    // Prepare variable agument list 
    va_start(vaArgList, cFormatString);
    int ret = shrLogV(iLogMode, iErrNum, cFormatString, vaArgList);

    // end variable argument handler
    va_end(vaArgList);

    return ret;
}

// Function to log standardized information to console, file or both
// *********************************************************************
int shrLog(const char* cFormatString = "", ...)
{
    va_list vaArgList;

    // Prepare variable agument list 
    va_start(vaArgList, cFormatString);
    int ret = shrLogV(LOGBOTH, 0, cFormatString, vaArgList);

    // end variable argument handler
    va_end(vaArgList);

    return ret;
}

//////////////////////////////////////////////////////////////////////////////
//! Find the path for a file assuming that
//! files are found in the searchPath.
//!
//! @return the path if succeeded, otherwise 0
//! @param filename         name of the file
//! @param executable_path  optional absolute path of the executable
//////////////////////////////////////////////////////////////////////////////
char* shrFindFilePath(const char* filename, const char* executable_path) 
{
    // <executable_name> defines a variable that is replaced with the name of the executable

    // Typical relative search paths to locate needed companion files (e.g. sample input data, or JIT source files)
    // The origin for the relative search may be the .exe file, a .bat file launching an .exe, a browser .exe launching the .exe or .bat, etc
    const char* searchPath[] = {
        "./",                                       // same dir 
        "./data/",                                  // "/data/" subdir 
        "./src/",                                   // "/src/" subdir
        "./inc/",                                   // "/inc/" subdir
        "../",                                      // up 1 in tree 
        "../data/",                                 // up 1 in tree, "/data/" subdir 
        "../src/",                                  // up 1 in tree, "/src/" subdir 
        "../inc/",                                  // up 1 in tree, "/inc/" subdir 
        "../OpenCL/src/<executable_name>/",         // up 1 in tree, "/OpenCL/src/<executable_name>/" subdir 
        "../OpenCL/src/<executable_name>/data/",    // up 1 in tree, "/OpenCL/src/<executable_name>/data/" subdir 
        "../OpenCL/src/<executable_name>/src/",     // up 1 in tree, "/OpenCL/src/<executable_name>/src/" subdir 
        "../OpenCL/src/<executable_name>/inc/",     // up 1 in tree, "/OpenCL/src/<executable_name>/inc/" subdir 
        "../C/src/<executable_name>/",              // up 1 in tree, "/C/src/<executable_name>/" subdir 
        "../C/src/<executable_name>/data/",         // up 1 in tree, "/C/src/<executable_name>/data/" subdir 
        "../C/src/<executable_name>/src/",          // up 1 in tree, "/C/src/<executable_name>/src/" subdir 
        "../C/src/<executable_name>/inc/",          // up 1 in tree, "/C/src/<executable_name>/inc/" subdir 
        "../DirectCompute/src/<executable_name>/",      // up 1 in tree, "/DirectCompute/src/<executable_name>/" subdir 
        "../DirectCompute/src/<executable_name>/data/", // up 1 in tree, "/DirectCompute/src/<executable_name>/data/" subdir 
        "../DirectCompute/src/<executable_name>/src/",  // up 1 in tree, "/DirectCompute/src/<executable_name>/src/" subdir 
        "../DirectCompute/src/<executable_name>/inc/",  // up 1 in tree, "/DirectCompute/src/<executable_name>/inc/" subdir 
        "../../",                                   // up 2 in tree 
        "../../data/",                              // up 2 in tree, "/data/" subdir 
        "../../src/",                               // up 2 in tree, "/src/" subdir 
        "../../inc/",                               // up 2 in tree, "/inc/" subdir 
        "../../../",                                // up 3 in tree 
        "../../../src/<executable_name>/",          // up 3 in tree, "/src/<executable_name>/" subdir 
        "../../../src/<executable_name>/data/",     // up 3 in tree, "/src/<executable_name>/data/" subdir 
        "../../../src/<executable_name>/src/",      // up 3 in tree, "/src/<executable_name>/src/" subdir 
        "../../../src/<executable_name>/inc/",      // up 3 in tree, "/src/<executable_name>/inc/" subdir 
        "../../../sandbox/<executable_name>/",      // up 3 in tree, "/sandbox/<executable_name>/" subdir
        "../../../sandbox/<executable_name>/data/", // up 3 in tree, "/sandbox/<executable_name>/data/" subdir
        "../../../sandbox/<executable_name>/src/",  // up 3 in tree, "/sandbox/<executable_name>/src/" subdir
        "../../../sandbox/<executable_name>/inc/"   // up 3 in tree, "/sandbox/<executable_name>/inc/" subdir
    };
    
    // Extract the executable name
    std::string executable_name;
    if (executable_path != 0) 
    {
        executable_name = std::string(executable_path);

    #ifdef _WIN32        
        // Windows path delimiter
        size_t delimiter_pos = executable_name.find_last_of('\\');        
        executable_name.erase(0,delimiter_pos+1);

		if (executable_name.rfind(".exe") != string::npos) {
			// we strip .exe, only if the .exe is found
			executable_name.resize(executable_name.size() - 4);        
		}
    #else
        // Linux & OSX path delimiter
        size_t delimiter_pos = executable_name.find_last_of('/');        
        executable_name.erase(0,delimiter_pos+1);
    #endif
        
    }
    
    // Loop over all search paths and return the first hit
    for( unsigned int i=0; i<sizeof(searchPath)/sizeof(char*); ++i )
    {
        std::string path(searchPath[i]);        
        size_t executable_name_pos = path.find("<executable_name>");

        // If there is executable_name variable in the searchPath 
        // replace it with the value
        if( executable_name_pos != std::string::npos )
        {
            if( executable_path != 0 ) {
                
                path.replace(executable_name_pos, strlen("<executable_name>"), executable_name);

            } 
            else 
            {
                // Skip this path entry if no executable argument is given
                continue;
            }
        }
        
        // Test if the file exists
        path.append(filename);
        std::fstream fh(path.c_str(), std::fstream::in);
        if (fh.good())
        {
            // File found
            // returning an allocated array here for backwards compatibility reasons
            char* file_path = (char*) malloc(path.length() + 1);
        #ifdef _WIN32  
            strcpy_s(file_path, path.length() + 1, path.c_str());
        #else
            strcpy(file_path, path.c_str());
        #endif                
            return file_path;
        }
    }    

    // File not found
    return 0;
}

// Round Up Division function
size_t shrRoundUp(int group_size, int global_size) 
{
    int r = global_size % group_size;
    if(r == 0) 
    {
        return global_size;
    } else 
    {
        return global_size + group_size - r;
    }
}

