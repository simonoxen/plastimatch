/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _string_util_h_
#define _string_util_h_

#include "plmsys_config.h"
#include <string>
#include <vector>

PLMSYS_API int plm_strcmp (const char* s1, const char* s2);
PLMSYS_API void string_util_rtrim_whitespace (char *s);
PLMSYS_API int parse_int13 (int *arr, const char *string);
PLMSYS_API int parse_dicom_float2 (float *arr, const char *string);
PLMSYS_API int parse_dicom_float3 (float *arr, const char *string);
PLMSYS_API int parse_dicom_float6 (float *arr, const char *string);
PLMSYS_API std::vector<int> parse_int3_string (const char* s);
PLMSYS_API std::vector<float> parse_float3_string (const char* s);
PLMSYS_API const std::string trim (
    const std::string& str,
    const std::string& whitespace = " \t\r\n"
);
PLMSYS_API std::string slurp_file (const char* fn);
PLMSYS_API std::string string_format (const std::string &fmt, ...);

#endif
