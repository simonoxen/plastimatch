#ifndef __plm_config_h__
#define __plm_config_h__

#cmakedefine HAVE_GETOPT_LONG 1
#cmakedefine BUILD_SHARED_LIBS 1
#cmakedefine HAVE_F2C_LIBRARY 1
#cmakedefine HAVE_BROOK_LIBRARY 1
#cmakedefine BUILD_BSPLINE_BROOK 1

#if _MSC_VER >= 1400
/* 4996 warnings are generated when using C library functions */
/* 4819 warnings are generated by a bug in the itk 3.4 headers */
#pragma warning( disable : 4996 4244 4819 )
#endif

#if _MSC_VER
#define inline __inline
#endif

#if defined(__BORLANDC__) || defined(_MSC_VER)
#define snprintf _snprintf
#endif

#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 1)
#pragma GCC diagnostic ignored "-Wwrite-strings"
#endif

/* This code is for exporting symbols when building DLLs on windows */
#if (defined(_WIN32) || defined(WIN32)) && defined (BUILD_SHARED_LIBS)
# ifdef plastimatch1_EXPORTS
#  define plastimatch1_EXPORT __declspec(dllexport)
# else
#  define plastimatch1_EXPORT __declspec(dllimport)
# endif
#else
/* unix needs nothing */
#define plastimatch1_EXPORT 
#endif

#endif /* __plm_config_h__ */
