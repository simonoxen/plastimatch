/* -------------------------------------------------------------------------*
    See COPYRIGHT for copyright information.
 * -------------------------------------------------------------------------*/
#ifndef __config_h__
#define __config_h__

#cmakedefine HAVE_MIL 1

#cmakedefine HAVE_BITFLOW 1

/* Requred for QueueUserAPC() */
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0400
#endif

#if _MSC_VER >= 1400
#pragma warning( disable : 4996 4244 )
#endif

#if _MSC_VER
#define inline __inline
#endif

#if defined(__BORLANDC__) || defined(_MSC_VER)
#define snprintf _snprintf
#endif

#endif /* __config_h__ */
