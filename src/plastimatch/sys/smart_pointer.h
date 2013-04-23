/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _smart_pointer_h_
#define _smart_pointer_h_

#include "plm_config.h"

#if (PLM_CUDA_COMPILE)
/* There is a bug in Linux CUDA 4.x nvcc compiler which causes it to 
   barf with either dlib::shared_ptr or std::shared_ptr */
# define SMART_POINTER_SUPPORT(T)               \
    typedef void* Pointer
#else
# include "dlib/smart_pointers.h"
# define plm_shared_ptr dlib::shared_ptr
# define SMART_POINTER_SUPPORT(T)               \
    public:                                     \
    typedef T Self;                             \
    typedef plm_shared_ptr<Self> Pointer;       \
    static T::Pointer New () {                  \
        return T::Pointer (new T);              \
    }                                           \
    static T::Pointer New (T* t) {              \
        return T::Pointer (t);                  \
    }
#endif

#endif
