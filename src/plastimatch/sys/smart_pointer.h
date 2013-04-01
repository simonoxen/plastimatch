/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _smart_pointer_h_
#define _smart_pointer_h_

#include "dlib/smart_pointers.h"

#define plm_shared_ptr dlib::shared_ptr

#define SMART_POINTER_SUPPORT(T)                \
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
