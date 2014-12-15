/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _smart_pointer_h_
#define _smart_pointer_h_

#include "plm_config.h"

#if PLM_CUDA_COMPILE
/* There is a bug in Linux CUDA 4.x nvcc compiler which causes it to 
   barf with either dlib::shared_ptr or std::shared_ptr */
# define SMART_POINTER_SUPPORT(T)                         \
    typedef void* Pointer
#else
# if SHARED_PTR_USE_MEMORY
#  include <memory>
#  define plm_shared_ptr std::shared_ptr
# elif TR1_SHARED_PTR_USE_TR1_MEMORY
#  include <tr1/memory>
#  define plm_shared_ptr std::tr1::shared_ptr
# elif TR1_SHARED_PTR_USE_MEMORY
#  include <memory>
#  define plm_shared_ptr std::tr1::shared_ptr
# else
#  include "dlib/smart_pointers.h"
#  define plm_shared_ptr dlib::shared_ptr
# endif
# define SMART_POINTER_SUPPORT(T)                               \
    public:                                                     \
    typedef T Self;                                             \
    typedef plm_shared_ptr<Self> Pointer;                       \
    static T::Pointer New () {                                  \
        return T::Pointer (new T);                              \
    }                                                           \
    static T::Pointer New (T* t) {                              \
        return T::Pointer (t);                                  \
    }                                                           \
    template<class U1                                           \
             > static T::Pointer New (                          \
                 U1 u1) {                                       \
        return T::Pointer (                                     \
            new T(u1));                                         \
    }                                                           \
    template<class U1, class U2                                 \
             > static T::Pointer New (                          \
                 U1 u1, U2 u2) {                                \
        return T::Pointer (                                     \
            new T(u1, u2));                                     \
    }                                                           \
    template<class U1, class U2, class U3                       \
             > static T::Pointer New (                          \
                 U1 u1, U2 u2, U3 u3) {                         \
        return T::Pointer (                                     \
            new T(u1, u2, u3));                                 \
    }                                                           \
    template<class U1, class U2, class U3,                      \
             class U4, class U5, class U6                       \
             > static T::Pointer New (                          \
                 U1 u1, U2 u2, U3 u3,                           \
                 U4& u4, U5 u5, U6 u6) {                        \
        return T::Pointer (                                     \
            new T(u1, u2, u3, u4, u5, u6));                     \
    }

/* The above code seems to not work as well as I'd thought.
   (1) Template parameter assignment produces types, and therefore 
   arguments to New() are pass-by value
   (2) Therefore, classes without copy constructors require 
   explicit reference in signature (see U4& in six-parameter macro
   (3) Ideally, we can make everything const, such as:
    template<class U1                                           \
             > static T::Pointer New (                          \
                 const U1& u1) {                                \
        return T::Pointer (                                     \
            new T(u1));                                         \
    }                                                           \
    But this doesn't acutally work.  (Why not?)
*/

#endif

#endif
