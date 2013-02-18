/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _volume_macros_h_
#define _volume_macros_h_

#include "plmbase_config.h"
#include "plm_int.h"

/* -----------------------------------------------------------------------
   Macros
   ----------------------------------------------------------------------- */
static inline plm_long 
volume_index (const plm_long dims[3], plm_long i, plm_long j, plm_long k)
{
    return i + (dims[0] * (j + dims[1] * k));
}

static inline plm_long 
volume_index (const plm_long dims[3], const plm_long ijk[3])
{
    return ijk[0] + (dims[0] * (ijk[1] + dims[1] * ijk[2]));
}

#define COORDS_FROM_INDEX(ijk, idx, dim)                                \
    ijk[2] = idx / (dim[0] * dim[1]);                                   \
    ijk[1] = (idx - (ijk[2] * dim[0] * dim[1])) / dim[0];               \
    ijk[0] = idx - ijk[2] * dim[0] * dim[1] - (ijk[1] * dim[0]);

#define LOOP_Z(ijk,fxyz,vol)                                            \
    for (                                                               \
    ijk[2] = 0,                                                         \
        fxyz[2] = vol->offset[2];                                       \
    ijk[2] < vol->dim[2];                                               \
    ++ijk[2],                                                           \
        fxyz[2] = vol->offset[2] + ijk[2]*vol->step[2*3+2]              \
        )
#define LOOP_Z_OMP(k,vol)                                               \
    for (                                                               \
    long k = 0;                                                         \
    k < vol->dim[2];                                                    \
    ++k                                                                 \
        )
#define LOOP_Y(ijk,fxyz,vol)                                            \
    for (                                                               \
    ijk[1] = 0,                                                         \
        fxyz[1] = vol->offset[1] + ijk[2]*vol->step[1*3+2];             \
    ijk[1] < vol->dim[1];                                               \
    ++ijk[1],                                                           \
        fxyz[2] = vol->offset[2] + ijk[2]*vol->step[2*3+2]              \
        + ijk[1]*vol->step[2*3+1],                                      \
        fxyz[1] = vol->offset[1] + ijk[2]*vol->step[1*3+2]              \
        + ijk[1]*vol->step[1*3+1]                                       \
        )
#define LOOP_X(ijk,fxyz,vol)                                            \
    for (                                                               \
    ijk[0] = 0,                                                         \
        fxyz[0] = vol->offset[0] + ijk[2]*vol->step[0*3+2]              \
        + ijk[1]*vol->step[0*3+1];                                      \
    ijk[0] < vol->dim[0];                                               \
    ++ijk[0],                                                           \
        fxyz[0] += vol->step[0*3+0],                                    \
        fxyz[1] += vol->step[1*3+0],                                    \
        fxyz[2] += vol->step[2*3+0]                                     \
        )

#define PROJECT_Z(xyz,proj)                                             \
    (xyz[0] * proj[2*3+0] + xyz[1] * proj[2*3+1] + xyz[2] * proj[2*3+2])
#define PROJECT_Y(xyz,proj)                                             \
    (xyz[0] * proj[1*3+0] + xyz[1] * proj[1*3+1] + xyz[2] * proj[1*3+2])
#define PROJECT_X(xyz,proj)                                             \
    (xyz[0] * proj[0*3+0] + xyz[1] * proj[0*3+1] + xyz[2] * proj[0*3+2])

#endif
