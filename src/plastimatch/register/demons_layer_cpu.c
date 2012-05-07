/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "demons.h"
Volume*
demons (Volume* fixed, Volume* moving, Volume* moving_grad, 
	Volume* vf_init, Demons_parms* parms)
{
    return demons_c (fixed, moving, moving_grad, vf_init, parms);
}
