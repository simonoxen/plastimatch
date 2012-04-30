/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _proton_dose_opts_h_
#define _proton_dose_opts_h_


typedef struct proton_dose_parms Proton_dose_parms;

void proton_dose_parse_args (
        proton_dose_parms* parms, 
        int argc, char* argv[]
);

#endif
