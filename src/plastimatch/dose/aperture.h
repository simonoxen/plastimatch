/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _aperture_h_
#define _aperture_h_

class Aperture {
public:
    Aperture ();

public:
    double ap_offset;     /* distance from beam nozzle */
    double vup[3];        /* orientation */
    double ic [2];        /* center */
    int ires[2];          /* resolution (vox) */
    double ic_room[3];    /* loc of center (room coords) */
    double ul_room[3];    /* loc of upper left corder (room coords) */
    double incr_r[3];     /* row increment vector */
    double incr_c[3];     /* col increment vector */
    double nrm[3];        /* unit vec: normal */
    double pdn[3];        /* unit vec: down */
    double prt[3];        /* unit vec: right */
    double tmp[3];
};

#endif

