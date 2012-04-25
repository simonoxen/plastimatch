/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xio_studyset_h_
#define _xio_studyset_h_

#include "plm_config.h"
#include <string>
#include <vector>
#include "cxt_io.h"

enum Xio_version {
    XIO_VERSION_UNKNOWN,
    XIO_VERSION_4_2_1,         /* MGH proton Xio */
    XIO_VERSION_4_33_02,       /* Older MGH photon Xio */
    XIO_VERSION_4_5_0,         /* Current MGH photon Xio */
};

using namespace std;

class Xio_studyset_slice
{
public:
    string name;
    float location;

    string filename_scan;
    string filename_contours;
    
public:
    Xio_studyset_slice (string slice_filename_scan, const float slice_location);
    ~Xio_studyset_slice ();

    bool operator < (const Xio_studyset_slice &cmp) const
    {
	return location < cmp.location;
    }
};

class Xio_studyset
{
public:
    std::string studyset_dir;
    int number_slices;
    float thickness;
    vector<Xio_studyset_slice> slices;
    
public:
    Xio_studyset (const char *studyset_dir);
    ~Xio_studyset ();
};

#endif
