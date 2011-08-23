/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "dcmtk_file.h"
#include "dcmtk_series.h"
#include "print_and_exit.h"

Dcmtk_series::Dcmtk_series ()
{
}

Dcmtk_series::~Dcmtk_series ()
{
    std::list<Dcmtk_file*>::iterator it;
    for (it = m_flist.begin(); it != m_flist.end(); ++it) {
	delete (*it);
    }
}

void
Dcmtk_series::insert (Dcmtk_file *df)
{
    m_flist.push_back (df);
}

void
Dcmtk_series::sort (void)
{
    m_flist.sort (dcmtk_file_compare_z_position);
}

void
Dcmtk_series::debug (void) const
{
    std::list<Dcmtk_file*>::const_iterator it;
    for (it = m_flist.begin(); it != m_flist.end(); ++it) {
	Dcmtk_file *df = (*it);
	df->debug ();
    }
}
