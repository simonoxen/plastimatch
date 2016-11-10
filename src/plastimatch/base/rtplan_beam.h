/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtplan_beam_h_
#define _rtplan_beam_h_

#include "plmbase_config.h"
#include <string>

class Rtplan_control_pt;

/*! \brief 
 * The Rtplan_beam class describes a single beam within an Rtplan.
 */
class PLMBASE_API Rtplan_beam {
public:
    /*! \brief Beam name */
    std::string name;    
    /*! \brief Beam description */
    std::string description;
    /*! \brief Meterset at end of all control points */
    float final_cumulative_meterset_weight;
    /*! \brief Coordiates of point where beam dose is specified */
    std::string beam_dose_specification_point;
    /*! \brief Dose in Gy at beam specification point */
    float beam_dose;
    float snout_position;

    /*! \brief Control point list */
    std::vector<Rtplan_control_pt*> cplist;

public:
    Rtplan_beam();
    ~Rtplan_beam();

    void clear ();
    Rtplan_control_pt* add_control_pt ();
    bool check_isocenter_identical ();
#if defined (commentout)
    float* get_isocenter_pos (); //float[3], dicom coordinate
#endif
};


#endif