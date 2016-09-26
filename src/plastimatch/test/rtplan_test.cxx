/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "compiler_warnings.h"
#include "dcmtk_rt_study.h"
#include "rtplan.h"
#include "rtplan_beam.h"
#include "rtplan_control_pt.h"
#include "string_util.h"

int
main (int argc, char *argv[])
{
    char *dicom_dir;
    UNUSED_VARIABLE (dicom_dir);
    if (argc == 2) {
	dicom_dir = argv[1];
    } else {
	printf ("Usage: rtplan_test output_dir\n");
	exit (1);
    }

    Dcmtk_rt_study drs;
    Rt_study_metadata::Pointer rsm = Rt_study_metadata::New ();
    Rtplan::Pointer rtplan = Rtplan::New ();

    drs.set_rt_study_metadata (rsm);
    drs.set_rtplan (rtplan);

    /* Fill in metadata */
    rsm->set_patient_name ("Test^Rtplan");

    /* Fill in plan data */
    for (size_t i = 0; i < 1; i++) {
        std::string beam_name = string_format ("Beam %d", (int) i);
        Rtplan_beam *beam = rtplan->add_beam (beam_name, (int) i);
        beam->description = string_format ("Beam %d description", (int) i);

        float cum_gp = 0.f;
        for (size_t seg = 0; seg < 3; seg++) {
            float snout_pos = 250.f;
            float energy = 100.f + seg * 25.f;
            float gp = 25.f;
            for (size_t cpi = 0; cpi < 2; cpi++) {
                Rtplan_control_pt *cp = beam->add_control_pt ();
                cp->cumulative_meterset_weight = cum_gp + cpi * gp;
                cp->nominal_beam_energy = energy;
                cp->meterset_rate = 400;
                cp->snout_position = snout_pos;

                cp->scan_spot_position_map.push_back (-10.f - seg);
                cp->scan_spot_position_map.push_back (-10.f);
                cp->scan_spot_position_map.push_back (+10.f + seg);
                cp->scan_spot_position_map.push_back (-10.f);
                cp->scan_spot_position_map.push_back (-10.f - seg);
                cp->scan_spot_position_map.push_back (+10.f);
                cp->scan_spot_position_map.push_back (+10.f + seg);
                cp->scan_spot_position_map.push_back (+10.f);

                cp->scan_spot_meterset_weights.push_back (5);
                cp->scan_spot_meterset_weights.push_back (5);
                cp->scan_spot_meterset_weights.push_back (10);
                cp->scan_spot_meterset_weights.push_back (5);
            }
            cum_gp += gp;
        }
        beam->final_cumulative_meterset_weight = cum_gp;
    }

    /* Save to file */
    drs.save (dicom_dir);
    
    return 0;
}
