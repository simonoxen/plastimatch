/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <string>
#include <stdio.h>
#include "direction_cosines.h"
#include "plm_math.h"

#define DIRECTION_COSINES_IDENTITY_THRESH 1e-9

class Direction_cosines_private {
public:
    float direction_cosines[9];
    float inv_direction_cosines[9];
public:
    Direction_cosines_private () {
        this->set_identity ();
    }
    ~Direction_cosines_private () {
    }
public:
    void set_identity () {
	direction_cosines[0] = 1.;
	direction_cosines[1] = 0.;
	direction_cosines[2] = 0.;
	direction_cosines[3] = 0.;
	direction_cosines[4] = 1.;
	direction_cosines[5] = 0.;
	direction_cosines[6] = 0.;
	direction_cosines[7] = 0.;
	direction_cosines[8] = 1.;
    }
};

Direction_cosines::Direction_cosines ()
{
    this->d_ptr = new Direction_cosines_private;
    this->set_identity ();
}

Direction_cosines::~Direction_cosines ()
{
    delete this->d_ptr;
}

Direction_cosines::operator const float* () const
{
    return m_direction_cosines;
}

Direction_cosines::operator float* ()
{
    return this->m_direction_cosines;
}

void Direction_cosines::set_identity () {
    m_direction_cosines[0] = 1.;
    m_direction_cosines[1] = 0.;
    m_direction_cosines[2] = 0.;
    m_direction_cosines[3] = 0.;
    m_direction_cosines[4] = 1.;
    m_direction_cosines[5] = 0.;
    m_direction_cosines[6] = 0.;
    m_direction_cosines[7] = 0.;
    m_direction_cosines[8] = 1.;
    }
void Direction_cosines::set_rotated_1 () {
	m_direction_cosines[0] = 0.894427190999916;
	m_direction_cosines[1] = 0.447213595499958;
	m_direction_cosines[2] = 0.;
	m_direction_cosines[3] = -0.447213595499958;
	m_direction_cosines[4] = 0.894427190999916;
	m_direction_cosines[5] = 0.;
	m_direction_cosines[6] = 0.;
	m_direction_cosines[7] = 0.;
	m_direction_cosines[8] = 1.;
    }
void Direction_cosines::set_rotated_2 () {
	m_direction_cosines[0] = M_SQRT1_2;
	m_direction_cosines[1] = -M_SQRT1_2;
	m_direction_cosines[2] = 0.;
	m_direction_cosines[3] = M_SQRT1_2;
	m_direction_cosines[4] = M_SQRT1_2;
	m_direction_cosines[5] = 0.;
	m_direction_cosines[6] = 0.;
	m_direction_cosines[7] = 0.;
	m_direction_cosines[8] = 1.;
    }
void Direction_cosines::set_rotated_3 () {
	m_direction_cosines[0] = -0.855063803257865;
	m_direction_cosines[1] = 0.498361271551590;
	m_direction_cosines[2] = -0.143184969098287;
	m_direction_cosines[3] = -0.428158353951640;
	m_direction_cosines[4] = -0.834358655093045;
	m_direction_cosines[5] = -0.347168631377818;
	m_direction_cosines[6] = -0.292483018822660;
	m_direction_cosines[7] = -0.235545489638006;
	m_direction_cosines[8] = 0.926807426605751;
    }
void Direction_cosines::set_skewed () {
	m_direction_cosines[0] = 1.;
	m_direction_cosines[1] = 0.;
	m_direction_cosines[2] = 0.;
	m_direction_cosines[3] = M_SQRT1_2;
	m_direction_cosines[4] = M_SQRT1_2;
	m_direction_cosines[5] = 0.;
	m_direction_cosines[6] = 0.;
	m_direction_cosines[7] = 0.;
	m_direction_cosines[8] = 1.;
    }

void Direction_cosines::set (const float dc[]) {
	for (int i = 0; i < 9; i++) {
	    m_direction_cosines[i] = dc[i];
	}
    }

bool Direction_cosines::set_from_string (std::string& str) {
    float dc[9];
    int rc;

    /* First check presets */
    if (str == "identity") {
        this->set_identity ();
        return true;
    }
    else if (str == "rotated-1") {
        this->set_rotated_1 ();
        return true;
    }
    else if (str == "rotated-2") {
        this->set_rotated_2 ();
        return true;
    }
    else if (str == "rotated-3") {
        this->set_rotated_3 ();
        return true;
    }
    else if (str == "skewed") {
        this->set_skewed ();
        return true;
    }

    /* Not a preset, must be 9 digit string */
    rc = sscanf (str.c_str(), "%g %g %g %g %g %g %g %g %g", 
        &dc[0], &dc[1], &dc[2],
        &dc[3], &dc[4], &dc[5],
        &dc[6], &dc[7], &dc[8]);
    if (rc != 9) {
        return false;
    }
    this->set (dc);
    return true;
}

bool Direction_cosines::is_identity () {
    Direction_cosines id;
    float frob = 0.;
	
    for (int i = 0; i < 9; i++) {
        frob += fabs (m_direction_cosines[i] - id.m_direction_cosines[i]);
    }
    return frob < DIRECTION_COSINES_IDENTITY_THRESH;
}
