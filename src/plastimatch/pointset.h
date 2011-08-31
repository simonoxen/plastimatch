/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pointset_h_
#define _pointset_h_

#include "plm_config.h"
#include <string>
#include <vector>

class gpuit_EXPORT Point {
public:
    Point () {}
    Point (const std::string& label, float x, float y, float z) {
	p[0] = x;
	p[1] = y;
	p[2] = z;
    }
public:
    float p[3];
public:
    void set_label (const char* s) {}
    std::string get_label (void) const {
	return "";
    }
};

class gpuit_EXPORT Labeled_point {
public:
    Labeled_point () {}
    Labeled_point (const std::string& label, float x, float y, float z) {
	this->label = label;
	p[0] = x;
	p[1] = y;
	p[2] = z;
    }
public:
    std::string label;
    float p[3];
public:
    void set_label (const char* s) {
	this->label = s;
    }
    std::string get_label (void) const {
	return this->label;
    }
};

template<class T>
class gpuit_EXPORT Pointset {
  public:
    std::vector<T> point_list;
  public:
    void load_fcsv (const char *fn);
    void save_fcsv (const char *fn);
    void insert_lps (const std::string& label, float x, float y, float z);
    void insert_ras (const std::string& label, float x, float y, float z);
};

typedef Pointset<Labeled_point> Labeled_pointset;
typedef Pointset<Point> Unlabeled_pointset;

typedef struct pointset_old Pointset_old;
struct pointset_old {
    int num_points;
    float *points;
};

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
Pointset_old*
pointset_load (const char *fn);
gpuit_EXPORT
void
pointset_save (Pointset_old* ps, const char *fn);
gpuit_EXPORT
void
pointset_save_fcsv_by_cluster (Pointset_old* ps, int *clust_id, int which_cluster, const char *fn);
gpuit_EXPORT
Pointset_old *
pointset_create (void);
gpuit_EXPORT
void
pointset_destroy (Pointset_old *ps);

gpuit_EXPORT
void
pointset_resize (Pointset_old *ps, int new_size);
gpuit_EXPORT
void
pointset_add_point (Pointset_old *ps, float lm[3]);
gpuit_EXPORT
void
pointset_add_point_noadjust (Pointset_old *ps, float lm[3]);
gpuit_EXPORT
void
pointset_debug (Pointset_old* ps);

#if defined __cplusplus
}
#endif

#endif
