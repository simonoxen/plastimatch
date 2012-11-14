/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pointset_h_
#define _pointset_h_

#include "plmbase_config.h"
#include <string>
#include <vector>
#include "compiler_warnings.h"

class Pstring;

class PLMBASE_API Point {
public:
    Point () {}
    Point (const std::string& label, float x, float y, float z) {
        UNUSED_VARIABLE (label);
        p[0] = x;
        p[1] = y;
        p[2] = z;
    }
    Point (float x, float y, float z) {
        p[0] = x;
        p[1] = y;
        p[2] = z;
    }
public:
    float p[3];
public:
    void set_label (const char* s) {
        UNUSED_VARIABLE (s);
    }
    std::string get_label (void) const {
        return "";
    }
};

class PLMBASE_API Labeled_point {
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
class PLMBASE_API Pointset {
public:
    Pointset();
    ~Pointset();
public:
    std::vector<T> point_list;
public:
    void load (const char *fn);
    void load_txt (const char *fn);
    void load_fcsv (const char *fn);
    void save (const char *fn);
    void save_fcsv (const char *fn);
    void save_txt (const char *fn);
    void set_ras (const Pstring& p);

    /* Insert single points */
    void insert_lps (const std::string& label, float x, float y, float z);
    void insert_ras (const std::string& label, float x, float y, float z);

    /* Return the number of points */
    size_t count (void) const;

    /* Truncate points at the end of the list */
    void truncate (size_t new_length);
};

typedef Pointset<Labeled_point> Labeled_pointset;
typedef Pointset<Point> Unlabeled_pointset;

#endif
