#include <stdio.h>
#include "smart_pointer.h"

class A {
public:
    SMART_POINTER_SUPPORT(A);
public:
    int *val;
public:
    A () {
        val = new int;
        *val = 3;
    }
    ~A () {
        delete val;
    }
};

int 
main (int argc, char* argv[])
{
    A::Pointer a;

    if (a) {
        printf ("A is non-null\n");
    } else {
        printf ("A is null\n");
    }

    return 0;
}
