#include <stdio.h>
#include "dlib/smart_pointers.h"

class A {
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

void
foo_1 (dlib::shared_ptr<A> a)
{
    (*a->val) ++;
    printf ("A %d\n", *(a->val));
}

void
test_1 ()
{
    dlib::shared_ptr<A> a (new A);
    for (int i = 0; i < 10; i++) {
        foo_1 (a);
    }
}

class B {
public:
    typedef dlib::shared_ptr<B> Pointer;
public:
    int *val;
public:
    B () {
        val = new int;
        *val = 3;
    }
    ~B () {
        delete val;
    }
public:
    static B::Pointer New () {
        return dlib::shared_ptr<B> (new B);
    }
};

void
foo_2 (B::Pointer b)
{
    (*b->val) ++;
    printf ("B %d\n", *(b->val));
}

void
test_2 ()
{
    B::Pointer b = B::New ();
    for (int i = 0; i < 10; i++) {
        foo_2 (b);
    }
}

int 
main (int argc, char* argv[])
{

    test_1();
    test_2();
    return 0;
}
