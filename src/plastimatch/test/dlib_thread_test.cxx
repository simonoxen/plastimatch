#include <stdio.h>
#include "dlib_threads.h"

class Struct {
public:
//    dlib::thread_function *tf;
};

void thread_func (void* param)
{
    int* i = (int*) param;
    ++*i;
}

int main ()
{
//    Struct s;
    int j = 0;
//    s.thread_function = new dlib::thread_function (thread_func, (void*) &j);
//    delete s.thread_function;
    printf ("j = %d\n", j);
    return 0;
}
