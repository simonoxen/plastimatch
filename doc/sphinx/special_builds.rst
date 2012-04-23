Special build instructions
==========================

Coverage (gcov) build
---------------------
#. Build plastimatch in debug mode, with special flags::

    CMAKE_BUILD_TYPE:STRING=Debug
    CMAKE_CXX_FLAGS:STRING=-fprofile-arcs -ftest-coverage

#. To run standalone, go to the build directory, and run::

    ./run_lcov.sh

#. To run an entire test suite, do this::

    ctest -D NightlyCoverage

Memcheck (valgrind) build
-------------------------
#. Build ITK in debug mode (optional, but recommended).

#. Build plastimatch in debug mode.

#. To run an individual test case, do this::

    valgrind --leak-check=yes plastimatch [options]

#. To run an entire test suite, normally, you would do something like this::

    ctest -D ExperimentalMemCheck

   However, this doesn't work, because ctest prepends valgrind directly
   to the test command, which in our case is a script.  
