######################################################
##  Check if nullptr is supported by the compiler
##  Returns HAVE_NULLPTR
######################################################
set (NULLPTR_TEST_SOURCE "int main() {nullptr;}")
check_cxx_source_compiles ("${NULLPTR_TEST_SOURCE}" HAVE_NULLPTR)
