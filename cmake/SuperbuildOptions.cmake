##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------

# macro: sb_option
# Create an option in the cmake-gui, and mark the variable as a superbuild
# variable to be passed to the inner build
macro (sb_option _var _desc _defval)
  option (${_var} ${_desc} ${_defval})
  list (APPEND sb_cmake_vars ${_var})
endmacro ()

# macro: sb_set
# Set a variable and add it to the list of superbuild variables
# that are passed to the inner build
macro (sb_set _var _val)
  set (${_var} ${_val})
  list (APPEND sb_cmake_vars ${_var})
endmacro ()

# macro: sb_variable
# Add a variable to the list of superbuild variables that are passed to the
# inner build
macro (sb_variable _var)
  list (APPEND sb_cmake_vars ${_var})
endmacro ()
