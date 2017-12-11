##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
# macro: sb_option
# Create an option in the cmake-gui that is passed to a nested superbuild
macro (sb_option _var _desc _defval)
  option (${_var} ${_desc} ${_defval})
  list (APPEND sb_cmake_vars ${_var})
endmacro ()
