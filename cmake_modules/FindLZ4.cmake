# - Find LZ4 include dirs and libraries
# Use this module by invoking find_package with the form:
#  find_package(LZ4
#    [REQUIRED]             # Fail with error if LZ4 is not found
#    )
# Once done, this will define
#
#  LZ4_FOUND - system has LZ4
#  LZ4_INCLUDE_DIR - the LZ4 include directory
#  LZ4_LIBRARY - the LZ4 library
#  LZ4::lz4 - imported target for the LZ4 library

find_path(LZ4_INCLUDE_DIR NAMES lz4frame.h)
find_library(LZ4_LIBRARY NAMES lz4)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LZ4 REQUIRED_VARS LZ4_LIBRARY LZ4_INCLUDE_DIR)

if(LZ4_FOUND AND NOT TARGET LZ4::lz4)
  add_library(LZ4::lz4 UNKNOWN IMPORTED)
  set_target_properties(LZ4::lz4 PROPERTIES
    IMPORTED_LOCATION "${LZ4_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${LZ4_INCLUDE_DIR}")
endif()

mark_as_advanced(LZ4_INCLUDE_DIR LZ4_LIBRARY)
