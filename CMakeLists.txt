# CMakeLists.txt for VisionPreview

target_include_directories(xr PUBLIC ${CMAKE_CURRENT_LIST_DIR})

get_filename_component(_MODULE_DIRNAME ${CMAKE_CURRENT_LIST_DIR} NAME)
set(_DEPS_TARGET "${_MODULE_DIRNAME}_deps")
add_library(${_DEPS_TARGET} INTERFACE)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc calib3d highgui videoio)

target_link_libraries(${_DEPS_TARGET} INTERFACE ${OpenCV_LIBS})

set_property(GLOBAL APPEND PROPERTY XR_MODULE_DEPS ${_DEPS_TARGET})
