cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

add_definitions(-std=c++11)

project(playback)

#add_compile_options(-Wall)
add_compile_options(-g)
add_compile_options(-O2)

find_package(PCL 1.2 REQUIRED)

include_directories(src)
include_directories(SYSTEM ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

add_executable (ukf_highway
                src/main.cpp
                src/measurement_model.cpp
                src/measurement_model_lidar.cpp
                src/measurement_model_radar.cpp
                src/motion_model.cpp
                src/render/render.cpp
                src/tools.cpp
                src/ukf.cpp
                )
target_link_libraries (ukf_highway ${PCL_LIBRARIES})
