cmake_minimum_required(VERSION 2.8.12)

project(proj)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(proj proj.cpp)
target_link_libraries(proj ${OpenCV_LIBS})
