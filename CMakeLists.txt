cmake_minimum_required(VERSION 3.10)

project(ellipse_fit_ceres)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

include_directories(${CERES_INCLUDE_DIRS})

add_executable(
    ellipse_fit
    ellipse_fit.cpp
)
target_link_libraries(ellipse_fit Eigen3::Eigen ${CERES_LIBRARIES})