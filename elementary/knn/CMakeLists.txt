cmake_minimum_required(VERSION 3.12)
project(knn)

find_package(opencv)

include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES main.cpp)
add_executable(knn ${SOURCE_FILES})
target_link_libraries(knn ${OpenCV_LIBS})
