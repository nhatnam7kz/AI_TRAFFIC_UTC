cmake_minimum_required(VERSION 3.10)
project(YOLO_Edge_Optimize)

set(CMAKE_CXX_STANDARD 11)

# Chỉ định thư mục chứa file Header
include_directories(include)

# Biên dịch file main thành file thực thi
add_executable(edge_optimize src/main.cpp)