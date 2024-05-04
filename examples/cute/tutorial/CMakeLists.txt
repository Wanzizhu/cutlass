
cmake_minimum_required(VERSION 3.19)
project(Example VERSION 0.1)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_COMPILER icc)
set(CMAKE_CXX_COMPILER icpx)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(INCLUD_DIR /home/zhuwei/workspace/cutlass/tools/util/include /home/zhuwei/workspace/cutlass/include/)
message(STATUS "INCLUD_DIR: ${INCLUD_DIR}")

# for aot 
set(AOT_TARGET pvc)
set(SYCL_LINK_OPTIONS "-fsycl")
set(SYCL_LINK_OPTIONS ${SYCL_LINK_OPTIONS} "-fsycl-targets=spir64_gen")
set(SYCL_LINK_OPTIONS ${SYCL_LINK_OPTIONS} "-Xsycl-target-backend=spir64_gen")
set(SYCL_LINK_OPTIONS ${SYCL_LINK_OPTIONS} "-device ${AOT_TARGET} -internal_options -cl-intel-256-GRF-per-thread")
message(STATUS "SYCL_LINK_OPTIONS: ${SYCL_LINK_OPTIONS}")


file(GLOB sources *.cpp)

foreach(src ${sources})
  get_filename_component(exe ${src} NAME_WE)
  add_executable(${exe} ${src})
  target_include_directories(${exe} PRIVATE ${INCLUD_DIR})
  target_compile_options(${exe} PRIVATE -g -fsycl -fsycl-targets=spir64_gen -ffast-math)
  target_link_options(${exe} PRIVATE -lblas ${SYCL_LINK_OPTIONS})
endforeach()

add_custom_target(clean-custom
    COMMAND ${CMAKE_COMMAND} --build . --target clean
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/dump
    COMMENT "Custom clean target: removing dump directory and cleaning build files"
)
