
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

# for aot build
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
  target_compile_options(${exe} PRIVATE -fsycl -fsycl-targets=spir64_gen)
  target_link_options(${exe} PRIVATE ${SYCL_LINK_OPTIONS})
endforeach()
