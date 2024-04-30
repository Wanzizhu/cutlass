/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once
#include "cutlass/cutlass.h"
#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

// #include <cuda_runtime.h>

struct GPU_Clock {
private:
  std::vector<sycl::event> gpu_event_vec;
  std::vector<double> gpu_time_vec;

  inline void get_gpu_time_from_events() {
    for (const auto gpu_event : gpu_event_vec) {
      auto gpu_start = gpu_event.template get_profiling_info<
          sycl::info::event_profiling::command_start>();
      auto gpu_end = gpu_event.template get_profiling_info<
          sycl::info::event_profiling::command_end>();
      double gpu_time = (gpu_end - gpu_start) / 1000000.0; // convert to ms
      gpu_time_vec.push_back(gpu_time);
    }
  }

public:
  void start() {
    gpu_event_vec.clear();
    gpu_time_vec.clear();
  }
  void add_gpu_event(sycl::event gpu_event) {
    gpu_event_vec.push_back(gpu_event);
  }
  void end() {
    // auto last_event = gpu_event_vec.back();
    // last_event.wait();
    get_gpu_time_from_events();
  }

  // print total time in milliseconds
  float milliseconds() {
    if (gpu_time_vec.size() == 0) {
      printf("No GPU events recorded, maybe not running GPU_Clock::end() "
             "before\n");
      return 0.0;
    }
    double time =
        std::accumulate(gpu_time_vec.begin(), gpu_time_vec.end(), 0.0);
    return time;
  }

  float seconds() { return milliseconds() * float(1e-3); }

  void print_profiling_data() {

    if (gpu_time_vec.size() == 0) {
      printf("No GPU events recorded, maybe not running GPU_Clock::end() "
             "before\n");
      return;
    }
    std::sort(gpu_time_vec.begin(), gpu_time_vec.end());
    double min_time = gpu_time_vec[0];
    double max_time = gpu_time_vec[gpu_time_vec.size() - 1];
    double avg_time =
        std::accumulate(gpu_time_vec.begin(), gpu_time_vec.end(), 0.0) /
        gpu_time_vec.size();
    printf("Min GPU time: %.5f ms, Max GPU time: %.5f ms, Average GPU time: "
           "%.5f ms, \n",
           min_time, max_time, avg_time);
  }
};
