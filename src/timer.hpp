
#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <iostream>

struct Timer {
  std::string timerStr;
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  std::chrono::duration<float> duration;
  bool silent;

  Timer(std::string str, bool silent = false) : silent(silent) {
    timerStr = str;
    start = std::chrono::high_resolution_clock::now();
  }

  ~Timer() {
    if (!silent) {
      end = std::chrono::high_resolution_clock::now();
      duration = end - start;
      float sec = duration.count();
      std::cout << timerStr << sec << " s" << std::endl;
    }
  }

  // Returns elapsed time in seconds
  float GetDurationSec() {
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    return duration.count();
  }
};

#endif
