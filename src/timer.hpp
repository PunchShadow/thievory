
#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <iostream>

struct Timer {
  std::string timerStr;
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  std::chrono::duration<float> duration;

  Timer(std::string str) {
    timerStr = str;
    start = std::chrono::high_resolution_clock::now();
  }

  ~Timer() {
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    float ms = duration.count() * 1000.0f;
    std::cout << timerStr << ": " << ms << " ms" << std::endl;
  }

  float GetDuration() {
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    return duration.count() * 1000.0f;
  }
};

#endif
