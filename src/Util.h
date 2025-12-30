#ifndef UTIL_H
#define UTIL_H

#include <chrono>

std::chrono::high_resolution_clock::time_point getChronoNow();

double evalTime(const std::chrono::system_clock::time_point& start);

#endif
