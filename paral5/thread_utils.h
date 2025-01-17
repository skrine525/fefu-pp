#ifndef THREAD_UTILS_H
#define THREAD_UTILS_H

#include <omp.h>

unsigned g_num_threads;

void set_num_threads_auto() {
    g_num_threads = omp_get_max_threads();
    omp_set_num_threads(g_num_threads);
}

void set_num_threads(unsigned T) {
    g_num_threads = T;
    omp_set_num_threads(T);
};

unsigned get_num_threads() {
    return g_num_threads;
}

#endif 