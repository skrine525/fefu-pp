#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <filesystem> 
#include <condition_variable>
#include "thread_utils.h"

#define N (1u << 27)
#define CACHE_LINE 64
#define METHOD_ENTRY(method) {method, #method}

struct partial_sum_t {
    alignas(CACHE_LINE) unsigned val;
};

struct table_row {
    bool match;
    double time;
    double speedup;
    double efficiency;
};

class barrier {
    std::condition_variable cv;
    std::mutex mtx;
    bool generation = false;
    unsigned T;
    const unsigned T0;

public:
    barrier(unsigned threads) : T(threads), T0(threads) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> l(mtx);
        if (--T == 0) {
            T = T0;
            generation = !generation;
            cv.notify_all();
        }
        else {
            bool my_barrier = generation;
            cv.wait(l, [&] { return my_barrier != generation; });
        }
    }
};


typedef unsigned (*sum_ptr)(const unsigned* v, size_t n);

unsigned expected_sum(unsigned T, size_t n) {
    return (n * T) + (n * (n - 1)) / 2;
}

std::vector<table_row> run_experiment(sum_ptr sum) {
    unsigned P = get_num_threads();
    std::vector<table_row> table(P, { false, 0.0, 0.0, 0.0 });
    size_t n = N;
    auto V = std::make_unique<unsigned[]>(n);

    for (unsigned T = 1; T <= P; ++T) {
        set_num_threads_auto();

        for (size_t i = 0; i < n; ++i)
            V[i] = i + T;

        auto t1 = std::chrono::steady_clock::now();
        table[T - 1].match = (sum(V.get(), n) == expected_sum(T, n));
        auto t2 = std::chrono::steady_clock::now();

        table[T - 1].time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 1000.0;

        if (T > 1) {
            table[T - 1].speedup = table[0].time / table[T - 1].time;
            table[T - 1].efficiency = table[T - 1].speedup / T;
        }
        else {
            table[T - 1].speedup = 1.0;
            table[T - 1].efficiency = 1.0;
        }
    }

    return table;
}

unsigned sum(const unsigned* v, size_t n) {
    unsigned sum = 0;
    for (int i = 0; i < n; i++)
        sum += v[i];
    return sum;
}

unsigned sum_omp_reduce(const unsigned* v, size_t n) {
    unsigned sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < static_cast<int>(n); i++)
        sum += v[i];
    return sum;
}

unsigned sum_round_robin(const unsigned* v, size_t n) {
    unsigned sum = 0;
    unsigned T = 0;
    unsigned* partial_sums = nullptr;

#pragma omp parallel
    {
        T = omp_get_num_threads();
        unsigned t = omp_get_thread_num();
#pragma omp single
        {
            partial_sums = (unsigned*)calloc(T, sizeof(unsigned));
        }

        for (unsigned i = t; i < n; i += T)
            partial_sums[t] += v[i];
    }

    for (unsigned i = 0; i < T; ++i)
        sum += partial_sums[i];
    free(partial_sums);
    return sum;
}

unsigned sum_round_robin_aligned(const unsigned* v, size_t n) {
    unsigned sum = 0;
    partial_sum_t* partial_sums;
    unsigned T;
#pragma omp parallel
    {
        unsigned t = omp_get_thread_num();
#pragma omp single
        {
            T = omp_get_num_threads();
            partial_sums = (partial_sum_t*)calloc(sizeof partial_sums[0], T);
        }

        for (unsigned i = t; i < n; i += T)
            partial_sums[t].val += v[i];
    }

    for (unsigned i = 0; i < T; i++)
        sum += partial_sums[i].val;
    free(partial_sums);
    return sum;
}

unsigned sum_seq(const unsigned* v, size_t n) {
    unsigned sum = 0;
    for (size_t i = 0; i < n; ++i)
        sum += v[i];
    return sum;
}

unsigned vector_sum_la(const unsigned* v, size_t n) {
    unsigned T;
    unsigned sum = 0;
    partial_sum_t* partial_sums;
#pragma omp parallel
    {
        unsigned t = omp_get_thread_num();
#pragma omp single
        {
            T = omp_get_num_threads();
            partial_sums = (partial_sum_t*)malloc(sizeof partial_sums[0] * T);
        }
        partial_sums[t].val = 0;
        unsigned s_t = n / T, b_t = n % T;

        if (t <= b_t)
            b_t = ++s_t * t;
        else
            b_t += s_t * t;

        unsigned e_t = b_t + s_t;

        for (unsigned i = b_t; i < e_t; i++)
            partial_sums[t].val += v[i];
    }

    for (unsigned i = 0; i < T; i++)
        sum += partial_sums[i].val;
    free(partial_sums);
    return sum;
}

unsigned sum_spp_cs(const unsigned* v, size_t n) {
    unsigned sum = 0;
    unsigned T = get_num_threads();
    std::vector<std::thread> workers(T - 1);
    std::mutex mtx;

    auto worker_proc = [&v, &sum, &mtx, n, T](unsigned t) {
        unsigned s_t = n / T, b_t = n % T;

        if (t < b_t)
            b_t = ++s_t * t;
        else
            b_t += s_t * t;

        unsigned e_t = b_t + s_t;

        unsigned my_sum = 0;
        for (unsigned i = b_t; i < e_t; ++i)
            my_sum += v[i];

        {
            std::scoped_lock lock(mtx);
            sum += my_sum;
        }
    };

    for (unsigned t = 1; t < T; ++t)
        workers[t - 1] = std::thread(worker_proc, t);

    worker_proc(0);

    for (auto& worker : workers)
        worker.join();

    return sum;
}

unsigned sum_mutex(const unsigned* v, size_t n) {
    unsigned sum = 0;
#pragma omp parallel
    {
        unsigned my_sum = 0;
#pragma omp for
        for (int i = 0; i < static_cast<int>(n); i++) {
            my_sum += v[i];
        }

#pragma omp critical
        sum += my_sum;
    }
    return sum;
}

unsigned sum_barrier(const unsigned* v, size_t n) {
    unsigned sum = 0;
    unsigned T = get_num_threads();
    barrier sync_barrier(T);

    std::vector<std::thread> workers(T - 1);

    auto worker_proc = [&v, &sum, &sync_barrier, n, T](unsigned t) {
        unsigned s_t = n / T, b_t = n % T;

        if (t < b_t)
            b_t = ++s_t * t;
        else
            b_t += s_t * t;

        unsigned e_t = b_t + s_t;

        unsigned my_sum = 0;
        for (unsigned i = b_t; i < e_t; ++i)
            my_sum += v[i];

        sync_barrier.arrive_and_wait();

        {
            static std::mutex sum_mutex;
            std::scoped_lock lock(sum_mutex);
            sum += my_sum;
        }
    };

    for (unsigned t = 1; t < T; ++t)
        workers[t - 1] = std::thread(worker_proc, t);

    worker_proc(0);

    for (auto& worker : workers)
        worker.join();

    return sum;
}

void to_csv(std::ostream& os, const std::vector<table_row>& v) {
    os << "Threads,Match,Time (ms),Speedup,Efficiency\n";

    for (size_t i = 0; i < v.size(); ++i) {
        os << (i + 1) << ","
            << v[i].match << ","
            << v[i].time << ","
            << v[i].speedup << ","
            << v[i].efficiency << "\n";
    }
}

int main(int argc, char** argv) {
    set_num_threads_auto();
    auto V = std::make_unique<unsigned[]>(N);
    for (size_t i = 0; i < N; ++i)
        V[i] = i;

    std::pair<sum_ptr, const char*> methods[] = {
        METHOD_ENTRY(sum),
        METHOD_ENTRY(sum_omp_reduce),
        METHOD_ENTRY(sum_round_robin),
        METHOD_ENTRY(sum_round_robin_aligned),
        METHOD_ENTRY(sum_seq),
        METHOD_ENTRY(vector_sum_la),
        METHOD_ENTRY(sum_spp_cs),
        METHOD_ENTRY(sum_mutex),
        METHOD_ENTRY(sum_barrier)
    };

    for (const auto& [method, name] : methods) {
        auto results = run_experiment(method);
        std::string path = "../results/csv/results_" + std::string(name) + ".csv";

        std::filesystem::path dir = std::filesystem::path(path).parent_path();
        if (!std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir);
        }

        std::ofstream csv_file(path);
        if (csv_file.is_open()) {
            to_csv(csv_file, results);
            std::cout << "Results saved to " << path << "\n";
        }
        else {
            std::cerr << "Error: Could not open file for writing.\n";
        }
    }

    return 0;
}

