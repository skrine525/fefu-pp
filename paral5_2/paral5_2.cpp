#include <iostream>
#include <concepts>
#include <vector>
#include <thread>
#include <mutex>
#include "thread_utils.h"

template <class T = unsigned>
class additive_monoid {
    T x = T{ 0 };
public:
    additive_monoid() = default;
    explicit additive_monoid(T val) : x(val) {}

    additive_monoid& operator *= (const additive_monoid& right) {
        x += right.x;
        return *this;
    }

    explicit operator T() const {
        return x;
    }
};

template <class T = unsigned>
class affine_monoid {
    T m_a = 1, m_b = 0;
public:
    affine_monoid() = default;
    affine_monoid(T a, T b) : m_a(a), m_b(b) {}
    affine_monoid& operator*=(const affine_monoid& r) {
        m_b = m_b + m_a * r.m_b;
        m_a = m_a * r.m_a;
        return *this;
    }

    T operator()(T x) const {
        return m_a * x + m_b;
    }
};

template <class T = unsigned>
class multiplicative_monoid {
    T x = T{ 1 };
public:
    multiplicative_monoid() = default;
    explicit multiplicative_monoid(T val) : x(val) {}

    multiplicative_monoid& operator *= (const multiplicative_monoid& right) {
        x *= right.x;
        return *this;
    }

    explicit operator T() const {
        return x;
    }
};

template <class T>
concept Monoid =
requires(T x) {
    T();
    { x *= x } -> std::same_as<T&>;
};

template <Monoid T>
T pow(T x, std::unsigned_integral auto e) noexcept {
    auto r = T();
    while (e > 0) {
        if ((e & 1) != 0)
            r *= x;
        x *= x;
        e >>= 1;
    }
    return r;
}

unsigned range(unsigned x, unsigned xmin, unsigned xmax) {
    return xmin + (x % (xmax - xmin));
}

void randomize(unsigned* V, size_t n, unsigned x0, unsigned xmin, unsigned xmax) {
    const unsigned A = 0x8088405;
    const unsigned b = 1;
    const unsigned long long c = 0x100000000;

    unsigned T = get_num_threads();
    if (T == 0) T = 1;

    std::vector<std::thread> workers(T - 1);
    std::mutex mtx;

    auto subvector = [T, V, n, x0, xmin, xmax, &mtx, A, b](unsigned t) {
        unsigned x;
        unsigned s_t = n / T;
        unsigned b_t = n % T;

        affine_monoid m(A, b);

        if (t < b_t)
            b_t = ++s_t * t;
        else
            b_t += s_t * t;

        unsigned e_t = b_t + s_t;

        affine_monoid my_m = pow(m, b_t);

        for (unsigned i = b_t; i < e_t; i++) {
            x = my_m(x0) % c;
            V[i] = range(x, xmin, xmax);
            my_m *= m;
        }

        {
            std::scoped_lock lock(mtx);
        }
    };

    for (size_t t = 1; t < T; ++t)
        workers[t - 1] = std::thread(subvector, t);
    subvector(0);

    for (auto& worker : workers)
        worker.join();
}


int main() {
    size_t n;
    unsigned x0, xmin, xmax;

    std::cout << "Enter number of numbers to generate: ";
    std::cin >> n;
    std::cout << "Enter generator seed: ";
    std::cin >> x0;
    std::cout << "Enter minimum value of range: ";
    std::cin >> xmin;
    std::cout << "Enter maximum value of range: ";
    std::cin >> xmax;

    std::vector<unsigned> V(n);

    randomize(V.data(), n, x0, xmin, xmax);

    for (size_t i = 0; i < n; ++i) {
        std::cout << V[i] << std::endl;
    }

    return 0;
};