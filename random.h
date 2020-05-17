#pragma once

#include <random>

class Random
{
    // static std::random_device rd;
    // std::mt19937 random_generator(rd());  // TODO: for debugging fix seed.
    std::mt19937 random_generator;

public:
    Random(const int seed = static_cast<int>(std::time(nullptr)))
        : random_generator(static_cast<std::mt19937::result_type>(seed))
    {
    }

    double normal(const double mean = 0.0, const double std = 1.0)
    {
        std::normal_distribution dist(mean, std);
        return dist(random_generator);
    }

    double uniform(const double low = 0.0, const double high = 1.0)
    {
        std::uniform_real_distribution dist(low, high);
        return dist(random_generator);
    }
};