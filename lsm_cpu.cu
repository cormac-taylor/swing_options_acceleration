#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include "lsm_cpu.h"

using namespace std;

const int NUM_PATHS = 100000;
const int NUM_TIMESTEPS = 365;
const float STRIKE = 100.0f;
const float RISKFREE = 0.05f;
const float VOLATILITY = 0.2f;
const float S0 = 100.0f;
const int MAX_EXERCISE = 5;
const float DT = 1.0f / NUM_TIMESTEPS;

// GBM
vector<PathData> simulate_paths_lsm_cpu() {
    vector<PathData> paths(NUM_PATHS);
    mt19937 rng(1234);
    normal_distribution<float> norm(0.0f, 1.0f);

    for (int idx = 0; idx < NUM_PATHS; ++idx) {
        paths[idx].prices.resize(NUM_TIMESTEPS);
        float s = S0;
        for (int t = 0; t < NUM_TIMESTEPS; ++t) {
            float dW = norm(rng) * sqrtf(DT);
            s *= expf((RISKFREE - 0.5f * VOLATILITY * VOLATILITY) * DT + VOLATILITY * dW);
            paths[idx].prices[t] = s;
        }
    }
    return paths;
}

bool solve_3x3_lsm_cpu(const float A[3][3], const float B[3], float X[3]) {
    float det = A[0][0] * (A[1][1]*A[2][2] - A[1][2]*A[2][1])
              - A[0][1] * (A[1][0]*A[2][2] - A[1][2]*A[2][0])
              + A[0][2] * (A[1][0]*A[2][1] - A[1][1]*A[2][0]);

    if (fabs(det) < 1e-7) return false;

    float invDet = 1.0f / det;
    float invA[3][3];

    invA[0][0] = (A[1][1]*A[2][2] - A[1][2]*A[2][1]) * invDet;
    invA[0][1] = (A[0][2]*A[2][1] - A[0][1]*A[2][2]) * invDet;
    invA[0][2] = (A[0][1]*A[1][2] - A[0][2]*A[1][1]) * invDet;
    invA[1][0] = (A[1][2]*A[2][0] - A[1][0]*A[2][2]) * invDet;
    invA[1][1] = (A[0][0]*A[2][2] - A[0][2]*A[2][0]) * invDet;
    invA[1][2] = (A[0][2]*A[1][0] - A[0][0]*A[1][2]) * invDet;
    invA[2][0] = (A[1][0]*A[2][1] - A[1][1]*A[2][0]) * invDet;
    invA[2][1] = (A[0][1]*A[2][0] - A[0][0]*A[2][1]) * invDet;
    invA[2][2] = (A[0][0]*A[1][1] - A[0][1]*A[1][0]) * invDet;

    X[0] = invA[0][0]*B[0] + invA[0][1]*B[1] + invA[0][2]*B[2];
    X[1] = invA[1][0]*B[0] + invA[1][1]*B[1] + invA[1][2]*B[2];
    X[2] = invA[2][0]*B[0] + invA[2][1]*B[1] + invA[2][2]*B[2];
    return true;
}

extern "C" void lsm_cpu() {
    vector<PathData> paths = simulate_paths_lsm_cpu();
    const float discount = expf(-RISKFREE * DT);

    // init final cashflows
    for (auto& path : paths) {
        path.cashflow = max(path.prices.back() - STRIKE, 0.0f);
        path.remaining = MAX_EXERCISE;
    }

    // backward induction
    for (int t = NUM_TIMESTEPS-2; t >= 0; --t) {
        vector<int> itm_indices;
        for (int i = 0; i < NUM_PATHS; ++i) {
            if (paths[i].prices[t] > STRIKE + 1e-5)
                itm_indices.push_back(i);
        }

        if (itm_indices.empty()) {
            for (auto& path : paths)
                path.cashflow *= discount;
            continue;
        }

        // regression
        float XtX[3][3] = {0};
        float XtY[3] = {0};
        for (int i : itm_indices) {
            float S = paths[i].prices[t];
            float X[] = {1.0f, S, S*S};
            float Y = paths[i].cashflow * discount;

            for (int row = 0; row < 3; ++row) {
                XtY[row] += X[row] * Y;
                for (int col = 0; col < 3; ++col)
                    XtX[row][col] += X[row] * X[col];
            }
        }

        // coefficients
        float coeff[3];
        if (!solve_3x3_lsm_cpu(XtX, XtY, coeff)) {
            for (auto& path : paths)
                path.cashflow *= discount;
            continue;
        }

        // update cashflows
        for (auto& path : paths) {
            float S = path.prices[t];
            float immediate = max(S - STRIKE, 0.0f);

            if (immediate > 1e-5) {
                float continuation = coeff[0] + coeff[1]*S + coeff[2]*S*S;
                if (immediate > continuation && path.remaining > 0) {
                    path.cashflow = immediate;
                    path.remaining--;
                } else {
                    path.cashflow *= discount;
                }
            } else {
                path.cashflow *= discount;
            }
        }
    }

    // average price
    float sum = 0;
    for (const auto& path : paths)
        sum += path.cashflow;
    cout << "Swing Option Price: " << sum / NUM_PATHS << endl;
}