#pragma once
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int ceilDiv(int a, int b) {
    return (a + b - 1) / b;
}
constexpr int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}
