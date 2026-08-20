#include <iostream>
#include <numeric>
#include <vector>

int main(){
    const std::vector<int> values{1, 2, 3, 4};
    const int sum = std::accumulate(
        values.begin(), values.end(), 0);
    if (sum != 10) {
        std::cerr<<"self check failed\n";
        return 1;
    }
    std::cout<<"C++20 baseline OK\n";
    return 0;
}