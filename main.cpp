#include <print>
#include <format>
#include <vector>
#include <cmath>
#include <openblas/cblas.h>
using namespace std;


static double sigmoid(const double x) {
	return 1.0 / (1.0 + exp(-x));
}


static double dot(const vector<double>& a, const vector<double>& b) {
	if (a.size() != b.size()) throw invalid_argument("Vectors must be the same size!");
	return cblas_ddot(static_cast<int>(a.size()), a.data(), 1, b.data(), 1);
}


class Neuron {
private:
	double bias;
	vector<double> weights;

public:
	Neuron(const vector<double> &weights, const double bias) {
		this->weights = weights;
		this->bias = bias;
	}

	double feedforward(const vector<double> &input) const {
		return sigmoid(dot(input, weights) + bias);
	}
};


int main() {
	Neuron n({0.0, 1.0}, 4.0);

	print("Neuron result: {}", n.feedforward({2.0, 3.0}));
}
