#include <iostream>
#include <numeric>
#include <cmath>
#include <vector>
#include <algorithm>


#include "kernel_prot.hpp"
#include "statistical_model.hpp"

// static inline double powi(double base, int times)
// {
// 	double tmp = base, ret = 1.0;

// 	for(int t=times; t>0; t/=2)
// 	{
// 		if(t%2==1) ret*=tmp;
// 		tmp = tmp * tmp;
// 	}
// 	return ret;
// }

Kernel_prot::Kernel_prot(const kernel_parameter& param, Statistical_model* _stat):
	kernel_type(param.kernel_type),
	degree(param.degree),
	gamma(param.gamma),
	coef0(param.coef0)
  {stat = _stat;}


double Kernel_prot::k(const std::vector<int> &x1, const std::vector<int> &x2) const {
	switch (kernel_type) {
		case LINEAR:
			return kernel_linear(x1, x2);
		case POLY:
			return kernel_poly(x1, x2);
		case RBF:
			return kernel_rbf(x1, x2);
		case SIGMOID:
			return kernel_sigmoid(x1, x2);
		case RATQUAD:
			return kernel_ratquad(x1, x2);
		case PSEUDO_LINEAR:
			return pseudo_kernel_linear(x1, x2);
		case PSEUDO_POLY:
			return pseudo_kernel_poly(x1, x2);
		case PSEUDO_RBF:
			return pseudo_kernel_rbf(x1, x2);
		case PSEUDO_SIGMOID:
			return pseudo_kernel_sigmoid(x1, x2);
		case PSEUDO_RATQUAD:
			return pseudo_kernel_ratquad(x1, x2);
        case PROBABILISTIC:
            return probabilistic_kernel(x1, x2);
            // return custom_product_kernel(x1, x2);
            // return custom_sum_kernel(x1, x2);
		default:
			std::cout << "Invalid kernel" << std::endl;
			return 0.0;
	}
};

double Kernel_prot::dot(const std::vector<int> &x1, const std::vector<int> &x2) {
	// there can be edge cases if they're not the same length
	return std::inner_product(x1.begin(), x1.end(), x2.begin(), 0.0);
};

double Kernel_prot::pseudo_dot(const std::vector<int> &x1, const std::vector<int> &x2) {
	double sum = 0.0;
    for (int i = 0; i < x1.size()/26; i++) {
        char a1;
        char a2;
        int aa1 = -1;
        int aa2 = -1;
        for (int j = 0; j < 26; j++) { // retransform the vecor of 0 and 1 of length 26 in a letter of the word
            if (x1[(26*i)+j] == 1) {
                a1 = 'A' + j;
            }
            if (x2[(26*i)+j] == 1) {
                a2 = 'A' + j;
            }
        }
        switch(a1) { // make the connection between the amino acid and the columns and lines of the BLOSUM62 matrix
            case 'A':
                aa1 = 0;
            case 'R':
                aa1 = 1;
            case 'N':
                aa1 = 2;
            case 'D':
                aa1 = 3;
            case 'C':
                aa1 = 4;
            case 'Q':
                aa1 = 5;
            case 'E':
                aa1 = 6;
            case 'G':
                aa1 = 7;
            case 'H':
                aa1 = 8;
            case 'I':
                aa1 = 9;
            case 'L':
                aa1 = 10;
            case 'K':
                aa1 = 11;
            case 'M':
                aa1 = 12;
            case 'F':
                aa1 = 13;
            case 'P':
                aa1 = 14;
            case 'S':
                aa1 = 15;
            case 'T':
                aa1 = 16;
            case 'W':
                aa1 = 17;
            case 'Y':
                aa1 = 18;
            case 'V':
                aa1 = 19;
        }
        switch(a2) {
            case 'A':
                aa2 = 0;
            case 'R':
                aa2 = 1;
            case 'N':
                aa2 = 2;
            case 'D':
                aa2 = 3;
            case 'C':
                aa2 = 4;
            case 'Q':
                aa2 = 5;
            case 'E':
                aa2 = 6;
            case 'G':
                aa2 = 7;
            case 'H':
                aa2 = 8;
            case 'I':
                aa2 = 9;
            case 'L':
                aa2 = 10;
            case 'K':
                aa2 = 11;
            case 'M':
                aa2 = 12;
            case 'F':
                aa2 = 13;
            case 'P':
                aa2 = 14;
            case 'S':
                aa2 = 15;
            case 'T':
                aa2 = 16;
            case 'W':
                aa2 = 17;
            case 'Y':
                aa2 = 18;
            case 'V':
                aa2 = 19;
        }
        if ((aa1 != -1) && (aa2 != -1)) {
            sum += M[aa1][aa2];
        }
    }
	return sum;
};

double Kernel_prot::kernel_linear(const std::vector<int> &x1, const std::vector<int> &x2) const {
	return dot(x1, x2);
};

double Kernel_prot::kernel_poly(const std::vector<int> &x1, const std::vector<int> &x2) const {
	double res = 0.0;
	res = pow(((gamma * dot(x1, x2)) + coef0), degree);
	return res;
};

double Kernel_prot::kernel_rbf(const std::vector<int> &x1, const std::vector<int> &x2) const {
	std::vector<int> x = std::vector<int>(x1.size());
	for (int i = 0; i < x1.size(); i++) {
		x[i] = x1[i] - x2[i];
	}
	return exp((-gamma) * dot(x, x));
};

double Kernel_prot::kernel_sigmoid(const std::vector<int> &x1, const std::vector<int> &x2) const {
	return tanh((gamma * dot(x1, x2)) + coef0);
};

double Kernel_prot::kernel_ratquad(const std::vector<int> &x1, const std::vector<int> &x2) const {
	std::vector<int> x = std::vector<int>(x1.size());
	for (int i = 0; i < x1.size(); i++) {
		x[i] = x1[i] - x2[i];
	}
	return (coef0 / (dot(x, x) + coef0));
};

double Kernel_prot::pseudo_kernel_linear(const std::vector<int> &x1, const std::vector<int> &x2) const {
	return pseudo_dot(x1, x2);
};

double Kernel_prot::pseudo_kernel_poly(const std::vector<int> &x1, const std::vector<int> &x2) const {
	double res = 0.0;
	res = pow(((gamma * pseudo_dot(x1, x2)) + coef0), degree);
	return res;
};

double Kernel_prot::pseudo_kernel_rbf(const std::vector<int> &x1, const std::vector<int> &x2) const {
	std::vector<int> x = std::vector<int>(x1.size());
	for (int i = 0; i < x1.size(); i++) {
		x[i] = x1[i] - x2[i];
	}
	return exp((-gamma) * pseudo_dot(x, x));
};

double Kernel_prot::pseudo_kernel_sigmoid(const std::vector<int> &x1, const std::vector<int> &x2) const {
	return tanh((gamma * pseudo_dot(x1, x2)) + coef0);
};

double Kernel_prot::pseudo_kernel_ratquad(const std::vector<int> &x1, const std::vector<int> &x2) const {
	std::vector<int> x = std::vector<int>(x1.size());
	for (int i = 0; i < x1.size(); i++) {
		x[i]= x1[i] - x2[i];
	}
	return (coef0 / (pseudo_dot(x, x) + coef0));
};

double Kernel_prot::probabilistic_kernel(const std::vector<int> &x1, const std::vector<int> &x2) const {
    std::vector<std::vector<double>> s = (*stat).get_s();
    double sum = 0;
    for (int i = 0; i < x1.size()/26; i++){
        int a1;
        int a2;
        for (int j = 0; j < 26; j++) {
            if (x1[(26*i)+j] == 1) {
                a1 = j;
            }
            if (x2[(26*i)+j] == 1) {
                a2 = j;
            }
        }
        if (a1==a2) {sum += s[a1][i] + log(1 + exp(s[a1][i]));}
        else {sum += s[a1][i] + s[a2][i];}
    }
    return exp(sum);
}

double Kernel_prot::custom_product_kernel(const std::vector<int> &x1, const std::vector<int> &x2) const {
    return probabilistic_kernel(x1, x2) * pseudo_kernel_rbf(x1, x2);
}

double Kernel_prot::custom_sum_kernel(const std::vector<int> &x1, const std::vector<int> &x2) const {
    return (0.5*probabilistic_kernel(x1, x2)) + (0.5*pseudo_kernel_rbf(x1, x2));
}

int Kernel_prot::get_kernel_type() const {
	return kernel_type;
}

double Kernel_prot::get_degree() const {
  return degree;
}
    
double Kernel_prot::get_gamma() const {
  return gamma;
}
    
double Kernel_prot::get_coef0() const {
  return coef0;
}
    