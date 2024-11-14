#include "dataset_prot.hpp"
#include "statistical_model.hpp"

#ifndef KERNEL_PROT_HPP
#define KERNEL_PROT_HPP

/**
  All SVM parameters.
*/
struct kernel_parameter
{
	int kernel_type;
	int degree;	 /* for poly */
	double gamma;  /* for poly/rbf/sigmoid */
	double coef0;  /* for poly/sigmoid */
};

/**
  Kernel types.
*/
enum {LINEAR, POLY, RBF, SIGMOID, RATQUAD, PSEUDO_LINEAR, PSEUDO_POLY, PSEUDO_RBF, PSEUDO_SIGMOID, PSEUDO_RATQUAD, PROBABILISTIC};

/**
  BLOSUM62 matrix
*/
const int M[24][24] = {{4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1,  1, 0, -3, -2, 0, -2, -1, 0, -4}, 
            {-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1, -4}, 
            {-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1, -4}, 
            {-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1, -4}, 
            {0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4}, 
            {-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1, -4}, 
            {-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4}, 
            {0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1, -4}, 
            {-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1, -4}, 
            {-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1, -4}, 
            {-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3, -1, -4}, 
            {-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1, -4}, 
            {-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1, -4}, 
            {-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3, -1, -4}, 
            {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -1, -2, -4}, 
            {1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0, -4}, 
            {0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0, -4}, 
            {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2, -4}, 
            {-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -2, -1, -4}, 
            {0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, -2, -1, -4}, 
            {-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1, -4}, 
            {-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4}, 
            {0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1, -4}, 
            {-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1}};

/**
  The Kernel class defines the kernel type, its parameters, and computes the kernel.
*/
class Kernel_prot {
    public:
		/**
		 The constructor needs:
		@param kernel_parameter the kernel parameters
		*/
	    Kernel_prot(const kernel_parameter& param, Statistical_model* _stat);
		/**
		 The kernel function.
		*/
    double k(const std::vector<int> &x1, const std::vector<int> &x2) const;
		int get_kernel_type() const;
    double get_degree() const;
    double get_coef0() const;
    double get_gamma() const;


    private:
      /**
        The kernel parameters
      */
	    const int kernel_type;
	    const int degree;
	    const double gamma;
	    const double coef0;
      Statistical_model* stat;
      
      /**
        The calculation methods
      */
	    static double dot(const std::vector<int> &x1, const std::vector<int> &x2);
	    static double pseudo_dot(const std::vector<int> &x1, const std::vector<int> &x2);
	    double kernel_linear(const std::vector<int> &x1, const std::vector<int> &x2) const;
	    double kernel_poly(const std::vector<int> &x1, const std::vector<int> &x2) const;
	    double kernel_rbf(const std::vector<int> &x1, const std::vector<int> &x2) const;
	    double kernel_sigmoid(const std::vector<int> &x1, const std::vector<int> &x2) const;
	    double kernel_ratquad(const std::vector<int> &x1, const std::vector<int> &x2) const;
	    double pseudo_kernel_linear(const std::vector<int> &x1, const std::vector<int> &x2) const;
	    double pseudo_kernel_poly(const std::vector<int> &x1, const std::vector<int> &x2) const;
	    double pseudo_kernel_rbf(const std::vector<int> &x1, const std::vector<int> &x2) const;
	    double pseudo_kernel_sigmoid(const std::vector<int> &x1, const std::vector<int> &x2) const;
	    double pseudo_kernel_ratquad(const std::vector<int> &x1, const std::vector<int> &x2) const;
      double probabilistic_kernel(const std::vector<int> &x1, const std::vector<int> &x2) const;
      double custom_product_kernel(const std::vector<int> &x1, const std::vector<int> &x2) const;
      double custom_sum_kernel(const std::vector<int> &x1, const std::vector<int> &x2) const;
};

#endif
