#include <string>
#include <vector>
#include <functional>
#include "dataset_prot.hpp"
#include "confusion_matrix_prot.hpp"
#include "kernel_prot.hpp"

#ifndef SVM_PROT_HPP
#define SVM_PROT_HPP

class SVM_prot {
    private:
        // training dataset
        int p;
        int q;
        int t;
        bool all;
        std::vector<std::vector<int>> train_prots;
        std::vector<int> train_labels;
        std::vector<std::vector<int>> intra_test_prots;
        std::vector<int> intra_test_labels;
        // kernel
        Kernel_prot kernel;
        std::vector<std::vector<double>> computed_kernel;
        // estimation result
        void compute_kernel();
        std::vector<double> alpha;
        double beta_0;
        // only consider support inside the margin by at least clipping_epsilon
        const double clipping_epsilon = 0.000000000000000000000001;
        // consider stopping gradient ascent when the derivative is smaller than stopping_criterion
        const double stopping_criterion = 0.001;

    public:
        // constructor
        SVM_prot() = delete;
        SVM_prot(Dataset_prot* dataset, int p, int q, int t, bool all, bool allTest, Kernel_prot K);
        // destructor
        ~SVM_prot();
        // only public for test purposes, should be private
        void compute_beta_0(double C=1.0);

        // getters - setters - for test purposes
        Kernel_prot get_kernel() const;
        std::vector<int> get_train_labels() const;
        std::vector<std::vector<int>> get_train_prots() const;
        std::vector<int> get_intra_test_labels() const;
        std::vector<std::vector<int>> get_intra_test_prots() const;
        std::vector<std::vector<double>> get_computed_kernel() const;
        std::vector<double> get_alphas() const;
        double get_beta_0() const;
        void set_alphas(std::vector<double> alpha);
        int get_p() const;
        int get_q() const;
        int get_t() const;
        bool get_all() const;

        // methods
        void train(const double C, const double lr);
        ConfusionMatrix_prot train_and_test_lib(const double C, const double lr, bool scaled);
        int f_hat(const std::vector<int> x);
        double f_hat_value(const std::vector<int> x);
        ConfusionMatrix_prot test(const Dataset_prot* test);
        double PredictCleavageSites(const Dataset_prot* test, int t);
        ConfusionMatrix_prot intra_test();
};

#endif