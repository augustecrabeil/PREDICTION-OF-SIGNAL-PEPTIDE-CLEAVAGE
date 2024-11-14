#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <fstream>
#include "dataset_prot.hpp"
#include "confusion_matrix_prot.hpp"

#ifndef STATISTICAL_MODEL_HPP
#define STATISTICAL_MODEL_HPP

class Statistical_model {
    private:
    /**
	  f(a,i)
		*/
    std::vector<std::vector<double>> f;
    /**
	  g(a)
		*/
    std::vector<double> g;
   	/**
	  s(a,i)
		*/
    std::vector<std::vector<double>> s;

    int p;
    int q;
    int t;

    /**
    alpha for the "pseudo-counts"
    */
    double alpha = 0.00000001;
    
    
    public:
    /**
    The constructor 
		*/
    Statistical_model(Dataset_prot* dataset_prot, int _p, int _q, int _t);
    /**
    The destructor
    */
    ~Statistical_model();
    /** 
    the classification threshold
    */
    double threshold;
    
    int get_p();
    int get_q();
    std::vector<std::vector<double>> get_s();
    double get_threshold();
    void set_threshold_1(Dataset_prot* dataset_prot);
    void set_threshold_2(Dataset_prot* dataset_prot);
    void set_threshold_3(Dataset_prot* dataset_prot);
    void set_threshold_4(Dataset_prot* dataset_prot, double e);
    double score(const std::string& w);
    int best_place(const std::string& w);
    int binary_classifer(const std::string& w);
    ConfusionMatrix_prot test(Dataset_prot* dataset_prot);
    int s_csv();
};

#endif
