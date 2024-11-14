#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <cstdlib>

#include "kernel_prot.hpp"
#include "confusion_matrix_prot.hpp"
#include "svm_prot.hpp"

SVM_prot::SVM_prot(Dataset_prot* dataset, int p, int q, int t, bool all, bool allTest, Kernel_prot K):
    p(p), q(q), t(t), all(all), kernel(K) {
    // Collect intra_test_prots and intra_test_labels for the future test using the r-fold repartition
    if (allTest) {
        for (int i = (dataset->GetNbrSamples()/dataset->GetR())*t; i < (dataset->GetNbrSamples()/dataset->GetR())*(t+1); i++) {
            for (int ip = p; ip < dataset->GetNAA(dataset->GetRFold()[i])-q; ip++) { // ip is the index of the cleavage localization
                if (ip == dataset->GetCleavLoc(dataset->GetRFold()[i])) {
                    intra_test_labels.push_back(1);
                }
                else {
                    intra_test_labels.push_back(0);
                }
                std::vector<int> instance(26*(p+q));
                for (int j = 0; j < p+q; j++) {
                    // creation of the vector of length 26 representing the jth letter of the protein
                    for (int m = 0; m < 26; m++) {
                        if (dataset->GetInstance(dataset->GetRFold()[i])[ip-p+j] - 'A' == m) {
                            instance[(26*j)+m] = 1;
                        }
                        else {
                            instance[(26*j)+m] = 0;
                        }
                    }
                }
                intra_test_prots.push_back(instance);
            }
        }
    }
    else {
        for (int i = (dataset->GetNbrSamples()/dataset->GetR())*t; i < (dataset->GetNbrSamples()/dataset->GetR())*(t+1); i++) {
            std::vector<int> others;
            for (int ip = p; ip < dataset->GetNAA(dataset->GetRFold()[i])-q; ip++) { // ip is the index of the cleavage localization
                if (ip == dataset->GetCleavLoc(dataset->GetRFold()[i])) {
                    intra_test_labels.push_back(1);
                    std::vector<int> instance(26*(p+q));
                    for (int j = 0; j < p+q; j++) {
                        // creation of the vector of length 26 representing the jth letter of the protein
                        for (int m = 0; m < 26; m++) {
                            if (dataset->GetInstance(dataset->GetRFold()[i])[ip-p+j] - 'A' == m) {
                                instance[(26*j)+m] = 1;
                            }
                            else {
                                instance[(26*j)+m] = 0;
                            }
                        }
                    }
                    intra_test_prots.push_back(instance);

                }
                else {
                    others.push_back(ip);
                }
            }
            if(others.size() > 0) {
                std::random_shuffle(others.begin(), others.end());
                intra_test_labels.push_back(0);
                std::vector<int> instance(26*(p+q));
                for (int j = 0; j < p+q; j++) {
                    for (int m = 0; m < 26; m++) {
                        if (dataset->GetInstance(dataset->GetRFold()[i])[others[0]-p+j] - 'A' == m) {
                            instance[(26*j)+m] = 1;
                        }
                        else {
                            instance[(26*j)+m] = 0;
                        }
                    }
                }
                intra_test_prots.push_back(instance);
            }
        }
    }

    // Collect train_prots and train_labels for the future test using the r-fold repartition
    if (all) { // for each protein, we select every possibility of cleavage localization
        for (int i = 0; i < (dataset->GetNbrSamples()/dataset->GetR())*t; i++) {
            for (int ip = p; ip < dataset->GetNAA(dataset->GetRFold()[i])-q; ip++) {
                if (ip == dataset->GetCleavLoc(dataset->GetRFold()[i])) {
                    train_labels.push_back(1);
                }
                else {
                    train_labels.push_back(-1);
                }
                std::vector<int> instance(26*(p+q));
                for (int j = 0; j < p+q; j++) {
                    for (int m = 0; m < 26; m++) {
                        if (dataset->GetInstance(dataset->GetRFold()[i])[ip-p+j] - 'A' == m) {
                            instance[(26*j)+m] = 1;
                        }
                        else {
                            instance[(26*j)+m] = 0;
                        }
                    }
                }
                train_prots.push_back(instance);
            }
        }
        for (int i = (dataset->GetNbrSamples()/dataset->GetR())*(t+1); i < dataset->GetNbrSamples(); i++) {
            for (int ip = p; ip < dataset->GetNAA(dataset->GetRFold()[i])-q; ip++) {
                if (ip == dataset->GetCleavLoc(dataset->GetRFold()[i])) {
                    train_labels.push_back(1);
                }
                else {
                    train_labels.push_back(-1);
                }
                std::vector<int> instance(26*(p+q));
                for (int j = 0; j < p+q; j++) {
                    for (int m = 0; m < 26; m++) {
                        if (dataset->GetInstance(dataset->GetRFold()[i])[ip-p+j] - 'A' == m) {
                            instance[(26*j)+m] = 1;
                        }
                        else {
                            instance[(26*j)+m] = 0;
                        }
                    }
                }
                train_prots.push_back(instance);
            }
        }
    }
    else { // for each protein, we select at random one possibility of cleavage localization in addition to the correct one
        for (int i = 0; i < (dataset->GetNbrSamples()/dataset->GetR())*t; i++) {
            std::vector<int> others;
            for (int ip = p; ip < dataset->GetNAA(dataset->GetRFold()[i])-q; ip++) {
                if (ip == dataset->GetCleavLoc(dataset->GetRFold()[i])) {
                    train_labels.push_back(1);
                    std::vector<int> instance(26*(p+q));
                    for (int j = 0; j < p+q; j++) {
                        for (int m = 0; m < 26; m++) {
                            if (dataset->GetInstance(dataset->GetRFold()[i])[ip-p+j] - 'A' == m) {
                                instance[(26*j)+m] = 1;
                            }
                            else {
                                instance[(26*j)+m] = 0;
                            }
                        }
                    }
                    train_prots.push_back(instance);
                }
                else {
                    others.push_back(ip);
                }
            }
            if (others.size() > 0) {
                std::random_shuffle(others.begin(), others.end());
                train_labels.push_back(-1);
                std::vector<int> instance(26*(p+q));
                for (int j = 0; j < p+q; j++) {
                    for (int m = 0; m < 26; m++) {
                        if (dataset->GetInstance(dataset->GetRFold()[i])[others[0]-p+j] - 'A' == m) {
                            instance[(26*j)+m] = 1;
                        }
                        else {
                            instance[(26*j)+m] = 0;
                        }
                    }
                }
                train_prots.push_back(instance);
            }
        }
        for (int i = (dataset->GetNbrSamples()/dataset->GetR())*(t+1); i < dataset->GetNbrSamples(); i++) {
            std::vector<int> others;
            for (int ip = p; ip < dataset->GetNAA(dataset->GetRFold()[i])-q; ip++) {
                if (ip == dataset->GetCleavLoc(dataset->GetRFold()[i])) {
                    train_labels.push_back(1);
                    std::vector<int> instance(26*(p+q));
                    for (int j = 0; j < p+q; j++) {
                        for (int m = 0; m < 26; m++) {
                            if (dataset->GetInstance(dataset->GetRFold()[i])[ip-p+j] - 'A' == m) {
                                instance[(26*j)+m] = 1;
                            }
                            else {
                                instance[(26*j)+m] = 0;
                            }
                        }
                    }
                    train_prots.push_back(instance);
                }
                else {
                    others.push_back(ip);
                }
            }
            if(others.size() > 0) {
                std::random_shuffle(others.begin(), others.end());
                train_labels.push_back(-1);
                std::vector<int> instance(26*(p+q));
                for (int j = 0; j < p+q; j++) {
                    for (int m = 0; m < 26; m++) {
                        if (dataset->GetInstance(dataset->GetRFold()[i])[others[0]-p+j] - 'A' == m) {
                            instance[(26*j)+m] = 1;
                        }
                        else {
                            instance[(26*j)+m] = 0;
                        }
                    }
                }
                train_prots.push_back(instance);
            }
        }
    }
    compute_kernel();
}

SVM_prot::~SVM_prot() {
}

void SVM_prot::compute_kernel() {
    const int n = train_prots.size();
    alpha = std::vector<double>(n);
    computed_kernel = std::vector<std::vector<double> >(n, std::vector<double>(n));

    // put y_i y_j k(x_i, x_j) in the (i, j)th coordinate of computed_kernel
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            computed_kernel[i][j] = train_labels[i] * train_labels[j] * kernel.k(train_prots[i], train_prots[j]);
        }
    }
}

void SVM_prot::compute_beta_0(double C) {
    // count keeps track of the number of support vectors (denoted by n_s)
    int count = 0;
    beta_0 = 0.0;
    // Use clipping_epsilon < alpha < C - clipping_epsilon instead of 0 < alpha < C
    for (int s = 0; s < train_prots.size(); s++) {
        if ((clipping_epsilon < alpha[s]) && (alpha[s] < C - clipping_epsilon)) {
            count ++;
            for (int i = 0; i < train_prots.size(); i++) {
                beta_0 += (alpha[i] * train_labels[i] * kernel.k(train_prots[i], train_prots[s]));
            }
            beta_0 -= train_labels[s];
        }
    }
    // This performs 1/n_s
    beta_0 /= count;
}

void SVM_prot::train(const double C, const double lr) {
    // Perform projected gradient ascent
    // While some alpha is not clipped AND its gradient is above stopping_criterion
    // (1) Set stop = false
    // (2) While not stop do
    // (2.1) Set stop = true
    // (2.2) For i = 1 to n do
    // (2.2.1) Compute the gradient - HINT: make good use of computed_kernel
    // (2.2.2) Move alpha in the direction of the gradient - eta corresponds to lr (for learning rate)
    // (2.2.3) Project alpha in the constraint box by clipping it
    // (2.2.4) Adjust stop if necessary
    // (3) Compute beta_0
    bool stop = false;
    while (!stop) {
        stop = true;
        for (int i = 0; i < train_prots.size(); i++) {
            double sum = 0.0;
            for (int j = 0; j < train_prots.size(); j++) {
                sum += alpha[j] * computed_kernel[i][j];
            }
            double gradi = 1 - sum;
            alpha[i] += lr * gradi;
            if (alpha[i] > C) {
                alpha[i] = C;
            }
            else if (alpha[i] < 0) {
                alpha[i] = 0;
            }
            else {
                if (std::abs(gradi) > stopping_criterion) {
                    stop = false;
                }
            }
        }
    }

    // Update beta_0
    compute_beta_0(C);
}

// With the python library
ConfusionMatrix_prot SVM_prot::train_and_test_lib(const double C, const double lr, bool scaled) {
    ConfusionMatrix_prot cm;
    // file pointer
    std::fstream fouttrain;
    std::fstream fouttest;
  
    // create new files
    fouttrain.open("train_prots.csv", std::ios::out | std::ios::app);
    fouttest.open("test_prots.csv", std::ios::out | std::ios::app);

    // Insert the data to files
    for (int i = 0; i < train_prots.size(); i++) {
        int lab = (train_labels[i] == 1) ? 1 : 0; // retransform the -1 in 0 for the python library
        fouttrain << i+1 << ";" << lab << ";";
        for (int j = 0; j < 26*(p+q); j++) {
            if (j == (26*(p+q))-1) {
                fouttrain << train_prots[i][j];
            }
            else {
                fouttrain << train_prots[i][j] << ";";
            }
        }
        fouttrain << std::endl;
    }
    for (int i = 0; i < intra_test_prots.size(); i++) {
        fouttest << i+1 << ";" << intra_test_labels[i] << ";";
        for (int j = 0; j < 26*(p+q); j++) {
            if (j == (26*(p+q))-1) {
                fouttest << intra_test_prots[i][j];
            }
            else {
                fouttest << intra_test_prots[i][j] << ";";
            }
        }
        fouttest << std::endl;
    }
    std::string kernType[11] = {"linear", "poly", "rbf", "sigmoid", "ratquad", "pseudo_linear", "pseudo_poly", "pseudo_rbf", "pseudo_sigmoid", "pseudo_ratquad", "probabilistic"};
    std::string kern = kernType[kernel.get_kernel_type()];
    // execute the svm calculations with the python script
    std::ostringstream oss;
    oss << "python3 svm_with_sklearn.py train_prots.csv test_prots.csv " << kern << " " << kernel.get_degree() << " " << kernel.get_gamma() << " " << kernel.get_coef0() << " " << C << " " << lr << " " << scaled;
    std::string var = oss.str();
    std::system(var.c_str());
    
    // File pointer
    std::fstream fin;
  
    // Open an existing file
    if (scaled) {
        fin.open("outputScaled.csv", std::ios::in);
    }
    else {
        fin.open("output.csv", std::ios::in);
    }
  
    // Read the Data from the file as String Vector
    std::vector<int> preds = std::vector<int>();
    std::string line, word;
  
    // read an entire row and store it in a string variable 'line'
    getline(fin, line);
  
    // used for breaking words
    std::stringstream s(line);
  
    // read every column data of the row and store it in a string variable, 'word'
    while (getline(s, word, ',')) {
        // add all the column data of a row to a vector
        preds.push_back(word[0] - '0');
    }

    for (int i = 0; i < intra_test_prots.size(); i++) {
        cm.AddPrediction(intra_test_labels[i], preds[i]);
    }
    if (scaled) {
        remove("outputScaled.csv");
    }
    else {
        remove("output.csv");
    }
    remove("train_prots.csv");
    remove("test_prots.csv");

    return cm;
}

int SVM_prot::f_hat(const std::vector<int> x) {
    // the classifier
    double res = 0.0;
    for (int i = 0; i < train_prots.size(); i++) {
        res += (alpha[i] * train_labels[i] * kernel.k(train_prots[i], x));
    }
    res -= beta_0;
    if (res >= 0) {
        return 1;
    }
    return -1;
}

double SVM_prot::f_hat_value(const std::vector<int> x) {
    // the "value" of the classifier instead of the sign
    double res = 0.0;
    for (int i = 0; i < train_prots.size(); i++) {
        res += (alpha[i] * train_labels[i] * kernel.k(train_prots[i], x));
    }
    res -= beta_0;
    return res;
}

ConfusionMatrix_prot SVM_prot::intra_test() {
    // the prediction for the test set of the cross-validation
    ConfusionMatrix_prot intra_cm;
    // Use f_hat to predict and put into the confusion matrix
    for (int i = 0; i < intra_test_prots.size(); i++) {
        int pred = (f_hat(intra_test_prots[i]) == 1) ? 1 : 0;
        intra_cm.AddPrediction(intra_test_labels[i], pred);
    }

    return intra_cm;
}

ConfusionMatrix_prot SVM_prot::test(const Dataset_prot* test) {
    // the prediction for another data set
    // Collect test_prots and test_labels and compute confusion matrix
    std::vector<int> test_labels;
    std::vector<std::vector<int>> test_prots;
    ConfusionMatrix_prot cm;

    // Put test dataset in features and labels
    for (int i = 0; i < test->GetNbrSamples(); i++) {
        for (int ip = p; ip < test->GetNAA(i)-q-1; ip++) {
            if (ip == test->GetCleavLoc(i)) {
                test_labels.push_back(1);
            }
            else {
                test_labels.push_back(0);
            }
            std::vector<int> instance(26*(p+q));
            for (int j = 0; j < p+q; j++) {
                for (int m = 0; m < 26; m++) {
                    if (test->GetInstance(i)[ip-p+j] - 'A' == m) {
                        instance[(26*j)+m] = 1;
                    }
                    else {
                        instance[(26*j)+m] = 0;
                    }
                }
            }
            test_prots.push_back(instance);
        }
    }

    // Use f_hat to predict and put into the confusion matrix
    for (int i = 0; i < test_prots.size(); i++) {
        int pred = (f_hat(test_prots[i]) == 1) ? 1 : 0;
        cm.AddPrediction(test_labels[i], pred);
    }

    return cm;
}

double SVM_prot::PredictCleavageSites(const Dataset_prot* test, int t) {
    // the best cleavage site possible for each protein of the 1/rth of the dataset
    // Collect test_prots and test_labels
    std::vector<int> realCleavSite;
    std::vector<std::vector<std::vector<int>>> test_prots;

    // Put the 1/rth of the test dataset in features and labels
    for (int i = 0; i < (test->GetNbrSamples()/test->GetR()); i++) {
        realCleavSite.push_back(test->GetCleavLoc(test->GetRFold()[i]));
        std::vector<std::vector<int>> vectors_prot;
        for (int ip = p; ip < test->GetNAA(i)-q-1; ip++) {
            std::vector<int> instance(26*(p+q));
            for (int j = 0; j < p+q; j++) {
                for (int m = 0; m < 26; m++) {
                    if (test->GetInstance(test->GetRFold()[i])[ip-p+j] - 'A' == m) {
                        instance[(26*j)+m] = 1;
                    }
                    else {
                        instance[(26*j)+m] = 0;
                    }
                }
            }
            vectors_prot.push_back(instance); // vectors_prot contains every possibility of cleavage localization for a protein
        }
        test_prots.push_back(vectors_prot);
    }

    // Use f_hat to predict and put into the confusion matrix
    std::vector<int> predCleavSite;
    for (int i = 0; i < test_prots.size(); i++) {
        double BestPred = 0.0;
        int site = 0;
        for (int j = 0; j < test_prots[i].size(); j++) { // select the most likely cleavage localization
            double pred = f_hat_value(test_prots[i][j]);
            if (pred > BestPred) {
                BestPred = pred;
                site = p + j;
            }
        }
        predCleavSite.push_back(site);
    }

    // calculation of the success rate
    double res = 0.0;
    int n = predCleavSite.size();
    for (int i = 0; i < n; i++) {
        if (predCleavSite[i] == realCleavSite[i]) {
            res += 1.0;
        }
    }
    res = double(res / double(n));

    return res;
}

Kernel_prot SVM_prot::get_kernel() const {
    return kernel;
}

std::vector<int> SVM_prot::get_train_labels() const {
    return train_labels;
}

std::vector<std::vector<int>> SVM_prot::get_train_prots() const {
    return train_prots;
}

std::vector<int> SVM_prot::get_intra_test_labels() const {
    return intra_test_labels;
}

std::vector<std::vector<int>> SVM_prot::get_intra_test_prots() const {
    return intra_test_prots;
}

std::vector<std::vector<double>> SVM_prot::get_computed_kernel() const {
    return computed_kernel;
}

std::vector<double> SVM_prot::get_alphas() const {
    return alpha;
}

double SVM_prot::get_beta_0() const {
    return beta_0;
}

int SVM_prot::get_p() const {
    return p;
}

int SVM_prot::get_q() const {
    return q;
}

int SVM_prot::get_t() const {
    return t;
}

bool SVM_prot::get_all() const {
    return all;
}

void SVM_prot::set_alphas(std::vector<double> alpha) {
    this->alpha = alpha;
}