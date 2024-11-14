#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdarg>
#include <iterator>
#include <string>
#include <regex>
#include <numeric>
#include <cmath>
#include <fstream>
#include <random>
#include <limits>
#include <array>
#include <random>
#include <algorithm>
#include <vector>

#include "../gradinglib/gradinglib.hpp"
#include "../dataset_prot.hpp"
#include "../kernel_prot.hpp"
#include "../confusion_matrix_prot.hpp"
#include "../svm_prot.hpp"

namespace tdgrading {

using namespace testlib;
using namespace std;


//test of the statistical model
int ex1_statistical_test(std::ostream &out, const std::string test_name) 
{
    std::string entity_name = "[Ex1]_StatisticalTest";
    start_test_suite(out, test_name);
    //To begin, we define all the variables: r, p, q, the statistical_model, and so on.
    const char* train_file = "data/EUKSIG_13.red";
    int r = 10;
    Dataset_prot train_dataset(train_file, r);
    int p = 13;
    int q = 2;
    std::cout << "We will apply on the dataset the " << r << "-fold method." << std::endl;
    //Then, we define all the variables that will allow us to evaluate our results.
    ConfusionMatrix_prot cm_tot = ConfusionMatrix_prot();
    double avg_error_rate = 0.0;
    double avg_false_alarm_rate = 0.0;
    double avg_detection_rate	= 0.0;
    double avg_f_score = 0.0;
    double avg_precision = 0.0;
    //for all the segments of the r-fold method
    for (int t = 0; t < r; t++) {
      //we create our statistical model i.e. s(a,i)
      Statistical_model stat(&train_dataset,p,q,t);
      //we define the threshold
      stat.set_threshold_4(&train_dataset,0.5);
      //We retrieve the results
      ConfusionMatrix_prot cm = stat.test(&train_dataset);
      cm_tot.AddMatrix(cm);
      avg_error_rate += cm.error_rate();
      avg_false_alarm_rate += cm.false_alarm_rate();
      avg_detection_rate += cm.detection_rate();
      avg_f_score += cm.f_score();
      avg_precision += cm.precision();
      std::cout << "The confusion matrix number " << t+1 << " was calculated..." << std::endl;
    }
    std::cout << std::endl;
    avg_error_rate /= r;
    avg_false_alarm_rate /= r;
    avg_detection_rate /= r;
    avg_f_score /= r;
    avg_precision /= r;
    std::cout << "Here are some average score results with the " << r << "-fold method:" << std::endl;
    std::cout << "Average error rate = " << avg_error_rate << std::endl;
    std::cout << "Average false alarm rate = " << avg_false_alarm_rate << std::endl;
    std::cout << "Average detection rate = " << avg_detection_rate << std::endl;
    std::cout << "Average f-score = " << avg_f_score << std::endl;
    std::cout << "Average precision = " << avg_precision << std::endl;
    std::cout << std::endl;
    std::cout << "Here is the confusion matrix containing all the dataset but with each subset classified with a classifier using the rest of the dataset:" << std::endl;
    cm_tot.PrintEvaluation();
    return 0;
}


//test of the svm
int ex2_intra_test(std::ostream &out, const std::string test_name) 
{
    std::string entity_name = "[Ex2]_IntraTest";
    start_test_suite(out, test_name);
    //To begin, we define all the variables: r, p, q, C, the kernel, and so on.
    const char* train_file = "data/EUKSIG_13.red";
    int r = 10;
    int p = 13;
    int q = 2;
    Dataset_prot train_dataset(train_file, r);
    Statistical_model stat(&train_dataset, p, q, r);
    int degree = 2; // the degree because it is the more common and larger tend to overfit
    double n = double(double(2*(r-1)*train_dataset.GetNbrSamples()) / double(r)); // the number of training samples (2 because for each protein, we take 2 words)
    double gamma = double(1.0 / n);
    double coef0 = 1.0; // a arbitrary choice because it is common
    int kern = 0;
    //choice of the kernel
    std::string kernType[11] = {"LINEAR", "POLY", "RBF", "SIGMOID", "RATQUAD", "PSEUDO_LINEAR", "PSEUDO_POLY", "PSEUDO_RBF", "PSEUDO_SIGMOID", "PSEUDO_RATQUAD", "PROBABILISTIC"};
    std::cout << "The kernel is " << kernType[kern] << " and the coefficients are degree = " << degree << ", gamma = " << gamma << " and coef0 = " << coef0 << std::endl;
    Kernel_prot kernel({kern, degree, gamma, coef0}, &stat);
    std::cout << std::endl;
    std::cout << "We will apply on the dataset the " << r << "-fold method." << std::endl;
    bool all = false;
    bool allTest = true;
    if (!all) {
      std::cout << "We chose to train our classifier with a balanced dataset between the correct and incorrect cleavage sites. All the possibilities are not used to train the classifier." << std::endl;
    }
    else {
      std::cout << "We chose to train our classifier with an unbalanced dataset between the correct and incorrect cleavage sites. All the possibilities are used to train the classifier." << std::endl;
    }
    std::cout << std::endl;
    double C = 1.0;
    double lr = 0.01;
    std::cout << "The values for the training are C = " << C << " and learning constant = " << lr << std::endl;
    
    //Then, we define all the variables that will allow us to evaluate our results.
    ConfusionMatrix_prot cm_tot = ConfusionMatrix_prot();
    double avg_error_rate = 0.0;
    double avg_false_alarm_rate = 0.0;
    double avg_detection_rate	= 0.0;
    double avg_f_score = 0.0;
    double avg_precision = 0.0;
    //for all the segments of the r-fold method
    for (int i = 0; i < r; i++) {
      //We create our svm
      SVM_prot svm(&train_dataset, p, q, i, all, allTest, kernel);
      //We train it
      svm.train(C, lr);
      //We retrieve the results
      ConfusionMatrix_prot cm = svm.intra_test();
      cm_tot.AddMatrix(cm);
      avg_error_rate += cm.error_rate();
      avg_false_alarm_rate += cm.false_alarm_rate();
      avg_detection_rate += cm.detection_rate();
      avg_f_score += cm.f_score();
      avg_precision += cm.precision();
      std::cout << "The confusion matrix number " << i+1 << " was calculated..." << std::endl;
    }
    std::cout << std::endl;
    avg_error_rate /= r;
    avg_false_alarm_rate /= r;
    avg_detection_rate /= r;
    avg_f_score /= r;
    avg_precision /= r;
    std::cout << "Here are some average score results with the " << r << "-fold method:" << std::endl;
    std::cout << "Average error rate = " << avg_error_rate << std::endl;
    std::cout << "Average false alarm rate = " << avg_false_alarm_rate << std::endl;
    std::cout << "Average detection rate = " << avg_detection_rate << std::endl;
    std::cout << "Average f-score = " << avg_f_score << std::endl;
    std::cout << "Average precision = " << avg_precision << std::endl;
    std::cout << std::endl;
    std::cout << "Here is the confusion matrix containing all the dataset but with each subset classified with a classifier using the rest of the dataset:" << std::endl;
    cm_tot.PrintEvaluation();
    return 0;
}


// SVM method from the library
int ex3_intra_test_lib(std::ostream &out, const std::string test_name) 
{
    std::string entity_name = "[Ex3]_IntraTestWithLibrary";
    start_test_suite(out, test_name);
    //To begin, we define all the variables: r, p, q, C, the kernel, and so on.
    const char* train_file = "data/EUKSIG_13.red";
    int r = 10;
    int p = 13;
    int q = 2;
    Dataset_prot train_dataset(train_file, r);
    Statistical_model stat(&train_dataset, p, q, r);
    int degree = 2; // the degree because it is the more common and larger tend to overfit
    double n = double(double(2*(r-1)*train_dataset.GetNbrSamples()) / double(r)); // the number of training samples (2 because for each protein, we take 2 words)
    double gamma = double(1.0 / n);
    double coef0 = 1.0; // a arbitrary choice because it is common
    int kern = 0;
    //choice of the kernel
    std::string kernType[11] = {"LINEAR", "POLY", "RBF", "SIGMOID", "RATQUAD", "PSEUDO_LINEAR", "PSEUDO_POLY", "PSEUDO_RBF", "PSEUDO_SIGMOID", "PSEUDO_RATQUAD", "PROBABILISTIC"};
    std::cout << "The kernel is " << kernType[kern] << " and the coefficients are degree = " << degree << ", gamma = " << gamma << " and coef0 = " << coef0 << std::endl;
    Kernel_prot kernel({kern, degree, gamma, coef0}, &stat);
    std::cout << std::endl;
    std::cout << "We will apply on the dataset the " << r << "-fold method." << std::endl;
    bool all = false;
    bool allTest = true;
    if (!all) {
      std::cout << "We chose to train our classifier with a balanced dataset between the correct and incorrect cleavage sites. All the possibilities are not used to train the classifier." << std::endl;
    }
    else {
      std::cout << "We chose to train our classifier with an unbalanced dataset between the correct and incorrect cleavage sites. All the possibilities are used to train the classifier." << std::endl;
    }
    std::cout << std::endl;
    double C = 1.0;
    double lr = 0.01;
    bool scaled = false;
    std::string sc;
    if (scaled) {
      sc = "";
    }
    else {
      sc = "not";
    }
    std::cout << "The values for the training of the library are C = " << C << " and learning constant = " << lr  << " and the data are " << sc << " scaled."<< std::endl;
    //Then, we define all the variables that will allow us to evaluate our results.
    ConfusionMatrix_prot cm_tot = ConfusionMatrix_prot();
    double avg_error_rate = 0.0;
    double avg_false_alarm_rate = 0.0;
    double avg_detection_rate	= 0.0;
    double avg_f_score = 0.0;
    double avg_precision = 0.0;
    //for all the segments of the r-fold method
    for (int i = 0; i < r; i++) {
      SVM_prot svm(&train_dataset, p, q, i, all, allTest, kernel);
      ConfusionMatrix_prot cm = svm.train_and_test_lib(C, lr, scaled);
      cm_tot.AddMatrix(cm);
      avg_error_rate += cm.error_rate();
      avg_false_alarm_rate += cm.false_alarm_rate();
      avg_detection_rate += cm.detection_rate();
      avg_f_score += cm.f_score();
      avg_precision += cm.precision();
      std::cout << "The confusion matrix number " << i+1 << " was calculated..." << std::endl;
    }
    std::cout << std::endl;
    avg_error_rate /= r;
    avg_false_alarm_rate /= r;
    avg_detection_rate /= r;
    avg_f_score /= r;
    avg_precision /= r;
    std::cout << "Here are some average score results with the " << r << "-fold method:" << std::endl;
    std::cout << "Average error rate = " << avg_error_rate << std::endl;
    std::cout << "Average false alarm rate = " << avg_false_alarm_rate << std::endl;
    std::cout << "Average detection rate = " << avg_detection_rate << std::endl;
    std::cout << "Average f-score = " << avg_f_score << std::endl;
    std::cout << "Average precision = " << avg_precision << std::endl;
    std::cout << std::endl;
    std::cout << "Here is the confusion matrix containing all the dataset but with each subset classified with a classifier using the rest of the dataset:" << std::endl;
    cm_tot.PrintEvaluation();
    return 0;
}


//we test all the kernel to find the best one in terme of f-score
int ex4_choice_of_kernel(std::ostream &out, const std::string test_name) 
{
    std::string entity_name = "[Ex4]_ChoiceOfKernel";
    start_test_suite(out, test_name);
    //To begin, we define all the variables: r, p, q, C, and so on.
    const char* train_file = "data/GRAM+SIG_13.red";
    int r = 10;
    int p = 13;
    int q = 2;
    Dataset_prot train_dataset(train_file, r);
    Statistical_model stat(&train_dataset, p, q, r);
    int degree = 2; // the degree because it is the more common and larger tend to overfit
    double n = double(double(2*(r-1)*train_dataset.GetNbrSamples()) / double(r)); // the number of training samples (2 because for each protein, we take 2 words)
    double gamma = double(1.0 / n);
    double coef0 = 1.0; // a arbitrary choice because it is common
    std::cout << "We will apply on the dataset the " << r << "-fold method." << std::endl;
    bool all = false;
    bool allTest = true;
    if (!all) {
      std::cout << "We chose to train our classifier with a balanced dataset between the correct and incorrect cleavage sites. All the possibilities are not used to train the classifier." << std::endl;
    }
    else {
      std::cout << "We chose to train our classifier with an unbalanced dataset between the correct and incorrect cleavage sites. All the possibilities are used to train the classifier." << std::endl;
    }
    std::cout << std::endl;
    double C = 1.0;
    double lr = 0.01;
    bool scaled = false;
    std::string sc;
    if (scaled) {
      sc = "";
    }
    else {
      sc = "not";
    }
    std::cout << "The values for the training of the library are C = " << C << " and learning constant = " << lr  << " and the data are " << sc << " scaled."<< std::endl;
    
    //we define the best f-score to find the best kernel
    double best_f_score;
    //we kept the best kernel
    int best_kernel;
    std::string kernType[11] = {"LINEAR", "POLY", "RBF", "SIGMOID", "RATQUAD", "PSEUDO_LINEAR", "PSEUDO_POLY", "PSEUDO_RBF", "PSEUDO_SIGMOID", "PSEUDO_RATQUAD", "PROBABILISTIC"};
    
    //we test all the kernel
    for (int j = 0; j < 10; j++) {
      std::cout << "The kernel is " << kernType[j] << " and the coefficients are degree = " << degree << ", gamma = " << gamma << " and coef0 = " << coef0 << std::endl;
      Kernel_prot kernel({j, degree, gamma, coef0},&stat);
      //we define all the variables that will allow us to evaluate the quality of the kernel.
      ConfusionMatrix_prot cm_tot = ConfusionMatrix_prot();
      double avg_error_rate = 0.0;
      double avg_false_alarm_rate = 0.0;
      double avg_detection_rate	= 0.0;
      double avg_f_score = 0.0;
      double avg_precision = 0.0;
      //we test the kernel
      for (int i = 0; i < r; i++) {
        SVM_prot svm(&train_dataset, p, q, i, all, allTest, kernel);
        ConfusionMatrix_prot cm = svm.train_and_test_lib(C, lr, scaled);
        cm_tot.AddMatrix(cm);
        avg_error_rate += cm.error_rate();
        avg_false_alarm_rate += cm.false_alarm_rate();
        avg_detection_rate += cm.detection_rate();
        avg_f_score += cm.f_score();
        avg_precision += cm.precision();
        std::cout << "The confusion matrix number " << i+1 << " was calculated..." << std::endl;
      }
      std::cout << std::endl;
      avg_error_rate /= r;
      avg_false_alarm_rate /= r;
      avg_detection_rate /= r;
      avg_f_score /= r;
      avg_precision /= r;
      std::cout << "Here are some average score results with the " << r << "-fold method:" << std::endl;
      std::cout << "Average error rate = " << avg_error_rate << std::endl;
      std::cout << "Average false alarm rate = " << avg_false_alarm_rate << std::endl;
      std::cout << "Average detection rate = " << avg_detection_rate << std::endl;
      std::cout << "Average f-score = " << avg_f_score << std::endl;
      std::cout << "Average precision = " << avg_precision << std::endl;
      std::cout << std::endl;
      std::cout << "Here is the confusion matrix containing all the dataset but with each subset classified with a classifier using the rest of the dataset:" << std::endl;
      cm_tot.PrintEvaluation();
      
      //we test if we find a better kernel
      if (best_f_score < avg_f_score) {
        best_f_score = avg_f_score;
        best_kernel = j;
      }
      std::cout << std::endl;
      std::cout << std::endl;
    }
    std::cout << "According to the tests and based on the f-score (which is biased towards positives...), the best kernel is " << kernType[best_kernel] << std::endl;
    return 0;
}


//we test all the q-value to find the best one in terme of f-score
int ex5_choice_of_q(std::ostream &out, const std::string test_name) 
{
    std::string entity_name = "[Ex5]_ChoiceOfq";
    start_test_suite(out, test_name);
    //To begin, we define all the variables: p, C, the kernel, and so on.
    const char* train_file = "data/EUKSIG_13.red";
    int r = 10;
    int p = 13;
    Dataset_prot train_dataset(train_file, r);
    Statistical_model stat(&train_dataset, p, 2, r);
    int degree = 2; // the degree because it is the more common and larger tend to overfit
    double n = double(double(2*(r-1)*train_dataset.GetNbrSamples()) / double(r)); // the number of training samples (2 because for each protein, we take 2 words)
    double gamma = double(1.0 / n);
    double coef0 = 1.0; // a arbitrary choice because it is common
    std::cout << "We will apply on the dataset the " << r << "-fold method." << std::endl;
    bool all = false;
    bool allTest = false;
    if (!all) {
      std::cout << "We chose to train our classifier with a balanced dataset between the correct and incorrect cleavage sites. All the possibilities are not used to train the classifier." << std::endl;
    }
    else {
      std::cout << "We chose to train our classifier with an unbalanced dataset between the correct and incorrect cleavage sites. All the possibilities are used to train the classifier." << std::endl;
    }
    std::cout << std::endl;
    double C = 1.0;
    double lr = 0.01;
    bool scaled = false;
    std::string sc;
    if (scaled) {
      sc = "";
    }
    else {
      sc = "not";
    }
    std::cout << "The values for the training of the library are C = " << C << " and learning constant = " << lr  << " and the data are " << sc << " scaled."<< std::endl;
    
    //we define the best f-score to find the best q
    double best_f_score;
    //we kept the q
    int best_q = -1;
    
    //we define the kernel
    int kern = 1;
    std::string kernType[11] = {"LINEAR", "POLY", "RBF", "SIGMOID", "RATQUAD", "PSEUDO_LINEAR", "PSEUDO_POLY", "PSEUDO_RBF", "PSEUDO_SIGMOID", "PSEUDO_RATQUAD", "PROBABILISTIC"};
    std::cout << "The kernel is " << kernType[kern] << " and the coefficients are degree = " << degree << ", gamma = " << gamma << " and coef0 = " << coef0 << std::endl;
    Kernel_prot kernel({kern, degree, gamma, coef0},&stat);
    
    //we test all the q
    for (int q = 1; q < 30; q++) {
      std::cout << "For q = " << q << std::endl;
      //we define all the variables that will allow us to evaluate the quality of the q.
      ConfusionMatrix_prot cm_tot = ConfusionMatrix_prot();
      double avg_error_rate = 0.0;
      double avg_false_alarm_rate = 0.0;
      double avg_detection_rate	= 0.0;
      double avg_f_score = 0.0;
      double avg_precision = 0.0;
      //we test the q
      for (int i = 0; i < r; i++) {
        SVM_prot svm(&train_dataset, p, q, i, all, allTest, kernel);
        ConfusionMatrix_prot cm = svm.train_and_test_lib(C, lr, scaled);
        cm_tot.AddMatrix(cm);
        avg_error_rate += cm.error_rate();
        avg_false_alarm_rate += cm.false_alarm_rate();
        avg_detection_rate += cm.detection_rate();
        avg_f_score += cm.f_score();
        avg_precision += cm.precision();
        std::cout << "The confusion matrix number " << i+1 << " was calculated..." << std::endl;
      }
      std::cout << std::endl;
      avg_error_rate /= r;
      avg_false_alarm_rate /= r;
      avg_detection_rate /= r;
      avg_f_score /= r;
      avg_precision /= r;
      std::cout << "Here are some average score results with the " << r << "-fold method:" << std::endl;
      std::cout << "Average error rate = " << avg_error_rate << std::endl;
      std::cout << "Average false alarm rate = " << avg_false_alarm_rate << std::endl;
      std::cout << "Average detection rate = " << avg_detection_rate << std::endl;
      std::cout << "Average f-score = " << avg_f_score << std::endl;
      std::cout << "Average precision = " << avg_precision << std::endl;
      std::cout << std::endl;
      std::cout << "Here is the confusion matrix containing all the dataset but with each subset classified with a classifier using the rest of the dataset:" << std::endl;
      cm_tot.PrintEvaluation();
      
      //we test if we find a better q
      if (best_f_score < avg_f_score) {
        best_f_score = avg_f_score;
        best_q = q;
      }
      std::cout << std::endl;
      std::cout << std::endl;
    }
    std::cout << "According to the tests and based on the f-score (which is biased towards positives...), the best q is " << best_q << std::endl;
    return 0;
}


//we test all the q-value to find the best one in terme of f-score
int ex6_choice_of_p(std::ostream &out, const std::string test_name) 
{
    std::string entity_name = "[Ex5]_ChoiceOfp";
    start_test_suite(out, test_name);
    //To begin, we define all the variables: q, C, the kernel, and so on.
    const char* train_file = "data/EUKSIG_13.red";
    int r = 10;
    int q = 4;
    Dataset_prot train_dataset(train_file, r);
    Statistical_model stat(&train_dataset, 13, q, r);
    int degree = 2; // the degree because it is the more common and larger tend to overfit
    double n = double(double(2*(r-1)*train_dataset.GetNbrSamples()) / double(r)); // the number of training samples (2 because for each protein, we take 2 words)
    double gamma = double(1.0 / n);
    double coef0 = 1.0; // a arbitrary choice because it is common
    std::cout << "We will apply on the dataset the " << r << "-fold method." << std::endl;
    bool all = false;
    bool allTest = false;
    if (!all) {
      std::cout << "We chose to train our classifier with a balanced dataset between the correct and incorrect cleavage sites. All the possibilities are not used to train the classifier." << std::endl;
    }
    else {
      std::cout << "We chose to train our classifier with an unbalanced dataset between the correct and incorrect cleavage sites. All the possibilities are used to train the classifier." << std::endl;
    }
    std::cout << std::endl;
    double C = 1.0;
    double lr = 0.01;
    bool scaled = false;
    std::string sc;
    if (scaled) {
      sc = "";
    }
    else {
      sc = "not";
    }
    std::cout << "The values for the training of the library are C = " << C << " and learning constant = " << lr  << " and the data are " << sc << " scaled."<< std::endl;
    
    //we define the best f-score to find the best p
    double best_f_score;
    //we kept the p
    int best_p = -1;
    
    //we define the kernel
    int kern = 1;
    std::string kernType[11] = {"LINEAR", "POLY", "RBF", "SIGMOID", "RATQUAD", "PSEUDO_LINEAR", "PSEUDO_POLY", "PSEUDO_RBF", "PSEUDO_SIGMOID", "PSEUDO_RATQUAD", "PROBABILISTIC"};
    std::cout << "The kernel is " << kernType[kern] << " and the coefficients are degree = " << degree << ", gamma = " << gamma << " and coef0 = " << coef0 << std::endl;
    Kernel_prot kernel({kern, degree, gamma, coef0},&stat);
    
    //we test all the p
    for (int p = 0; p < 14; p++) {
      std::cout << "For p = " << p << std::endl;
      //we define all the variables that will allow us to evaluate the quality of the p.
      ConfusionMatrix_prot cm_tot = ConfusionMatrix_prot();
      double avg_error_rate = 0.0;
      double avg_false_alarm_rate = 0.0;
      double avg_detection_rate	= 0.0;
      double avg_f_score = 0.0;
      double avg_precision = 0.0;
      //we test the p
      for (int i = 0; i < r; i++) {
        SVM_prot svm(&train_dataset, p, q, i, all, allTest, kernel);
        ConfusionMatrix_prot cm = svm.train_and_test_lib(C, lr, scaled);
        cm_tot.AddMatrix(cm);
        avg_error_rate += cm.error_rate();
        avg_false_alarm_rate += cm.false_alarm_rate();
        avg_detection_rate += cm.detection_rate();
        avg_f_score += cm.f_score();
        avg_precision += cm.precision();
        std::cout << "The confusion matrix number " << i+1 << " was calculated..." << std::endl;
      }
      std::cout << std::endl;
      avg_error_rate /= r;
      avg_false_alarm_rate /= r;
      avg_detection_rate /= r;
      avg_f_score /= r;
      avg_precision /= r;
      std::cout << "Here are some average score results with the " << r << "-fold method:" << std::endl;
      std::cout << "Average error rate = " << avg_error_rate << std::endl;
      std::cout << "Average false alarm rate = " << avg_false_alarm_rate << std::endl;
      std::cout << "Average detection rate = " << avg_detection_rate << std::endl;
      std::cout << "Average f-score = " << avg_f_score << std::endl;
      std::cout << "Average precision = " << avg_precision << std::endl;
      std::cout << std::endl;
      std::cout << "Here is the confusion matrix containing all the dataset but with each subset classified with a classifier using the rest of the dataset:" << std::endl;
      cm_tot.PrintEvaluation();
      
      //we test if we find a better p
      if (best_f_score < avg_f_score) {
        best_f_score = avg_f_score;
        best_p = p;
      }
      std::cout << std::endl;
      std::cout << std::endl;
    }
    std::cout << "According to the tests and based on the f-score (which is biased towards positives...), the best p is " << best_p << std::endl;
    return 0;
}


//exercice to test the prediction of  cleavage site
int ex7_test_of_cleav_loc(std::ostream &out, const std::string test_name) 
{
    std::string entity_name = "[Ex6]_TestOfCleavLoc";
    start_test_suite(out, test_name);
    //To begin, we define all the variables: r, p, q, C, the kernel, and so on.
    const char* train_file = "data/EUKSIG_13.red";
    int r = 10;
    int p = 12;
    int q = 4;
    Dataset_prot train_dataset(train_file, r);
    Statistical_model stat(&train_dataset, p, q, r);
    int degree = 2; // the degree because it is the more common and larger tend to overfit
    double n = double(double(2*(r-1)*train_dataset.GetNbrSamples()) / double(r)); // the number of training samples (2 because for each protein, we take 2 words)
    double gamma = double(1.0 / n);
    double coef0 = 1.0; // a arbitrary choice because it is common
    int kern = 1;
    //choice of the kernel
    std::string kernType[11] = {"LINEAR", "POLY", "RBF", "SIGMOID", "RATQUAD", "PSEUDO_LINEAR", "PSEUDO_POLY", "PSEUDO_RBF", "PSEUDO_SIGMOID", "PSEUDO_RATQUAD", "PROBABILISTIC"};
    std::cout << "The kernel is " << kernType[kern] << " and the coefficients are degree = " << degree << ", gamma = " << gamma << " and coef0 = " << coef0 << std::endl;
    Kernel_prot kernel({kern, degree, gamma, coef0}, &stat);
    std::cout << std::endl;
    std::cout << "We will apply on the dataset the " << r << "-fold method." << std::endl;
    bool all = false;
    bool allTest = true;
    if (!all) {
      std::cout << "We chose to train our classifier with a balanced dataset between the correct and incorrect cleavage sites. All the possibilities are not used to train the classifier." << std::endl;
    }
    else {
      std::cout << "We chose to train our classifier with an unbalanced dataset between the correct and incorrect cleavage sites. All the possibilities are used to train the classifier." << std::endl;
    }
    std::cout << std::endl;
    double C = 1.0;
    double lr = 0.01;
    std::cout << "The values for the training are C = " << C << " and learning constant = " << lr << std::endl;
    
    //we define and train our svm
    int t = 0;
    SVM_prot svm(&train_dataset, p, q, t, all, allTest, kernel);
    svm.train(C, lr);
    //we collect the rate of well predicted cleavage site for all the proteins
    double prediction_rate = svm.PredictCleavageSites(&train_dataset, t);
    
    std::cout << "The rate of well predicted cleavage site for all the proteins of the data set: " << prediction_rate << std::endl;
    return 0;
}


//exercice special for testing the problem of our "probabilistic kernel"
int ex8_probabilistic_kernel(std::ostream &out, const std::string test_name) 
{
    std::string entity_name = "[Ex8]_Probababilistic_kernel";
    start_test_suite(out, test_name);

    const char* train_file = "data/GRAM+SIG_13.red";

    int r = 10;
    
    Dataset_prot train_dataset(train_file, r);
    
    int p = 12;
    int q = 4;
    std::cout << "The maximum value of p is " << train_dataset.GetMaxp() << " and p = " << p << std::endl;
    std::cout << "The maximum value of q is " << train_dataset.GetMaxq() << " and q = " << q << std::endl;
    std::cout << std::endl;
    int degree = 2; // the degree because it is the more common and larger tend to overfit
    double n = double(double(2*(r-1)*train_dataset.GetNbrSamples()) / double(r)); // the number of training samples (2 because for each protein, we take 2 words)
    double gamma = double(1.0 / n);
    double coef0 = 1.0; // a arbitrary choice because it is common
    int kern = 10;
    std::string kernType[11] = {"LINEAR", "POLY", "RBF", "SIGMOID", "RATQUAD", "PSEUDO_LINEAR", "PSEUDO_POLY", "PSEUDO_RBF", "PSEUDO_SIGMOID", "PSEUDO_RATQUAD", "PROBABILISTIC"};
    
    std::cout << "The kernel is " << kernType[kern] << " and the coefficients are degree = " << degree << ", gamma = " << gamma << " and coef0 = " << coef0 << std::endl;

    std::cout << std::endl;
    std::cout << "We will apply on the dataset the " << r << "-fold method." << std::endl;
    bool all = false;
    bool allTest = true;
    if (!all) {
      std::cout << "We chose to train our classifier with a balanced dataset between the correct and incorrect cleavage sites. All the possibilities are not used to train the classifier." << std::endl;
    }
    else {
      std::cout << "We chose to train our classifier with an unbalanced dataset between the correct and incorrect cleavage sites. All the possibilities are used to train the classifier." << std::endl;

    }
    std::cout << std::endl;
    double C = 1.0;
    double lr = 0.01; bool scaled = false;
    std::string sc;
    if (scaled) {
      sc = "";
    }
    else {
      sc = "not";
    }
    std::cout << "The values for the training of the library are C = " << C << " and learning constant = " << lr  << " and the data are " << sc << " scaled."<< std::endl;
    
    ConfusionMatrix_prot cm_tot = ConfusionMatrix_prot();
    double avg_error_rate = 0.0;
    double avg_false_alarm_rate = 0.0;
    double avg_detection_rate	= 0.0;
    double avg_f_score = 0.0;
    double avg_precision = 0.0;
    for (int i = 0; i < r; i++) {
      Statistical_model stat(&train_dataset, p, q, i);
      stat.s_csv();
      Kernel_prot kernel({10, degree, gamma, coef0}, &stat);
      SVM_prot svm(&train_dataset, p, q, i, all, allTest, kernel);
      ConfusionMatrix_prot cm = svm.train_and_test_lib(C, lr, scaled);
      cm_tot.AddMatrix(cm);
      avg_error_rate += cm.error_rate();
      avg_false_alarm_rate += cm.false_alarm_rate();
      avg_detection_rate += cm.detection_rate();
      avg_f_score += cm.f_score();
      avg_precision += cm.precision();
      std::cout << "The confusion matrix number " << i+1 << " was calculated..." << std::endl;
      remove("s.csv");
    }
    std::cout << std::endl;
    avg_error_rate /= r;
    avg_false_alarm_rate /= r;
    avg_detection_rate /= r;
    avg_f_score /= r;
    avg_precision /= r;
    std::cout << "Here are some average score results with the " << r << "-fold method:" << std::endl;
    std::cout << "Average error rate = " << avg_error_rate << std::endl;
    std::cout << "Average false alarm rate = " << avg_false_alarm_rate << std::endl;
    std::cout << "Average detection rate = " << avg_detection_rate << std::endl;
    std::cout << "Average f-score = " << avg_f_score << std::endl;
    std::cout << "Average precision = " << avg_precision << std::endl;
    std::cout << std::endl;
    std::cout << "Here is the confusion matrix containing all the dataset but with each subset classified with a classifier using the rest of the dataset:" << std::endl;
    cm_tot.PrintEvaluation();
    return 0;
}



int grading(std::ostream &out, const int test_case_number)
{
    int const total_test_cases = 8;
    std::string const test_names[total_test_cases] = {"Ex1_statistical_test", "Ex2_intratest", "Ex3_intratestlib", "Ex4_choiceofkernel", "Ex5_choiceofq", "Ex6_choiceofp", "Ex7_testofcleavloc", "Ex8_probabilistickernel"};
    int const points[total_test_cases] = {10, 10, 10, 10, 10, 10, 10, 10};
    int (*test_functions[total_test_cases]) (std::ostream &, const std::string) = {
      ex1_statistical_test, ex2_intra_test, ex3_intra_test_lib, ex4_choice_of_kernel, ex5_choice_of_q, ex6_choice_of_p, ex7_test_of_cleav_loc, ex8_probabilistic_kernel
    };

    return run_grading(out, test_case_number, total_test_cases,
                       test_names, points,
                       test_functions);
}

} // End of namepsace tdgrading


