#include "confusion_matrix_prot.hpp"
#include <iostream>

using namespace std;

ConfusionMatrix_prot::ConfusionMatrix_prot() {
    // Populate 2x2 matrix with 0s
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            m_confusion_matrix[i][j] = 0;
        }
    }
}

ConfusionMatrix_prot::~ConfusionMatrix_prot() {
    // Attribute m_confusion_matrix is deleted automatically
}

void ConfusionMatrix_prot::AddPrediction(int true_label, int predicted_label) {
    m_confusion_matrix[true_label][predicted_label]++;
}

void ConfusionMatrix_prot::AddMatrix(ConfusionMatrix_prot other) {
    m_confusion_matrix[0][0] += other.GetTN();
    m_confusion_matrix[0][1] += other.GetFP();
    m_confusion_matrix[1][0] += other.GetFN();
    m_confusion_matrix[1][1] += other.GetTP();
}

void ConfusionMatrix_prot::PrintEvaluation() const{
    // Prints the confusion matrix
    cout <<"\t\tPredicted\n";
    cout <<"\t\t0\t1\n";
    cout <<"Actual\t0\t"
        <<GetTN() <<"\t"
        <<GetFP() <<endl;
    cout <<"\t1\t"
        <<GetFN() <<"\t"
        <<GetTP() <<endl <<endl;
    // Prints the estimators
    cout <<"Error rate\t\t"
        <<error_rate() <<endl;
    cout <<"False alarm rate\t"
        <<false_alarm_rate() <<endl;
    cout <<"Detection rate\t\t"
        <<detection_rate() <<endl;
    cout <<"F-score\t\t\t"
        <<f_score() <<endl;
    cout <<"Precision\t\t"
        <<precision() <<endl;
}

int ConfusionMatrix_prot::GetTP() const {
    return m_confusion_matrix[1][1];
}

int ConfusionMatrix_prot::GetTN() const {
   return m_confusion_matrix[0][0];
}

int ConfusionMatrix_prot::GetFP() const {
    return m_confusion_matrix[0][1];
}

int ConfusionMatrix_prot::GetFN() const {
   return m_confusion_matrix[1][0];
}

double ConfusionMatrix_prot::f_score() const {
    double p = precision();
    double r = detection_rate();
    return 2 * p * r / (p + r);
}

double ConfusionMatrix_prot::precision() const {
    return (double)GetTP() / (double)(GetTP() + GetFP());
}

double ConfusionMatrix_prot::error_rate() const {
    return (double)(GetFP() + GetFN()) / 
        (double)(GetFP() + GetFN() + GetTP() + GetTN());
}

double ConfusionMatrix_prot::detection_rate() const {
    return (double)GetTP() / (double)(GetTP() + GetFN());
}

double ConfusionMatrix_prot::false_alarm_rate() const {
   return (double)GetFP() / (double)(GetFP() + GetTN());
}
