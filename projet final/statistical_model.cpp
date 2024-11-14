#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <fstream>

#include "statistical_model.hpp"
#include "confusion_matrix_prot.hpp"

/**
The constructor 
We count the amino acids of the data in order to calcul g(a),f(a,i) and s(a,i)
*/
Statistical_model::Statistical_model(Dataset_prot* dataset_prot, int _p, int _q, int _t){
    p = _p;
    q = _q;
    t = _t;
    int r = dataset_prot->GetR();
    int N = dataset_prot->GetNbrSamples();
    f = std::vector<std::vector<double>>(26, std::vector<double>(p+q));
    g = std::vector<double>(26);
    s = std::vector<std::vector<double>>(26, std::vector<double>(p+q));
    int count = 0;
    //fistly, we count the amino acids in the t-1 first segements of the data for cross-validation purposes
    for (int i = 0; i < (N * t/r); i++){
        const std::string& ligne = dataset_prot->GetInstance(dataset_prot->GetRFold()[i]);
        int L = ligne.size();
        for (int j = 1; j < L; j++){ //we didn't count the first letter to avoid a bias
            char lettre = ligne[j];
            int indice = lettre - 'A';
            g[indice] +=  1;
            count += 1;
        }
        int cleav_loc = dataset_prot->GetCleavLoc(dataset_prot->GetRFold()[i]);
        int m = std::max(0,cleav_loc - p);  //warning if cleav_loc < p
        for (int j = m; j < (cleav_loc + q); j++){
            char lettre = ligne[j];
            int indice = lettre - 'A';
            f[indice][j-m] +=  1;
        }
    }
    //secondly, we perform a count of the amino acids in the segment from t+1 to r of the data for cross-validation purposes.
    for (int i = (N * (t + 1) /r); i < N; i++){
        const std::string& ligne = dataset_prot->GetInstance(dataset_prot->GetRFold()[i]);
        int L = ligne.size();
        for (int j = 1; j < L; j++){
            char lettre = ligne[j];
            int indice = lettre - 'A';
            g[indice] +=  1;
            count += 1;
        }
        int cleav_loc = dataset_prot->GetCleavLoc(dataset_prot->GetRFold()[i]);
        int m = std::max(0,cleav_loc - p);
        for (int j = m; j < (cleav_loc + q); j++){
            char lettre = ligne[j];
            int indice = lettre - 'A';
            f[indice][j-m] +=  1;
        }
    }
    for (int a = 0; a < 26; a++){
        g[a] = (g[a] + alpha)/(count + 26 * alpha); //pseudo-counts
        for (int i = 0; i < (p+q); i++){
            f[a][i] = (f[a][i] + alpha)/(N + 26 * alpha); //pseudo-counts
            s[a][i] = log(f[a][i]) - log(g[a]);
        }
    }
}


Statistical_model::~Statistical_model() {
}

int Statistical_model::get_p() {
    return p;
}

int Statistical_model::get_q() {
    return q;
}

double Statistical_model::get_threshold() {
    return threshold;
}

std::vector<std::vector<double>> Statistical_model::get_s() {
    return s;
}


//we use the score describe by the subject
double Statistical_model::score(const std::string& w){
    double sum = 0;
    for (int i = 0;  i < p+q; i++){
        char lettre = w[i];
        int indice = lettre - 'A';
        sum += s[indice][i];
    }
    return sum;
}


//this threshold is the minimum of the score of all cleav_loc
void Statistical_model::set_threshold_1(Dataset_prot* dataset_prot){
    int N = dataset_prot->GetNbrSamples();
    threshold = 1000000000;
    for (int i = 0; i < N; i++){
        const std::string& ligne = dataset_prot->GetInstance(i);
        int cleav_loc = dataset_prot->GetCleavLoc(i);
        const std::string& w = ligne.substr(cleav_loc - p, p+q);
        double sc = score(w);
        threshold = std::min(threshold,sc);
    }
}


//this threshold is the maximum of the score of all cleav_loc
void Statistical_model::set_threshold_2(Dataset_prot* dataset_prot){
    int N = dataset_prot->GetNbrSamples();
    threshold = -10000000;
    for (int i = 0; i < N; i++){
        const std::string& ligne = dataset_prot->GetInstance(i);
        int cleav_loc = dataset_prot->GetCleavLoc(i);
        const std::string& w = ligne.substr(cleav_loc - p, p+q);
        double sc = score(w);
        threshold = std::max(threshold,sc);
    }
}


//this threshold is the mean of the score of all cleav_loc
void Statistical_model::set_threshold_3(Dataset_prot* dataset_prot){
    int r = dataset_prot->GetR();
    int N = dataset_prot->GetNbrSamples();
    threshold = 0;
    int count = 0;
    for (int i = 0; i < (N - N/r); i++){
        const std::string& ligne = dataset_prot->GetInstance(i);
        int cleav_loc = dataset_prot->GetCleavLoc(i);
        const std::string& w = ligne.substr(cleav_loc - p, p+q);
        threshold += score(w);
        count += 1;
    }
    threshold = threshold / count;
}


//this threshold is the best threshold (in terme of f-score) between all the number 
//from the minimum threshold and the maximum threshold with a step of e. 
void Statistical_model::set_threshold_4(Dataset_prot* dataset_prot,double e){
    set_threshold_1(dataset_prot);
    int threshold_min = std::round(threshold);
    ConfusionMatrix_prot cm = test(dataset_prot);
    double f_max = cm.f_score();
    double res = threshold_min;
    set_threshold_2(dataset_prot);
    int threshold_max = std::round(threshold);
    for (int i = 0; i < ((threshold_max - threshold_min + 1)/e); i++){
        threshold = threshold_min + e * i;
        ConfusionMatrix_prot cm = test(dataset_prot);
        double f = cm.f_score();
        if (f > f_max) {f_max = f; res = threshold;}
    }
    threshold = res;
}

//give the best cleavage site of a protein
int Statistical_model::best_place(const std::string& w){
    const std::string w0 = w.substr(0, p+q);
    double res = score(w);
    int place = p;
    int N = w.size();
    for (int i=1; i < (N - (q + p)); i++){
        const std::string w0 = w.substr(1, p+q);
        double s = score(w0);
        if (s>res){res = s; place = i+p;}
    }
    return place;
}


int Statistical_model::binary_classifer(const std::string& w){
    double sum = score(w);
    if (sum >= threshold) {return 1;}
    else {return 0;}
}


ConfusionMatrix_prot Statistical_model::test(Dataset_prot* dataset_prot){
    int r = dataset_prot->GetR();
    int N = dataset_prot->GetNbrSamples();
    ConfusionMatrix_prot cm;
    for (int i = (N * t/r); i < (N * (t+1) /r); i++){
        const std::string& ligne = dataset_prot->GetInstance(dataset_prot->GetRFold()[i]);
        int cleav_loc = dataset_prot->GetCleavLoc(dataset_prot->GetRFold()[i]);
        int L = ligne.size();
        for (int j = 0; j < (L - q - p); j++){
            const std::string& w = ligne.substr(j, p+q);
            int prev = binary_classifer(w);
            int res = 0;
            if (j + p == cleav_loc){res = 1;}
            cm.AddPrediction(res, prev);
        }
    }
    return cm;
}

int Statistical_model::s_csv(){
  std::ofstream file("s.csv");
  if (!file.is_open()) {
    std::cout << "error open file" << std::endl;
    return 1;
  }
  for (int i = 0; i < 26; i++){
    for (int j = 0; j < p+q; j++){
      file << s[i][j] << " ";
    }
    file << std::endl;
  }
  file.close();
  return 0;
}
