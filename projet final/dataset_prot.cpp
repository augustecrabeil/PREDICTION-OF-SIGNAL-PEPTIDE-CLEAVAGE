
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>
#include "dataset_prot.hpp"


int Dataset_prot::GetNbrSamples() const {
	return m_nsamples;
}

Dataset_prot::~Dataset_prot() {
}

void Dataset_prot::Show(bool verbose) const {
	std::cout<<"Dataset with "<<m_nsamples<<" samples."<<std::endl;

	if (verbose) {
		for (int i=0; i<m_nsamples; i++) {
			std::cout<<m_instances[i]<<std::endl;		
		}
	}
}

Dataset_prot::Dataset_prot(const char* file, int r) {
	m_nsamples = 0;
    m_r = r;
    m_p = 0;
    m_q = 0;

	std::ifstream fin(file);
	
	if (fin.fail()) {
		std::cout<<"Cannot read from file "<<file<<" !"<<std::endl;
		exit(1);
	}
	
    int loc;
    int naa;
    std::string line;
    bool first = true;
	while (true) {
        if (!getline(fin, line)) { // the useless information of the protein
            break;
        }
        getline(fin, line); // the amino acids of the protein
        
        int naa = line.length();
        m_instances.push_back(line);
        m_naa.push_back(naa);
        
        getline(fin, line); // the cleavage site localization of the protein
        int loc = 0;
        while (line[loc] != 'C') {
            loc ++;
        }
        m_cleav_locs.push_back(loc);

        if (first) {
            m_p = loc;
            m_q = naa - loc;
            first = false;
        }
        if (m_p > loc) {
            m_p = loc;
        }
        if (m_q > naa - loc) {
            m_q = naa - loc;
        }
    m_nsamples ++;
	}
    for (int i = 0; i < m_nsamples; i++) {
        m_rFold.push_back(i);
    }
    std::random_shuffle(m_rFold.begin(), m_rFold.end());
	
	fin.close();
}

const std::string& Dataset_prot::GetInstance(int i) const {
	return m_instances[i];
}

int Dataset_prot::GetCleavLoc(int i) const {
	return m_cleav_locs[i];
}

int Dataset_prot::GetNAA(int i) const {
	return m_naa[i];
}

int Dataset_prot::GetR() const {
	return m_r;
}

const std::vector<int>& Dataset_prot::GetRFold() const {
	return m_rFold;
}

const std::vector<int>& Dataset_prot::GetCleavLocs() const {
	return m_cleav_locs;
}

const std::vector<int>& Dataset_prot::GetNAAs() const {
	return m_naa;
}

int Dataset_prot::GetMaxp() const {
	return m_p;
}

int Dataset_prot::GetMaxq() const {
	return m_q;
}