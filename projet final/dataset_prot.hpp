#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>

#ifndef DATASET_PROT_HPP
#define DATASET_PROT_HPP


/**
  The Dataset class encapsulates a dataset in vectors of integers or strings and provides a kind of interface to manipulate them.
*/
class Dataset_prot {
    public:
      /**
        The constructor needs the path of the file as a string.
      */
      Dataset_prot(const char* file, int r);

      /**
        Standard destructor
      */
      ~Dataset_prot();

      /**
        The Show method displays the number of instances and columns of the Dataset.
        @param verbose If set to True, the Dataset is also printed.
      */
      void Show(bool verbose) const;

      /**
        Returns a copy of an instance.
        @param i Instance number (= row) to get.
      */
    	const std::string& GetInstance(int i) const;

      /**
          The getter to the number of instances / samples.
        */
    	int GetNbrSamples() const;

      /**
          The getter to the cleavage site localization of an instance.
        */
    	int GetCleavLoc(int i) const;

      /**
          The getter to the cleavage site localization of an instance.
        */
    	int GetNAA(int i) const;

      /**
          The getter to r for the r-fold non-exhaustive cross-validation.
        */
    	int GetR() const;

      /**
          The getter to the r-folding repartition.
        */
    	const std::vector<int>& GetRFold() const;

      /**
          The getter to the cleavage localizations.
        */
    	const std::vector<int>& GetCleavLocs() const;

      /**
          The getter to the length of the proteins.
        */
    	const std::vector<int>& GetNAAs() const;

      /**
          The getter to the maximum value for p.
        */
      int GetMaxp() const;

      /**
          The getter to the maximum value for q.
        */
      int GetMaxq() const;

    private:
      /**
        The number of instances / samples.
      */
      int m_nsamples;
      /**
        The dataset is stored as a vector of strings.
      */
      std::vector<std::string> m_instances;
      /**
        The cleavage site localizations are stored as a vector of integer.
      */
      std::vector<int> m_cleav_locs;
      /**
        The numbers of amino acids per protein are stored as a vector of integer.
      */
      std::vector<int> m_naa;
      /**
        The number of subsets for the r-fold method.
      */
      int m_r;
      /**
        The subsets for the r-fold method are stored as a vector of integer.
        The r firsts integer correspond to the instances in the first subset, etc.
      */
      std::vector<int> m_rFold;
      /**
        The maximum value of p.
      */
      int m_p;
      /**
        The maximum value of q.
      */
      int m_q;
};
#endif //DATASET_PROT_HPP
