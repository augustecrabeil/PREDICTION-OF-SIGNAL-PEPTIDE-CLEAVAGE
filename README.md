# Overview

The goal of this project is to predict signal peptide cleavage site using supervised learning.

# Table of Contents

- Project Structure
- Installation
- Data
- Models
- Evaluation


# Project Structure

- /projet final: Contains all the C++ code and the data for the project
- p2-cleavage.pdf contains the description of the project
- report.pdf contains the description of our solution

# Installation

In order to use our codes and re-carry out our tests (if you want to check them and try them
:)), there is a Makefile for the compilations and you can use "./grader x", where x is the number
of the test.

# Data

The projet final/data folder contrains for files containing sequences from various organisms, where
the signal peptide is followed by a part of the amino acids of the mature protein:

- EUKSIG 13.red contains 1005 sequences from eucaryotes,
- GRAM+SIG 13.red contains 140 sequences from Gram-positive prokaryotes.
- GRAM-SIG 13.red contains 263 sequences from Gram-negative prokaryotes.
- SIG 13.red is a concatenation of the three files listed above.

Each entry in these files consists in exactly three lines. This is an example of a protein description:
55 2SS8 HELAN 25 ALBUMIN 8 PRECURSOR (METHIONINE-RICH 2S PROTEIN) (SFA8).
MARFSIVFAAAGVLLLVAMAPVSEASTTTIITTIIEENPYGRGRTESGCYQQMEE
SSSSSSSSSSSSSSSSSSSSSSSSSCMMMMMMMMMMMMMMMMMMMMMMMMMMMMM

- The first line gives a description of the protein and its content may be ignored.
- The second line gives the beginning of the primary structure of the protein, with letters encoding the amino acids. Note that the first amino acid is almost always M (methionine), as it corresponds to the start codon.
- The third line is an annotation corresponding to the localization of the cleavage site. For each amino acid on the previous line, a letter indicates whether this amino acid belongs to the signal peptide (annotated with a letter S) or to the mature protein (annotated with a letter M). The first amino acid of the mature protein is annotated with a letter C.


# Models

The description of the model can be found in the report.pdf file.

# Evaluation

The evaluation of the model is simply the accuracy of the prediction.
