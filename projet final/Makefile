CXX = g++
CXXFLAGS = ${INCLUDES} -std=c++11 -O3
INCLUDES = -Igradinglib -Igrading -Idataset_prot -Iconfusion_matrix_prot -Ikernel_prot -Isvm_prot -Istatistical_model

.PHONY: all clean

all: grader

# project-specific part

OBJECTS_SPECIFIC = dataset_prot.o confusion_matrix_prot.o kernel_prot.o svm_prot.o statistical_model.o
OBJECTS_SPECIFIC = dataset_prot.cpp confusion_matrix_prot.cpp kernel_prot.cpp svm_prot.cpp statistical_model.cpp

dataset_prot.o: dataset_prot.cpp
	$(CXX) -c $(INCLUDES) $(CFLAGS) dataset_prot.cpp

confusion_matrix_prot.o: confusion_matrix_prot.cpp
	$(CXX) -c $(INCLUDES) $(CFLAGS) confusion_matrix_prot.cpp

kernel_prot.o: kernel_prot.cpp
	$(CXX) -c $(INCLUDES) $(CFLAGS) kernel_prot.cpp

svm_prot.o: svm_prot.cpp
	$(CXX) -c $(INCLUDES) $(CFLAGS) svm_prot.cpp
 
statistical_model.o: statistical_model.cpp
	$(CXX) -c $(INCLUDES) $(CFLAGS) statistical_model.cpp

# Common part
SOURCES_COMMON = gradinglib/gradinglib.cpp grading/grading.cpp main.cpp
OBJECTS_COMMON = gradinglib.o grading.o main.o

grader: $(OBJECTS_COMMON) $(OBJECTS_SPECIFIC)
	$(CXX) $(CXXFLAGS) -o grader $(OBJECTS_COMMON) $(OBJECTS_SPECIFIC)

gradinglib.o: gradinglib/gradinglib.cpp gradinglib/gradinglib.hpp
	$(CXX) -c $(CXXFLAGS) -o gradinglib.o gradinglib/gradinglib.cpp

grading.o: grading/grading.cpp gradinglib/gradinglib.hpp
	$(CXX) -c $(CXXFLAGS) -o grading.o grading/grading.cpp 

main.o: main.cpp grading/grading.hpp
	$(CXX) -c $(CXXFLAGS) -o main.o main.cpp

clean:
	rm -f grader *.o
