// defect_deform_corr.cpp
// A program which measures the deformation of cells near nematic defects

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <armadillo>
#include "position.hpp"
#include "array.hpp"

using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::stringstream;
using std::string;
using std::vector;
using namespace arma;

// Helper functions
struct Eigen {
  mat matrix = mat(2,2);
  vec eigenVal = vec(2);
  mat eigenVec = mat(2,2);
};
struct Defect {
  double pt[2];
  double q;
};

double ddiff(double len, double x1, double x2);
double ddistsq(int ndims, double* len, double* v1, double* v2);
void getEigen(double mxx, double myy, double mxy, Eigen& eigen);

int main(int argc, char* argv[]) {
  if (argc != 14) {
    cout << "Usage: defect_deform_corr npoints lx ly coeff thres "
	 << "startTime endTime timeInc posFile deformFile defectFile "
	 << "pveDefectCorrFile nveDefectCorrFile" << endl;
    return 1;
  }
  
  int argi = 0;
  int npoints = stoi(string(argv[++argi]), nullptr, 10);
  int lx = stoi(string(argv[++argi]), nullptr, 10);
  int ly = stoi(string(argv[++argi]), nullptr, 10);
  double coeff = stod(string(argv[++argi]), nullptr);
  double thres = stod(string(argv[++argi]), nullptr);
  long startTime = stol(string(argv[++argi]), nullptr, 10);
  long endTime = stol(string(argv[++argi]), nullptr, 10);
  long timeInc = stol(string(argv[++argi]), nullptr, 10);
  string posFile (argv[++argi]);
  string deformFile (argv[++argi]);
  string defectFile (argv[++argi]);
  string pveDefectCorrFile (argv[++argi]);
  string nveDefectCorrFile (argv[++argi]);
  double thressq = thres*thres;

  PositionReader posReader;
  if (!posReader.open(posFile, npoints, lx, ly)) {
    cout << "Error: cannot open the file " << posFile << endl;
    return 1;
  }

  ifstream deformReader;
  deformReader.open(deformFile);
  if (!deformReader) {
    cout << "Error: cannot open the file " << deformFile << endl;
    return 1;
  }

  ifstream defectReader;
  defectReader.open(defectFile);
  if (!defectReader) {
    cout << "Error: cannot open the file " << defectFile << endl;
    return 1;
  }

  ofstream pveDefectWriter;
  pveDefectWriter.open(pveDefectCorrFile);
  if (!pveDefectWriter) {
    cout << "Error: cannot open the file " << endl;
    return 1;
  }

  ofstream nveDefectWriter;
  nveDefectWriter.open(nveDefectCorrFile);
  if (!nveDefectWriter) {
    return 1;
  }

  const int ndims = 2;
  double** pos = create2DArray<double>(npoints,ndims);
  double* deform = create1DArray<double>(npoints);

  string str, line;
  stringstream ss;
  long t;
  int ndefects;
  double sxx, syy, sxy, lam0, lam1, x, y, q;
  Eigen eigen;
  vector<Defect> defects;

  bool readPos = false;
  bool readDeform = false;
  bool readDefect = false;
  double boxsize[2] = {(double) lx, (double) ly};
  
  for (long time = startTime; time <= endTime; time += timeInc) {
    // Read positions
    while (posReader.nextFrame()) {
      t = posReader.getTime();
      if (t > endTime) {
	readPos = false;
	break;
      } else if (t != time) {
	readPos = false;
      } else {
	for (int i = 0; i < npoints; i++) {
	  pos[i][0] = posReader.getPosition(i,0);
	  pos[i][1] = posReader.getPosition(i,1);
	}
	readPos = true;
	break;
      }
    }

    // Read deformation tensors
    while (getline(deformReader, line)) {
      getline(deformReader, line);
      ss.clear();
      ss.str(line);
      ss >> str >> t;
      if (t > endTime) {
	readDeform = false;
	break;
      } else if (t != time) {
	for (int i = 0; i < npoints; i++) {
	  getline(deformReader, line);
	}
	readDeform = false;
      } else {
	for (int i = 0; i < npoints; i++) {
	  getline(deformReader, line);
	  ss.clear();
	  ss.str(line);
	  ss >> sxx >> syy >> sxy;
	  sxx *= coeff;
	  syy *= coeff;
	  sxy *= coeff;
	  getEigen(sxx, syy, sxy, eigen);
	  // Compute deformability
	  lam0 = eigen.eigenVal.at(0);
	  lam1 = eigen.eigenVal.at(1);
	  deform[i] = fabs((lam0-lam1)/(lam0+lam1));
	}
	readDeform = true;
	break;
      }
    }
    
    // Read defects
    defects.clear();
    while (getline(defectReader, line)) {
      ss.clear();
      ss.str(line);
      ss >> str >> ndefects;
      getline(defectReader, line);
      ss.clear();
      ss.str(line);
      ss >> str >> t;
      if (t > endTime) {
	readDefect = false;
	break;
      } else if (t != time) {
	for (int i = 0; i < ndefects; i++) {
	  getline(defectReader, line);
	}
	readDefect = false;
      } else {
	defects.reserve(ndefects);
	for (int i = 0; i < ndefects; i++) {
	  getline(defectReader, line);
	  ss.clear();
	  ss.str(line);
	  ss >> x >> y >> q;
	  defects.push_back({{x,y},q});
	}
	readDefect = true;
	break;
      }
    }

    if (!readPos || !readDeform || !readDefect) break;
    
    // Correlate deformation and defects
    for (size_t i = 0; i < defects.size(); i++) {
      for (int j = 0; j < npoints; j++) {
	if (ddistsq(ndims, boxsize, defects[i].pt, pos[j]) < thressq) {
	  if (defects[i].q > 0.0) {
	    pveDefectWriter << deform[j] << "\n";
	  } else {
	    nveDefectWriter << deform[j] << "\n";
	  }
	}
      }
    }
  }
  
  posReader.close();
  deformReader.close();
  defectReader.close();
  pveDefectWriter.close();
  nveDefectWriter.close();
}

double ddiff(double len, double x1, double x2) {
  double dx = x1-x2;
  return dx-round(dx/len)*len;
}

double ddistsq(int ndims, double* len, double* v1, double* v2) {
  double sum = 0.0;
  double dx;
  for (int i = 0; i < ndims; i++) {
    dx = ddiff(len[i], v1[i], v2[i]);
    sum += dx*dx;
  }
  return sum;
}

void getEigen(double mxx, double myy, double mxy, Eigen& eigen) {
  eigen.matrix.at(0,0) = mxx;
  eigen.matrix.at(0,1) = mxy;
  eigen.matrix.at(1,0) = mxy;
  eigen.matrix.at(1,1) = myy;
  // Note that eigenvectors are stored in columns, not rows
  eig_sym(eigen.eigenVal, eigen.eigenVec, eigen.matrix);
}
