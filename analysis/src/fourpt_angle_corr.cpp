// fourpt_angle_corr.cpp
// A code that computes the orientational correlation function for the system,
// which is defined as
// L_n(t) = 1/N * < sum_i cos[n*(theta_i(t) - theta_i(0))] >_t
// where the average is taken over lag time

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <omp.h>
#include <armadillo>
#include "array.hpp"

using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::stringstream;
using std::string;
using namespace arma;

struct Eigen {
  mat matrix = mat(2,2);
  vec eigenVal = vec(2);
  mat eigenVec = mat(2,2);
};

void getEigen(double mxx, double myy, double mxy, Eigen& eigen);

int main (int argc, char* argv[]) {
  
  if (argc != 10) {
    cout << "Usage: fourpt_angle_corr npoints thres coeff startTime "
	 << "endTime timeInc endShiftTime deformFile outFile" << endl;
    return 1;
  }
  
  int argi = 0;
  int npoints = stoi(string(argv[++argi]), nullptr, 10);
  double thres = stod(string(argv[++argi]), nullptr);
  double coeff = stod(string(argv[++argi]), nullptr);
  long startTime = stoi(string(argv[++argi]), nullptr, 10);
  long endTime = stoi(string(argv[++argi]), nullptr, 10);
  long timeInc = stoi(string(argv[++argi]), nullptr, 10);
  long endShiftTime = stoi(string(argv[++argi]), nullptr, 10);
  string deformFile (argv[++argi]);
  string outFile (argv[++argi]);
  
  // Read deformation data
  ifstream reader;
  reader.open(deformFile);
  if (!reader) {
    cout << "Error: cannot read the file " << deformFile << endl;
  }
  
  int nbins = static_cast<int>((endTime-startTime)/timeInc)+1;
  double** angle = create2DArray<double>(nbins, npoints);
  long time;
  int ibin, axis;
  double sxx, syy, sxy, lam0, lam1, vx, vy;
  const double pi = M_PI;
  Eigen eigen;
  string line, str;
  stringstream ss;
  
  while (getline(reader, line)) {
    getline(reader, line);
    ss.clear();
    ss.str(line);
    ss >> str >> time;
    // Only use the data from the specified time period
    if (time < startTime || (time-startTime) % timeInc != 0) {
      // Skip the data in that time frame
      for (int i = 0; i < npoints; i++) {
	getline(reader, line);
      }
    } else if (time > endTime) {
      break;
    } else {
      ibin = static_cast<int>((time-startTime)/timeInc);
      // Read shape tensor data
      for (int n = 0; n < npoints; n++) {
	getline(reader, line);
	ss.clear();
	ss.str(line);
	// Input tensor contains trace
	ss >> sxx >> syy >> sxy;
	sxx *= coeff;
	syy *= coeff;
	sxy *= coeff;
	getEigen(sxx, syy, sxy, eigen);
	// Compute deformability
	lam0 = eigen.eigenVal.at(0);
	lam1 = eigen.eigenVal.at(1);
	// Always use the larger eigenvalue - this should correspond to the
	// elongation axis (coeff must be set to ensure this is the case)
	axis = lam0 < lam1 ? 1 : 0;
	vx = eigen.eigenVec.at(0,axis);
	vy = eigen.eigenVec.at(1,axis);
	angle[ibin][n] = atan2(-vy,-vx)+pi;
      }
    }
  }
  reader.close();
  
  // Compute the orientational correlation function
  int endShiftBin = static_cast<int>((endShiftTime-startTime)/timeInc)+1;
  if (endShiftBin >= nbins) endShiftBin = nbins-1;

#ifdef _OPENMP
  int nthreads = omp_get_max_threads();
#else
  int nthreads = 1;
#endif
  double** qavg = create2DArray<double>(nthreads, nbins);
  double** qavgsq = create2DArray<double>(nthreads, nbins);
  long** count = create2DArray<long>(nthreads, nbins);

#pragma omp parallel default(none)					\
  shared(nbins, npoints, thres, angle, qavg, qavgsq, count, endShiftBin) \
  private(ibin)
  {
#ifdef _OPENMP
    int id = omp_get_thread_num();
#else
    int id = 0;
#endif
    double dangle, q;
#pragma omp for schedule(dynamic,10)
    for (int i = 0; i < endShiftBin; i++) {
      for (int j = i+1; j < nbins; j++) {
	ibin = j-i;
	q = 0.0;
	for (int n = 0; n < npoints; n++) {	  
	  dangle = angle[j][n]-angle[i][n];
	  q += exp(-dangle*dangle/(2.0*thres*thres));
	}
	q /= npoints;
	qavg[id][ibin] += q;
	qavgsq[id][ibin] += q*q;
	count[id][ibin]++;
      }     
    }
  }

  // Combine results
  for (int i = 1; i < nthreads; i++) {
    for (int j = 0; j < nbins; j++) {
      qavg[0][j] += qavg[i][j];
      qavgsq[0][j] += qavgsq[i][j];
      count[0][j] += count[i][j];
    }
  }
  // Normalise results
  qavg[0][0] = 0.0;
  qavgsq[0][0] = 0.0;
  for (int i = 1; i < nbins; i++) {
    qavg[0][i] /= static_cast<double>(count[0][i]);
    qavgsq[0][i] /= static_cast<double>(count[0][i]);
  }

  // Output results
  ofstream writer;
  writer.open(outFile);
  if (!writer) {
    cout << "Error: cannot open the file " << outFile << endl;
    return 1;
  }
  
  for (int i = 0; i < nbins; i++) {
    writer << (i*timeInc) << " "
	   << npoints * (qavgsq[0][i] - qavg[0][i]*qavg[0][i]) << "\n";
  }
  writer.close();
  
  // Clean up
  deleteArray(angle);
  deleteArray(qavg);
  deleteArray(qavgsq);
  deleteArray(count);
}

void getEigen(double mxx, double myy, double mxy, Eigen& eigen) {
  eigen.matrix.at(0,0) = mxx;
  eigen.matrix.at(0,1) = mxy;
  eigen.matrix.at(1,0) = mxy;
  eigen.matrix.at(1,1) = myy;
  // Note that eigenvectors are stored in columns, not rows
  eig_sym(eigen.eigenVal, eigen.eigenVec, eigen.matrix);
}
