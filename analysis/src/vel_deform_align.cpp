// vel_deform_corr.cpp
// A program which computes the alignment between advected velocity / velocity // CM and the deformation axis 

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
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

double align(double vx1, double vy1, double vx2, double vy2);
void getEigen(double mxx, double myy, double mxy, Eigen& eigen);

int main(int argc, char* argv[]) {
  if (argc != 9) {
    cout << "Usage: vel_deform_align npoints coeff startTime endTime "
	 << "timeInc velFile deformFile outFile" << endl;
    return 1;
  }

  int argi = 0;
  int npoints = stoi(string(argv[++argi]), nullptr, 10);
  double coeff = stod(string(argv[++argi]), nullptr);
  long startTime = stol(string(argv[++argi]), nullptr, 10);
  long endTime = stol(string(argv[++argi]), nullptr, 10);
  long timeInc = stol(string(argv[++argi]), nullptr, 10);
  string velFile (argv[++argi]);
  string deformFile (argv[++argi]);
  string outFile (argv[++argi]);

  ifstream deformReader;
  deformReader.open(deformFile);
  if (!deformReader) {
    cout << "Error: cannot open the file " << deformFile << endl;
    return 1;
  }

  ifstream velReader;
  velReader.open(velFile);
  if (!velReader) {
    cout << "Error: cannot open the file " << velFile << endl;
    return 1;
  }
  
  ofstream writer;
  writer.open(outFile);
  if (!writer) {
    cout << "Error: cannot open the file " << outFile << endl;
    return 1;
  }

  string line, str;
  stringstream ss;
  int axis;
  long t;
  double mxx, myy, mxy, lam0, lam1;
  Eigen eigen;
  bool foundDeformData, foundVelData;
  double** deformAxis = create2DArray<double>(npoints,3); // vx vy weight
  double** va = create2DArray<double>(npoints,2);
  double** vcm = create2DArray<double>(npoints,2);
  
  for (long time = startTime; time <= endTime; time += timeInc) {
    // Get deform data
    foundDeformData = false;
    while (getline(deformReader, line)) {
      // Read the two header lines and get time
      getline(deformReader, line);
      ss.clear();
      ss.str(line);
      ss >> str >> t;
      // Only use the data from the specified time period
      if (t > endTime) {
	break;
      } else if (t != time) {
	// Skip data in irrelevant time frames
	for (int n = 0; n < npoints; n++) {
	  getline(deformReader, line);
	}
      } else { // t == time
	for (int n = 0; n < npoints; n++) {
	  getline(deformReader, line);
	  ss.clear();
	  ss.str(line);
	  ss >> mxx >> myy >> mxy;
	  mxx *= coeff;
	  myy *= coeff;
	  mxy *= coeff;
	  // Use armadillo to compute eigenvalues/eigenvectors to minimise
	  // numerical errors when cells are close to being a sphere
	  getEigen(mxx, myy, mxy, eigen);
	  lam0 = eigen.eigenVal(0);
	  lam1 = eigen.eigenVal(1);
	  // Always use the axis corresponding to the larger eigenvalue
	  // (i.e., the elongation axis)
	  axis = lam0 < lam1 ? 1 : 0;
	  deformAxis[n][0] = eigen.eigenVec.at(0,axis);
	  deformAxis[n][1] = eigen.eigenVec.at(1,axis);
	  deformAxis[n][2] = fabs((lam0-lam1)/(lam0+lam1));
	}
	foundDeformData = true;
	break;
      }
    }
    if (!foundDeformData) {
      cout << "Error: cannot find deform data for time = " << time << endl;
      return 1;
    }
    
    // Get velocity data
    foundVelData = false;
    double vax, vay, vcmx, vcmy;
    while (getline(velReader, line)) {
      getline(velReader, line);
      ss.clear();
      ss.str(line);
      ss >> str >> t;
      if (t > endTime) {
	break;
      } else if (t != time) {
	// Skip data in irrelevant time frames
	for (int n = 0; n < npoints; n++) {
	  getline(velReader, line);
	}
      } else { // t == time
	for (int n = 0; n < npoints; n++) {
	  getline(velReader, line);
	  ss.clear();
	  ss.str(line);
	  ss >> vax >> vay >> vcmx >> vcmy;
	  va[n][0] = vax;
	  va[n][1] = vay;
	  vcm[n][0] = vcmx;
	  vcm[n][1] = vcmy;
	}
	foundVelData = true;
	break;
      }
    }
    if (!foundVelData) {
      cout << "Error: cannot find the velocity data for time = "
	   << time << endl;
      return 1;
    }

    // Compute degree of nematic alignment
    double vaDeformAlign = 0.0;
    double vaDeformAlignWeighted = 0.0;
    double vcmDeformAlign = 0.0;
    double vcmDeformAlignWeighted = 0.0;
    double order;
    for (int n = 0; n < npoints; n++) {
      order = align(deformAxis[n][0], deformAxis[n][1], va[n][0], va[n][1]);
      vaDeformAlign += order;
      vaDeformAlignWeighted += order*deformAxis[n][2];
      order = align(deformAxis[n][0], deformAxis[n][1], vcm[n][0], vcm[n][1]);
      vcmDeformAlign += order;
      vcmDeformAlignWeighted += order*deformAxis[n][2];
    }
    
    // Normalise
    vaDeformAlign /= npoints;
    vaDeformAlignWeighted /= npoints;
    vcmDeformAlign /= npoints;
    vcmDeformAlignWeighted /= npoints;
    
    // Output reuslts
    writer << time << " " << vaDeformAlign << " " << vaDeformAlignWeighted
	   << " " << vcmDeformAlign << " " << vcmDeformAlignWeighted << "\n";
  } // Close loop over time

  deformReader.close();
  velReader.close();
  writer.close();

  // Clean up
  deleteArray(deformAxis);
  deleteArray(va);
  deleteArray(vcm);
}

// Compute nematic alignment between two vectors
double align(double vx1, double vy1, double vx2, double vy2) {
  double v1sq = vx1*vx1+vy1*vy1;
  double v2sq = vx2*vx2+vy2*vy2;
  double cos2tavg = 2.0*vx1*vx1/v1sq-1.0;
  double sin2tavg = 2.0*vx1*vy1/v1sq;
  cos2tavg += 2.0*vx2*vx2/v2sq-1.0;
  sin2tavg += 2.0*vx2*vy2/v2sq;
  cos2tavg /= 2.0;
  sin2tavg /= 2.0;
  return sqrt(cos2tavg*cos2tavg+sin2tavg*sin2tavg);
}

void getEigen(double mxx, double myy, double mxy, Eigen& eigen) {
  eigen.matrix.at(0,0) = mxx;
  eigen.matrix.at(0,1) = mxy;
  eigen.matrix.at(1,0) = mxy;
  eigen.matrix.at(1,1) = myy;
  // Note that eigenvectors are stored in columns, not rows
  eig_sym(eigen.eigenVal, eigen.eigenVec, eigen.matrix);
}
