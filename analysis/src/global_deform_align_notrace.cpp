// global_deform_align_notrace.cpp
// A program which computes the global deformation (elongation) and nematic
// alignment of the system by averaging over cell deformation tensors
// (not Q field)

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <armadillo>

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

int main(int argc, char* argv[]) {
  if (argc != 9) {
    cout << "Usage: global_deform_align_notrace npoints weighted coeff "
	 << "startTime endTime timeInc deformFile outFile" << endl;
    return 1;
  }

  int argi = 0;
  int npoints = stoi(string(argv[++argi]), nullptr, 10);
  int weighted = stoi(string(argv[++argi]), nullptr, 10);
  double coeff = stod(string(argv[++argi]), nullptr);
  long startTime = stol(string(argv[++argi]), nullptr, 10);
  long endTime = stol(string(argv[++argi]), nullptr, 10);
  long timeInc = stol(string(argv[++argi]), nullptr, 10);
  string deformFile (argv[++argi]); // deformation tensor file
  string outFile (argv[++argi]);

  ifstream reader;
  reader.open(deformFile);
  if (!reader) {
    cout << "Error: cannot open the file " << deformFile << endl;
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
  long time;
  int axis;
  double sxx, syy, sxy, lam0, lam1, vx, vy, w, cos2t, sin2t;
  double cos2tavg, sin2tavg, wcos2tavg, wsin2tavg, wavg, nem, wnem;
  Eigen eigen;
  while (getline(reader, line)) {
    // Read the two header lines and get time
    getline(reader, line);
    ss.clear();
    ss.str(line);
    ss >> str >> time;

    // Only use the data from the specified time period
    if (time < startTime || (time-startTime) % timeInc != 0) {
      // Skip the data in irrelevant time frames
      for (int n = 0; n < npoints; n++) {
	getline(reader, line);
      }
    } else if (time > endTime) {
      break;
    } else {
      cos2tavg = 0.0;
      sin2tavg = 0.0;
      wcos2tavg = 0.0;
      wsin2tavg = 0.0;
      wavg = 0.0;
      for (int n = 0; n < npoints; n++) {
	getline(reader, line);
	ss.clear();
	ss.str(line);
	ss >> sxx >> syy >> sxy;
	sxx *= coeff;
	syy *= coeff;
	sxy *= coeff;
	// Use armadillo to compute eigenvalues/eigenvectors to minimise
	// numerical errors when cells are close to being a sphere
	getEigen(sxx, syy, sxy, eigen);
	lam0 = eigen.eigenVal(0);
	lam1 = eigen.eigenVal(1);
	w = fabs(lam0-lam1);
	if (weighted == 1) {
	  w /= fabs(lam0+lam1);
	}
	// Always use the larger eigenvalue - this should correspond to the
	// elongation axis (coeff must be set to ensure this is the case)
	axis = lam0 < lam1 ? 1 : 0;
	vx = eigen.eigenVec.at(0,axis);
	vy = eigen.eigenVec.at(1,axis);
	cos2t = 2.0*vx*vx-1.0;
	sin2t = 2.0*vx*vy;
	cos2tavg += cos2t;
	sin2tavg += sin2t;
	wcos2tavg += w*cos2t;
	wsin2tavg += w*sin2t;
	wavg += w;
      }
      // Normalise
      cos2tavg /= npoints;
      sin2tavg /= npoints;
      wcos2tavg /= npoints;
      wsin2tavg /= npoints;
      wavg /= npoints;
      nem = sqrt(cos2tavg*cos2tavg+sin2tavg*sin2tavg);
      wnem = sqrt(wcos2tavg*wcos2tavg+wsin2tavg*wsin2tavg);
      writer << time << " " << wavg << " " << nem << " " << wnem << "\n";
    }
  }
  reader.close();
  writer.close();
}

void getEigen(double mxx, double myy, double mxy, Eigen& eigen) {
  eigen.matrix.at(0,0) = mxx;
  eigen.matrix.at(0,1) = mxy;
  eigen.matrix.at(1,0) = mxy;
  eigen.matrix.at(1,1) = myy;
  // Note that eigenvectors are stored in columns, not rows
  eig_sym(eigen.eigenVal, eigen.eigenVec, eigen.matrix);
}
