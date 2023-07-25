// local_deform_align_notrace.cpp
// A program which computes the local nematic order of the system

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <armadillo>
#include <omp.h>
#include "position.hpp"
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

double ddiff(double len, double x1, double x2);
double ddist(int ndims, double* len, double* v1, double* v2);
void getEigen(double mxx, double myy, double mxy, Eigen& eigen);

int main(int argc, char* argv[]) {
  if (argc != 17) {
    cout << "Usage: local_deform_align_notrace npoints lx ly ngridx ngridy "
	 << "rmin rmax rinc weighted coeff startTime endTime timeInc "
	 << "posFile deformFile outFile" << endl;
    return 1;
  }

  int argi = 0;
  int npoints = stoi(string(argv[++argi]), nullptr, 10);
  int lx = stoi(string(argv[++argi]), nullptr, 10);
  int ly = stoi(string(argv[++argi]), nullptr, 10);
  int ngridx = stoi(string(argv[++argi]), nullptr, 10);
  int ngridy = stoi(string(argv[++argi]), nullptr, 10);
  double rmin = stod(string(argv[++argi]), nullptr);
  double rmax = stod(string(argv[++argi]), nullptr);
  double rinc = stod(string(argv[++argi]), nullptr);
  int weighted = stoi(string(argv[++argi]), nullptr, 10);
  double coeff = stod(string(argv[++argi]), nullptr);
  long startTime = stol(string(argv[++argi]), nullptr, 10);
  long endTime = stol(string(argv[++argi]), nullptr, 10);
  long timeInc = stol(string(argv[++argi]), nullptr, 10);
  string posFile (argv[++argi]);
  string deformFile (argv[++argi]);
  string outFile (argv[++argi]);

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

  ofstream writer;
  writer.open(outFile);
  if (!writer) {
    cout << "Error: cannot open the file " << outFile << endl;
    return 1;
  }

  const int ndims = 2;
  const int norders = 3; // Number of alignment order observables
  double** pos = create2DArray<double>(npoints,ndims);
  double** alignData = create2DArray<double>(npoints,norders);
  double*** sep = create3DArray<double>(ngridx,ngridy,npoints);
  double* gridx = create1DArray<double>(ngridx);
  double* gridy = create1DArray<double>(ngridy);  
  
  // Compute grid positions
  double gridxWidth = lx/static_cast<double>(ngridx);
  double gridyWidth = ly/static_cast<double>(ngridy);
  for (int i = 0; i < ngridx; i++) {
    gridx[i] = (i+0.5)*gridxWidth;
  }
  for (int i = 0; i < ngridy; i++) {
    gridy[i] = (i+0.5)*gridyWidth;
  }

  int nbins = static_cast<int>((rmax-rmin)/rinc);  
  int ntbins = static_cast<int>((endTime-startTime)/timeInc)+1;  
  double** alignOrderAvg = create2DArray<double>(nbins,norders);
  double dims[ndims] = {(double) lx, (double) ly};
  double sxx, syy, sxy, w, vx, vy, lam0, lam1;
  int axis;
  long t;
  stringstream ss;
  string line, str;
  Eigen eigen;  
  bool foundPosData, foundAlignData;  
  
  for (long time = startTime; time <= endTime; time += timeInc) {
    cout << "Working on t = " << time << endl;
    foundPosData = false;
    foundAlignData = false;
    while (posReader.nextFrame()) {
      t = posReader.getTime();
      if (t > time) {
	break;
      } else if (t == time) {
	for (int n = 0; n < npoints; n++) {
	  pos[n][0] = posReader.getPosition(n,0);
	  pos[n][1] = posReader.getPosition(n,1);
	}
	foundPosData = true;
	break;
      }
    }
    if (!foundPosData) {
      cout << "Error: cannot find the position data for time = " 
	   << time << endl;
      return 1;
    }
    
    while (getline(deformReader, line)) {
      // Read the two header lines and get time
      getline(deformReader, line);
      ss.clear();
      ss.str(line);
      ss >> str >> t;
      if (t > endTime) {
	break;
      } else if (t != time) {
	// Skip data at current time frame
	for (int n = 0; n < npoints; n++) {
	  getline(deformReader, line);
	}	
      } else if (t == time) {
	for (int n = 0; n < npoints; n++) {
	  getline(deformReader, line);
	  ss.clear();
	  ss.str(line);
	  ss >> sxx >> syy >> sxy;
	  sxx *= coeff;
	  syy *= coeff;
	  sxy *= coeff;
	  // Use armadillo to compute eigenvalues/eigenvectors to minimise
	  // numerical errors when cells are close to be a sphere
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
	  alignData[n][0] = 2.0*vx*vx-1.0; // cos2t
	  alignData[n][1] = 2.0*vx*vy; // sin2t
	  alignData[n][2] = w; // weight
	}
	foundAlignData = true;
	break;
      }      
    }    
    
    if (!foundAlignData) {
      cout << "Error: cannot find the alignment data for time = "
	   << time << endl;
      return 1;
    }

    // Compute distance matrix
    double rg[ndims];
    for (int i = 0; i < ngridx; i++) {
      rg[0] = gridx[i];
      for (int j = 0; j < ngridy; j++) {
	rg[1] = gridy[j];
	for (int k = 0; k < npoints; k++) {
	  sep[i][j][k] = ddist(ndims, dims, rg, pos[k]);
	}
      }
    }
    
    // Compute local alignment order for each box length
#pragma omp parallel default(none),\
  shared(rmin, rinc, nbins, npoints, ngridx, ngridy, sep, alignData), \
  shared(alignOrderAvg) private(w)
    {
      int count, ngridptsWithData;
      double nemavgAll, wnemavgAll, wavgAll;
      double cos2tavg, sin2tavg, wcos2tavg, wsin2tavg, wavg, cos2t, sin2t;
#pragma omp for schedule(dynamic)
      for (int n = 0; n < nbins; n++) {
	double r = n*rinc+rmin;
	nemavgAll = 0.0;
	wnemavgAll = 0.0;
	wavgAll = 0.0;
	ngridptsWithData = 0;
	for (int i = 0; i < ngridx; i++) {
	  for (int j = 0; j < ngridy; j++) {
	    count = 0;
	    cos2tavg = 0.0;
	    sin2tavg = 0.0;
	    wcos2tavg = 0.0;
	    wsin2tavg = 0.0;
	    wavg = 0.0;
	    for (int k = 0; k < npoints; k++) {
	      if (sep[i][j][k] < r) {
		cos2t = alignData[k][0];
		sin2t = alignData[k][1];
		w = alignData[k][2];
		cos2tavg += cos2t;
		sin2tavg += sin2t;
		wcos2tavg += w*cos2t;
		wsin2tavg += w*sin2t;
		wavg += w;
		count++;
	      }
	    }
	    if (count > 0) {
	      cos2tavg /= count;
	      sin2tavg /= count;
	      wcos2tavg /= count;
	      wsin2tavg /= count;
	      wavg /= count;
	      nemavgAll += sqrt(cos2tavg*cos2tavg+sin2tavg*sin2tavg);
	      wnemavgAll += sqrt(wcos2tavg*wcos2tavg+wsin2tavg*wsin2tavg);
	      wavgAll += wavg;
	      ngridptsWithData++;
	    }
	  }
	} // Close loop over all grid points
	nemavgAll /= ngridptsWithData;
	wnemavgAll /= ngridptsWithData;
	wavgAll /= ngridptsWithData;
	alignOrderAvg[n][0] += wavgAll;
	alignOrderAvg[n][1] += nemavgAll;
	alignOrderAvg[n][2] += wnemavgAll;
      } // Close loop over different radii
    } // Close parallel region
  } // Close loop over time
  posReader.close();
  deformReader.close();

  // Averge over time
  for (int n = 0; n < nbins; n++) {
    alignOrderAvg[n][0] /= ntbins;
    alignOrderAvg[n][1] /= ntbins;
    alignOrderAvg[n][2] /= ntbins;    
  }
  
  // Output results
  for (int n = 0; n < nbins; n++) {
    double r = n*rinc+rmin;
    writer << r+rinc*0.5 << " " << alignOrderAvg[n][0] << " "
	   << alignOrderAvg[n][1] << " " << alignOrderAvg[n][2] << "\n";
  }
  writer.close();

  // Clean up
  deleteArray(pos);
  deleteArray(sep);
  deleteArray(gridx);
  deleteArray(gridy);
  deleteArray(alignData);
  deleteArray(alignOrderAvg);
}
  
double ddiff(double len, double x1, double x2) {
  double dx = x1-x2;
  return dx-round(dx/len)*len;
}

double ddist(int ndims, double* len, double* v1, double* v2) {
  double sum = 0.0;
  double dx;
  for (int i = 0; i < ndims; i++) {
    dx = ddiff(len[i], v1[i], v2[i]);
    sum += dx*dx;
  }
  return sqrt(sum);
}

void getEigen(double mxx, double myy, double mxy, Eigen& eigen) {
  eigen.matrix.at(0,0) = mxx;
  eigen.matrix.at(0,1) = mxy;
  eigen.matrix.at(1,0) = mxy;
  eigen.matrix.at(1,1) = myy;
  // Note that eigenvectors are stored in columns, not rows
  eig_sym(eigen.eigenVal, eigen.eigenVec, eigen.matrix);
}
