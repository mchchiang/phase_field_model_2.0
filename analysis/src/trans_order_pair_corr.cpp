// trans_order_pair_corr.cpp
// A program to compute the translational order parameter
// \psi_T = 1/N \sum_j \exp(i\vec{G}\cdot\vec{r})
// where \vec{G} is a reciprocal lattice vector
// We consider a triangular lattice with spacing w = 2R in real space
// and consider the reciprocal lattice vectors:
// \vec{G} = 2\pi/w (1,1/sqrt{3})
// \vec{G} = 2\pi/w (0,2/sqrt{3})

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include "array.hpp"
#include "position.hpp"

using std::cout;
using std::endl;
using std::ofstream;
using std::string;

double dist(double* x);
double dot(double* x1, double* y1);
double ddiff(double len, double x1, double x2);
double ddist(int ndims, double* len, double* v1, double* v2);

int main(int argc, char* argv[]) {
  
  if (argc != 13) {
    cout << "Usage: trans_order_pair_corr npoints lx ly r0 rmin rmax rinc "
	 << "startTime endTime timeInc posFile outFile" << endl;
    return 1;
  }

  int argi = 0;
  int npoints = stoi(string(argv[++argi]), nullptr, 10);
  int lx = stoi(string(argv[++argi]), nullptr, 10);
  int ly = stoi(string(argv[++argi]), nullptr, 10);
  double r0 = stod(string(argv[++argi]), nullptr);
  double rmin = stod(string(argv[++argi]), nullptr);
  double rmax = stod(string(argv[++argi]), nullptr);
  double rinc = stod(string(argv[++argi]), nullptr);
  long startTime = stoi(string(argv[++argi]), nullptr, 10);
  long endTime = stoi(string(argv[++argi]), nullptr, 10);
  long timeInc = stoi(string(argv[++argi]), nullptr, 10);
  string posFile (argv[++argi]);
  string outFile (argv[++argi]);
  
  PositionReader reader;
  if (!reader.open(posFile, npoints, lx, ly)) {
    cout << "Error: cannot open the file " << posFile << endl;
    return 1;
  }

  ofstream writer;
  writer.open(outFile);
  if (!writer) {
    cout << "Error: cannot open the file " << outFile << endl;
    return 1;
  }
  
  // Read the position data
  int nbins = static_cast<int>((rmax-rmin)/rinc);
  int ntbins = static_cast<int>((endTime-startTime)/timeInc)+1;
  long time;
  double prefact = M_PI/r0;
  double root3 = sqrt(3.0);
  double qvec[2][2] = {{prefact,prefact/root3},{0.0,2.0*prefact/root3}};
  double** pos = create2DArray<double>(npoints,2);
  double*** trans = create3DArray<double>(npoints,2,2);
  double* posDist = create1DArray<double>(nbins);
  double*** transDist = create3DArray<double>(nbins,2,2);
  double dims[2] = {(double) lx, (double) ly};
  double qr, sep;
  int ibin;  
  while (reader.nextFrame()) {
    time = reader.getTime();
    if (time < startTime) {
      continue;
    } else if (time > endTime) {
      break;
    } else if ((time-startTime) % timeInc == 0) {
      for (int i = 0; i < npoints; i++) {
	// Use wrapped position here as we only care about relative distances
	// and orientations
	pos[i][0] = reader.getPosition(i,0);
	pos[i][1] = reader.getPosition(i,1);
	for (int j = 0; j < 2; j++) {
	  qr = dot(qvec[j], pos[i]);
	  trans[i][j][0] = cos(qr);
	  trans[i][j][1] = sin(qr);
	}
      }
      // Compute correlation
      for (int i = 0; i < npoints; i++) {
	for (int j = i+1; j < npoints; j++) {
	  sep = ddist(2, dims, pos[i], pos[j]);
	  if (sep >= rmin && sep < rmax) {
	    ibin = static_cast<int>((sep-rmin)/rinc);
	    for (int k = 0; k < 2; k++) {
	      transDist[ibin][k][0] += 2.0*(trans[i][k][0]*trans[j][k][0] +
					    trans[i][k][1]*trans[j][k][1]);
	      transDist[ibin][k][1] += 2.0*(trans[i][k][1]*trans[j][k][0] -
					    trans[i][k][0]*trans[j][k][1]);
	      posDist[ibin] += 2.0;
	    }
	  }
	}
      }
    }
  } // Close loop over different time frames
  reader.close();

  // Normalise and output
  double rho = npoints/static_cast<double>(lx*ly);
  double r, norm;
  double twopi = 2.0*M_PI;
  for (int i = 0; i < nbins; i++) {
    r = rmin+rinc*i;
    norm = twopi*r*rinc*npoints*ntbins*rho;
    writer << r+rinc*0.5 << " ";    
    for (int j = 0; j < 2; j++) {
      transDist[i][j][0] /= norm;
      transDist[i][j][1] /= norm;
      writer << transDist[i][j][0] << " " << transDist[i][j][1] << " "
	     << dist(transDist[i][j]) << " ";
    }
    posDist[i] /= norm;
    writer << posDist[i] << "\n";
  }
  writer.close();

  // Clean up
  deleteArray(pos);
  deleteArray(trans);
  deleteArray(posDist);
  deleteArray(transDist);
}

inline double dist(double* x) {
  return sqrt(x[0]*x[0]+x[1]*x[1]);
}

inline double dot(double* x1, double* x2) {
  return x1[0]*x2[0]+x1[1]*x2[1];
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
