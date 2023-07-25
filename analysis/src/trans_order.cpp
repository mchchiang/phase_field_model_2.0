// trans_order.cpp
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
#include "position.hpp"

using std::cout;
using std::endl;
using std::ofstream;
using std::string;

double dist(double x[2]);
double dot(double x1[2], double y1[2]);

int main(int argc, char* argv[]) {
  
  if (argc != 10) {
    cout << "Usage: trans_order npoints lx ly r0 startTime endTime timeInc "
	 << "posFile outFile" << endl;
    return 1;
  }

  int argi = 0;
  int npoints = stoi(string(argv[++argi]), nullptr, 10);
  int lx = stoi(string(argv[++argi]), nullptr, 10);
  int ly = stoi(string(argv[++argi]), nullptr, 10);
  double r0 = stod(string(argv[++argi]), nullptr);
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
  long time;
  double prefact = M_PI/r0;
  double root3 = sqrt(3.0);
  double qvec[2][2] = {{prefact,prefact/root3},{0.0,2.0*prefact/root3}};
  double csqr[2][2]; // cos(qr) and sin(qr) for both q vectors
  double r[2];
  double qr;
  
  while (reader.nextFrame()) {
    time = reader.getTime();
    if (time < startTime) {
      continue;
    } else if (time > endTime) {
      break;
    } else if ((time-startTime) % timeInc == 0) {
      csqr[0][0] = 0.0;
      csqr[0][1] = 0.0;      
      csqr[1][0] = 0.0;
      csqr[1][1] = 0.0;      
      for (int i = 0; i < npoints; i++) {
	// Use wrapped position here as we only care about relative distances
	// and orientations
	r[0] = reader.getPosition(i,0);
	r[1] = reader.getPosition(i,1);
	qr = dot(qvec[0],r);
	csqr[0][0] += cos(qr);
	csqr[0][1] += sin(qr);
	qr = dot(qvec[1],r);
	csqr[1][0] += cos(qr);
	csqr[1][1] += sin(qr);	
      }
      csqr[0][0] /= npoints;
      csqr[0][1] /= npoints;      
      csqr[1][0] /= npoints;
      csqr[1][1] /= npoints;
      writer << time << " " << dist(csqr[0]) << " " << dist(csqr[1]) << "\n";
    }
  }
  reader.close();
  writer.close();
}

inline double dist(double x[2]) {
  return sqrt(x[0]*x[0]+x[1]*x[1]);
}

inline double dot(double x1[2], double x2[2]) {
  return x1[0]*x2[0]+x1[1]*x2[1];
}
