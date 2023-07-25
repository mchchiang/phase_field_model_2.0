// radial_distrb.cpp
// A program to compute the radial distribution function g(r)

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include "position.hpp"
#include "array.hpp"

using std::cout;
using std::endl;
using std::ofstream;
using std::string;

double ddiff(double len, double x1, double x2);
double ddist(int ndims, double* len, double* v1, double* v2);

int main(int argc, char* argv[]) {
  if (argc != 12) {
    cout << "Usage: radial_distrb npoints lx ly rmin rmax rinc "
	 << "startTime endTime timeInc posFile outFile" << endl;
    return 1;
  }

  int argi = 0;
  int npoints = stoi(string(argv[++argi]), nullptr, 10);
  int lx = stoi(string(argv[++argi]), nullptr, 10);
  int ly = stoi(string(argv[++argi]), nullptr, 10);
  double rmin = stod(string(argv[++argi]), nullptr);
  double rmax = stod(string(argv[++argi]), nullptr);
  double rinc = stod(string(argv[++argi]), nullptr);
  long startTime = stol(string(argv[++argi]), nullptr, 10);
  long endTime = stol(string(argv[++argi]), nullptr, 10);
  long timeInc = stol(string(argv[++argi]), nullptr, 10);
  string posFile (argv[++argi]);
  string outFile (argv[++argi]);

  PositionReader reader;
  if (!reader.open(posFile,npoints,lx,ly)) {
    cout << "Error: cannot open the file " << posFile << endl;
    return 1;
  }
  
  ofstream writer;
  writer.open(outFile);
  if (!writer) {
    cout << "Error: cannot open the file " << outFile << endl;
    return 1;
  }

  // Read the position data and compute separation
  double** pos = create2DArray<double>(npoints,2);
  int nbins = static_cast<int>((rmax-rmin)/rinc);
  int ntbins = static_cast<int>((endTime-startTime)/timeInc)+1;
  double* radialDist = create1DArray<double>(nbins);
  long time;
  int ibin;
  double dims[2] = {(double) lx, (double) ly};
  double sep;
  while (reader.nextFrame()) {
    time = reader.getTime();
    if (time < startTime) {
      continue;
    } else if (time > endTime) {
      break;
    } else if ((time-startTime) % timeInc == 0) {
      for (int i = 0; i < npoints; i++) {
	pos[i][0] = reader.getPosition(i,0);
	pos[i][1] = reader.getPosition(i,1);
      }
      for (int i = 0; i < npoints; i++) {
	for (int j = i+1; j < npoints; j++) {
	  sep = ddist(2, dims, pos[i], pos[j]);
	  if (sep >= rmin && sep < rmax) {
	    ibin = static_cast<int>((sep-rmin)/rinc);
	    radialDist[ibin] += 2;
	  }
	}
      }
    }
  }
  reader.close();

  // Normalise and output
  double rho = npoints/static_cast<double>(lx*ly);
  double r;
  double twopi = 2.0*M_PI;
  for (int i = 0; i < nbins; i++) {
    r = rmin+rinc*i;
    radialDist[i] /= (twopi*r*rinc*npoints*ntbins*rho);
    writer << r+rinc*0.5 << " " << radialDist[i] << "\n";
  }
  writer.close();
  
  // Clean up
  deleteArray(pos);
  deleteArray(radialDist);
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
