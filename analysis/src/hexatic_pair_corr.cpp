// hexatic_cell_corr.cpp
// A program to compute the correlation of the hexatic order parameter

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include "array.hpp"
#include "position.hpp"

using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::stringstream;
using std::string;

double ddiff(double len, double x1, double x2);
double ddist(int ndims, double* len, double* v1, double* v2);

int main(int argc, char* argv[]) {
  
  if (argc != 13) {
    cout << "usage: hexatic_pair_corr npoints lx ly rmin rmax rinc startTime "
	 << "endTime timeInc posFile hexFile outFile" << endl;
    return 1;
  }

  int argi = 0;
  int npoints = stoi(string(argv[++argi]), nullptr, 10);
  int lx = stoi(string(argv[++argi]), nullptr, 10);
  int ly = stoi(string(argv[++argi]), nullptr, 10);
  double rmin = stod(string(argv[++argi]), nullptr);
  double rmax = stod(string(argv[++argi]), nullptr);
  double rinc = stod(string(argv[++argi]), nullptr);
  long startTime = stoi(string(argv[++argi]), nullptr, 10);
  long endTime = stoi(string(argv[++argi]), nullptr, 10);
  long timeInc = stoi(string(argv[++argi]), nullptr, 10);
  string posFile (argv[++argi]);
  string hexFile (argv[++argi]);
  string outFile (argv[++argi]);
  
  PositionReader posReader;
  if (!posReader.open(posFile, npoints, lx, ly)) {
    cout << "Error: cannot open the file " << posFile << endl;
    return 1;
  }

  ifstream hexReader;
  hexReader.open(hexFile);
  if (!hexReader) {
    cout << "Error: cannot open the file " << hexFile << endl;
    return 1;
  }

  ofstream writer;
  writer.open(outFile);
  if (!writer) {
    cout << "Error: canont open the file " << outFile << endl;
    return 1;
  }

  int nbins = static_cast<int>((rmax-rmin)/rinc);
  int ntbins = static_cast<int>((endTime-startTime)/timeInc)+1;
  double** pos = create2DArray<double>(npoints,2);
  double** hexDist = create2DArray<double>(nbins,2);
  double* posDist = create1DArray<double>(nbins);
  double** hex = create2DArray<double>(npoints,2);
  double dims[2] = {(double) lx, (double) ly};
  double hexx, hexy, mag, proj, sep;
  long t;
  int ibin;
  bool foundPosData = false;
  bool foundHexData = false;
  stringstream ss;
  string str, line;
  for (long time = startTime; time <= endTime; time += timeInc) {
    foundPosData = false;
    foundHexData = false;
    // Read position data
    while (posReader.nextFrame()) {
      t = posReader.getTime();
      if (t > endTime) {
	break;
      } else if (t != time) {
	continue;
      } else {
	for (int i = 0; i < npoints; i++) {
	// Use wrapped position here as we only care about relative distances
	  pos[i][0] = posReader.getPosition(i,0);
	  pos[i][1] = posReader.getPosition(i,1);
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
    
    // Read hexatic order data
    while (getline(hexReader, line)) {
      getline(hexReader, line);
      ss.clear();
      ss.str(line);
      ss >> str >> t;
      if (t > endTime) {
	break;
      } else if (t != time) {
      // Skip data before start time
	for (int i = 0; i < npoints; i++) {
	  getline(hexReader, line);
	}
      } else {
	for (int i = 0; i < npoints; i++) {
	  getline(hexReader, line);
	  ss.clear();
	  ss.str(line);
	  ss >> hexx >> hexy >> mag >> proj;
	  hex[i][0] = hexx;
	  hex[i][1] = hexy;
	}
	foundHexData = true;
	break;
      }
    }
    if (!foundHexData) {
      cout << "Error: cannot find the hexatic data for time = "
	   << time << endl;
      return 1;
    }
    
    // Compute pairwise correlation
    for (int i = 0; i < npoints; i++) {
      for (int j = i+1; j < npoints; j++) {
	sep = ddist(2, dims, pos[i], pos[j]);
	if (sep >= rmin && sep < rmax) {
	  ibin = static_cast<int>((sep-rmin)/rinc);
	  hexDist[ibin][0] += 2.0*(hex[i][0]*hex[j][0]+hex[i][1]*hex[j][1]);
	  hexDist[ibin][1] += 2.0*(hex[i][1]*hex[j][0]-hex[i][0]*hex[j][1]);
	  posDist[ibin] += 2.0;
	}
      }
    }
  } // Close loop over different time frames
  posReader.close();
  hexReader.close();

  // Normalise and output
  double rho = npoints/static_cast<double>(lx*ly);
  double r, norm;
  double twopi = 2.0*M_PI;
  for (int i = 0; i < nbins; i++) {
    r = rmin+rinc*i;
    norm = twopi*r*rinc*npoints*ntbins*rho;
    hexDist[i][0] /= norm;
    hexDist[i][1] /= norm;
    posDist[i] /= norm;
    writer << r+rinc*0.5 << " " << hexDist[i][0] << " " << hexDist[i][1] << " "
	   << sqrt(hexDist[i][0]*hexDist[i][0]+hexDist[i][1]*hexDist[i][1])
	   << " " << posDist[i] << "\n";
  }
  writer.close();

  // Clean up
  deleteArray(pos);
  deleteArray(hex);
  deleteArray(hexDist);
  deleteArray(posDist);
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
