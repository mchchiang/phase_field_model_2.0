// hexatic_local_order.cpp
// A program to compute the local correlation of the hexatic order parameter

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include "array.hpp"
#include "position.hpp"

using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::stringstream;
using std::string;

// Helper functions
double diff(double len, double x1, double x2);
double dist(double x, double y);
void error(int n, double avg, double avgSq, double& dev, double& err);

int main(int argc, char* argv[]) {
  
  if (argc != 15) {
    cout << "usage: hexatic_local_order npoints lx ly ngridx ngridy "
	 << "rmin rmax rinc startTime endTime timeInc posFile hexFile outFile"
	 << endl;
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

  bool foundPosData, foundHexData;
  const int ntimepts = static_cast<int>((endTime-startTime)/timeInc)+1;
  const int nbins = static_cast<int>((rmax-rmin)/rinc)+1;
  int ngridptsWithData;
  double*** distsqMat = create3DArray<double>(ngridx, ngridy, npoints);
  double* gridx = create1DArray<double>(ngridx);
  double* gridy = create1DArray<double>(ngridy);
  double** hexOrder = create2DArray<double>(npoints,2);
  double** pos = create2DArray<double>(npoints,2);
  double* magHexOrderAvg = create1DArray<double>(nbins);
  double* magHexOrderAvgSq = create1DArray<double>(nbins);
  double hexx, hexy, hexmag, proj;
  long t;
  stringstream ss;
  string line, str;

  // Compute grid positions
  double gridxWidth = lx/static_cast<double>(ngridx);
  double gridyWidth = ly/static_cast<double>(ngridy);
  for (int i = 0; i < ngridx; i++) {
    gridx[i] = (i+0.5)*gridxWidth;
  }
  for (int i = 0; i < ngridy; i++) {
    gridy[i] = (i+0.5)*gridyWidth;
  }

  for (long time = startTime; time <= endTime; time += timeInc) {
    // Get position data for this time frame
    foundPosData= false;
    foundHexData = false;
    while (posReader.nextFrame()) {
      t = posReader.getTime();
      if (t == time) {
	foundPosData = true;
	break;
      } else if (t > time) {
	foundPosData = false;
	break;
      }
    }
    if (!foundPosData) {
      cout << "Error: cannot find the position data for time = "
	   << time << endl;
    }
    // Get hexatic order data for this time frame
    while (getline(hexReader, line)) {
      getline(hexReader, line);
      ss.clear();
      ss.str(line);
      ss >> str >> t;
      if (t < time) {
	for (int n = 0; n < npoints; n++) {
	  getline(hexReader, line);
	}
      } else if (t == time) {
	for (int n = 0; n < npoints; n++) {
	  getline(hexReader, line);
	  ss.clear();
	  ss.str(line);
	  ss >> hexx >> hexy >> hexmag >> proj;
	  hexOrder[n][0] = hexx;
	  hexOrder[n][1] = hexy;
	}
	foundHexData = true;
	break;
      } else { // t > time
	foundHexData = false;
	break;
      }
    }    
    if (!foundHexData) {
      cout << "Error: cannot find the hexatic order data for time = "
	   << time << endl;
    }

    // Compute distance matrix
    double xg, yg, xp, yp, dx, dy;
    for (int i = 0; i < ngridx; i++) {
      xg = gridx[i];
      for (int j = 0; j < ngridy; j++) {
	yg = gridy[j];
	for (int k = 0; k < npoints; k++) {
	  xp = posReader.getPosition(k,0);
	  yp = posReader.getPosition(k,1);
	  dx = diff(lx,xp,xg);
	  dy = diff(ly,yp,yg);
	  distsqMat[i][j][k] = dx*dx+dy*dy;
	}
      }
    }

    // Compute local average of hexatic order for each radius
    int count;
    double rsq, magHexOrder, hexxavg, hexyavg;
    for (int n = 0; n < nbins; n++) {
      rsq = n*rinc+rmin;
      rsq *= rsq;
      magHexOrder = 0.0;
      ngridptsWithData = 0;
      for (int i = 0; i < ngridx; i++) {
	for (int j = 0; j < ngridy; j++) {
	  hexxavg = 0.0;
	  hexyavg = 0.0;
	  count = 0;
	  for (int k = 0; k < npoints; k++) {
	    if (distsqMat[i][j][k] < rsq) {
	      hexxavg += hexOrder[k][0];
	      hexyavg += hexOrder[k][1];
	      count++;
	    }
	  }
	  // Compute hexatic order
	  if (count > 0) {
	    hexxavg /= count;
	    hexyavg /= count;
	    magHexOrder += dist(hexxavg,hexyavg);
	    ngridptsWithData++;
	  }
	}
      } // Close loop over all grid points
      if (ngridptsWithData > 0) {
	magHexOrder /= ngridptsWithData;
	magHexOrderAvg[n] += magHexOrder;
	magHexOrderAvgSq[n] += magHexOrder*magHexOrder;
      }
    } // Close loop over different radii
  } // Close loop over time
  
  posReader.close();
  hexReader.close();

  // Average over time
  for (int n = 0; n < nbins; n++) {
    magHexOrderAvg[n] /= ntimepts;
    magHexOrderAvgSq[n] /= ntimepts;
  }

  // Output results
  ofstream writer;
  writer.open(outFile);
  if (!writer) {
    cout << "Error: cannot open the file " << outFile << endl;
    return 1;
  }

  double stdev = 0.0;
  double stderr = 0.0;
  for (int n = 0; n < nbins; n++) {
    writer << (n*rinc+rmin) << " ";
    error(ntimepts, magHexOrderAvg[n], magHexOrderAvgSq[n], stdev, stderr);
    writer << magHexOrderAvg[n] << " " << stdev << " " << stderr << "\n";
  }
  writer.close();
  
  // Clean up
  deleteArray(distsqMat);
  deleteArray(pos);
  deleteArray(hexOrder);
  deleteArray(gridx);
  deleteArray(gridy);
  deleteArray(magHexOrderAvg);
  deleteArray(magHexOrderAvgSq);
}


double diff(double len, double x1, double x2) {
  double dx = x1-x2;
  return dx-round(dx/len)*len;
}

double dist(double x, double y) {
  return sqrt(x*x+y*y);
}

void error(int n, double avg, double avgSq, double& dev, double& err) {
  if (n > 1) {
    dev = sqrt(n/(n-1.0)*(avgSq-avg*avg));
    err = dev/sqrt(n);
  } else {
    dev = 0.0;
    err = 0.0;
  }
}
