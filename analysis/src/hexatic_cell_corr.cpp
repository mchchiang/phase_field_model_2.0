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
using std::vector;

double diff(double len, double x1, double x2);
double dist(double x, double y);
double dot(double x1, double y1, double x2, double y2);

int main(int argc, char* argv[]) {
  
  if (argc != 13) {
    cout << "usage: hexatic_cell_corr npoints lx ly rmin rmax rinc startTime "
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

  // Read the position data
  int nbins = static_cast<int>((endTime-startTime)/timeInc)+1;
  double*** pos = create3DArray<double>(nbins,npoints,2);
  long time;
  int ibin;
  while (posReader.nextFrame()) {
    time = posReader.getTime();
    if (time > endTime) {
      break;
    } else if (time < startTime || (time-startTime)%timeInc != 0) {
      continue;
    } else {
      ibin = static_cast<int>((time-startTime)/timeInc);
      for (int i = 0; i < npoints; i++) {
	// Use wrapped position here as we only care about relative distances
	pos[ibin][i][0] = posReader.getPosition(i,0);
	pos[ibin][i][1] = posReader.getPosition(i,1);
      }
    }
  }
  posReader.close();
  
  ifstream reader;
  reader.open(hexFile);
  if (!reader) {
    cout << "Error: cannot open the file " << hexFile << endl;
    return 1;
  }

  ofstream writer;
  writer.open(outFile);
  if (!writer) {
    cout << "Error: canont open the file " << outFile << endl;
    return 1;
  }
  
  int irbin;
  int nrbins = static_cast<int>(ceil((rmax-rmin)/rinc));
  int* distrbCount = create1DArray<int>(nrbins);
  double* hexDistrb = create1DArray<double>(nrbins);
  double** hexOrder = create2DArray<double>(npoints,2);

  double hexx, hexy, mag, proj;
  double xi, xj, yi, yj, dr;
  double hexxi, hexxj, hexyi, hexyj, corrx, corry;
  string line, str;
  stringstream ss;
  while (getline(reader, line)) {
    // Read header lines
    if (!getline(reader, line)) break;
    ss.clear();
    ss.str(line);
    ss >> str >> time;
    if (time > endTime) {
      break;
    } else if (time < startTime || (time-startTime)%timeInc != 0) {
      // Skip data before start time
      for (int i = 0; i < npoints; i++) {
	getline(reader, line);
      }
    } else {
      // Read the hexatic order for a single time frame
      ibin = static_cast<int>((time-startTime)/timeInc);
      for (int i = 0; i < npoints; i++) {
	getline(reader, line);
	ss.clear();
	ss.str(line);
	ss >> hexx >> hexy >> mag >> proj;
	hexOrder[i][0] = hexx;
	hexOrder[i][1] = hexy;
      }
      // Compute the correlation
      for (int i = 0; i < npoints; i++) {
	xi = pos[ibin][i][0];
	yi = pos[ibin][i][1];
	hexxi = hexOrder[i][0];
	hexyi = hexOrder[i][1];
	for (int j = 0; j <= i; j++) {
	  xj = pos[ibin][j][0];
	  yj = pos[ibin][j][1];
	  dr = dist(diff(lx,xi,xj), diff(ly,yi,yj));
	  irbin = static_cast<int>(floor((dr-rmin)/rinc));
	  if (irbin < 0 || irbin > nrbins) {
	    cout << "Distance out of range: dr = " << dr << endl;
	    continue;
	  }
	  hexxj = hexOrder[j][0];
	  hexyj = hexOrder[j][1];
	  corrx = hexxi*hexxj+hexyi*hexyj;
	  corry = hexxj*hexyi-hexxi*hexyj;
	  hexDistrb[irbin] += dist(corrx,corry);
	  distrbCount[irbin]++;
	}
      }
    }
  }
  reader.close();

  // Output distribution
  double left, centre, right;
  for (int i = 0; i < nrbins; i++) {
    // Normalise results
    if (distrbCount[i] > 0) {
      hexDistrb[i] /= distrbCount[i];
    }
    left = i*rinc+rmin;
    right = (i+1)*rinc+rmin;
    centre = (left+right)*0.5;
    writer << left << " " << centre << " " << right << " " 
	   << hexDistrb[i] << " " 
	   << distrbCount[i]/(2.0*M_PI*left*rinc)/nbins << endl;
  }
  writer.close();

  // Clean up
  deleteArray(pos);
  deleteArray(hexOrder);
}


double diff(double len, double x1, double x2) {
  double dx = x1-x2;
  return dx-round(dx/len)*len;
}

double dist(double x, double y) {
  return sqrt(x*x+y*y);
}

double dot(double x1, double y1, double x2, double y2) {
  return x1*x2+y1*y2;
}
