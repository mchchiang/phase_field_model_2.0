// pair_vel_corr.cpp
// A program to compute pair velocity correlation

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include "position.hpp"
#include "array.hpp"

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
    cout << "Usage: pair_vel_corr npoints lx ly rmin rmax rinc "
	 << "startTime endTime timeInc posFile velFile outFile" << endl;
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
  string velFile (argv[++argi]);
  string outFile (argv[++argi]);

  PositionReader posReader;
  if (!posReader.open(posFile,npoints,lx,ly)) {
    cout << "Error: cannot open the file " << posFile << endl;
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

  // Read the position data and compute separation
  double** pos = create2DArray<double>(npoints,2);
  double** velAdv = create2DArray<double>(npoints,3);
  double** velCM = create2DArray<double>(npoints,3);
  int nbins = static_cast<int>((rmax-rmin)/rinc);
  int ntbins = static_cast<int>((endTime-startTime)/timeInc)+1;
  double* posDist = create1DArray<double>(nbins);
  double* velAdvDist = create1DArray<double>(nbins);
  double* normVelAdvDist = create1DArray<double>(nbins);
  double* velCMDist = create1DArray<double>(nbins);
  double* normVelCMDist = create1DArray<double>(nbins);  
  long t;
  int ibin;
  double dims[2] = {(double) lx, (double) ly};
  double sep, vax, vay, vcmx, vcmy, vadot, vcmdot;
  bool readVel = false;
  bool readPos = false;
  stringstream ss;
  string str, line;
  for (long time = startTime; time <= endTime; time += timeInc) {
    readVel = false;
    readPos = false;
    while (getline(velReader, line)) {
      getline(velReader, line);
      ss.clear();
      ss.str(line);
      ss >> str >> t;
      if (t > endTime) {
	break;
      } else if (t != time) {
	// Skip data at current time frame
	for (int i = 0; i < npoints; i++) {
	  getline(velReader, line);
	}
	continue;
      } else {
	for (int i = 0; i < npoints; i++) {
	  getline(velReader, line);
	  ss.clear();
	  ss.str(line);
	  ss >> vax >> vay >> vcmx >> vcmy;
	  velAdv[i][0] = vax;
	  velAdv[i][1] = vay;
	  velAdv[i][2] = sqrt(vax*vax+vay*vay);	  
	  velCM[i][0] = vcmx;
	  velCM[i][1] = vcmy;
	  velCM[i][2] = sqrt(vcmx*vcmx+vcmy*vcmy);
	}
	readVel = true;
	break;
      }
    }
    while (posReader.nextFrame()) {
      t = posReader.getTime();
      if (t > endTime) {
	break;
      } else if (t != time) {
	continue;
      } else {
	for (int i = 0; i < npoints; i++) {
	  pos[i][0] = posReader.getPosition(i,0);
	  pos[i][1] = posReader.getPosition(i,1);
	}
	readPos = true;
	break;
      }
    }
    if (!readVel || !readPos) break;
    for (int i = 0; i < npoints; i++) {
      for (int j = i+1; j < npoints; j++) {
	sep = ddist(2, dims, pos[i], pos[j]);
	if (sep >= rmin && sep < rmax) {
	  ibin = static_cast<int>((sep-rmin)/rinc);
	  vadot = velAdv[i][0]*velAdv[j][0]+velAdv[i][1]*velAdv[j][1];
	  velAdvDist[ibin] += 2.0*vadot;
	  normVelAdvDist[ibin] += 2.0*vadot/(velAdv[i][2]*velAdv[j][2]);
	  vcmdot = velCM[i][0]*velCM[j][0]+velCM[i][1]*velCM[j][1];
	  velCMDist[ibin] += 2.0*vcmdot;
	  normVelCMDist[ibin] += 2.0*vcmdot/(velCM[i][2]*velCM[j][2]);
	  posDist[ibin] += 2.0;
	}
      }
    }
  }
  posReader.close();
  velReader.close();

  // Normalise and output
  double rho = npoints/static_cast<double>(lx*ly);
  double r, norm;
  double twopi = 2.0*M_PI;
  for (int i = 0; i < nbins; i++) {
    r = rmin+rinc*i;
    norm = twopi*r*rinc*npoints*ntbins*rho;
    posDist[i] /= norm;
    velAdvDist[i] /= norm;
    normVelAdvDist[i] /= norm;
    velCMDist[i] /= norm;
    normVelCMDist[i] /= norm;    
    writer << r+rinc*0.5 << " "
	   << velAdvDist[i] << " " << normVelAdvDist[i] << " "
	   << velCMDist[i] << " " << normVelCMDist[i] << " "
	   << posDist[i] << "\n";
  }
  writer.close();
  
  // Clean up
  deleteArray(pos);
  deleteArray(velAdv);
  deleteArray(velCM);
  deleteArray(velAdvDist);
  deleteArray(normVelAdvDist);
  deleteArray(velCMDist);
  deleteArray(normVelCMDist);  
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
