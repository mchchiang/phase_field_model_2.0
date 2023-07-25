// hexatic.cpp
// A program to compute the hexatic order parameter

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

// Helper functions
double diff(double len, double x1, double x2);
double dist(double x, double y);
double dot(double x1, double y1, double x2, double y2);

int main(int argc, char* argv[]) {
  
  if (argc != 11) {
    cout << "Usage: hexatic npoints lx ly startTime endTime timeInc "
	 << "posFile neighFile cellOutFile avgOutFile" << endl;
    return 1;
  }

  int argi = 0;
  int npoints = stoi(string(argv[++argi]), nullptr, 10);
  int lx = stoi(string(argv[++argi]), nullptr, 10);
  int ly = stoi(string(argv[++argi]), nullptr, 10);
  long startTime = stoi(string(argv[++argi]), nullptr, 10);
  long endTime = stoi(string(argv[++argi]), nullptr, 10);
  long timeInc = stoi(string(argv[++argi]), nullptr, 10);
  string posFile (argv[++argi]);
  string neighFile (argv[++argi]);
  string cellOutFile (argv[++argi]);
  string avgOutFile (argv[++argi]);
  
  PositionReader posReader;
  if (!posReader.open(posFile, npoints, lx, ly)) {
    cout << "Error: cannot open the file " << posFile << endl;
    return 1;
  }

  // Read the position data
  int nbins = static_cast<int>((endTime-startTime)/timeInc)+1;
  double*** pos = create3DArray<double>(nbins, npoints, 2);
  long time;
  int ibin;
  while (posReader.nextFrame()) {
    time = posReader.getTime();
    if (time < startTime) {
      continue;
    } else if (time > endTime) {
      break;
    } else if ((time-startTime) % timeInc == 0) {
      ibin = static_cast<int>((time-startTime)/timeInc);
      for (int i = 0; i < npoints; i++) {
	// Use wrapped position here as we only care about relative distances
	// and orientations
	pos[ibin][i][0] = posReader.getPosition(i,0);
	pos[ibin][i][1] = posReader.getPosition(i,1);
      }
    }
  }
  posReader.close();
  
  // Read the neighbour data
  vector<int> neighvec;
  neighvec.reserve(6); // Reserve space for at least 6 nearest neighbours
  vector<vector<int> > neighvecs (npoints, neighvec);
  vector<vector<vector<int> > > neighbours (nbins, neighvecs);
  ifstream neighReader;
  neighReader.open(neighFile);
  if (!neighReader) {
    cout << "Error: cannot open the file " << neighFile << endl;
    return 1;
  }
  string line, str;
  stringstream ss;
  int neighIndex;
  while (getline(neighReader, line)) {
    // Read time
    getline(neighReader, line);
    ss.clear();
    ss.str(line);
    ss >> str >> time;
    if (time > endTime) {
      break;
    } else if (time < startTime || (time-startTime)%timeInc != 0) {
      // Skip over unused data
      for (int i = 0; i < npoints; i++) {
	getline(neighReader, line);
      }
    } else {
      ibin = static_cast<int>((time-startTime)/timeInc);
      for (int i = 0; i < npoints; i++) {
	getline(neighReader, line);
	ss.clear();
	ss.str(line);
	while (ss) {
	  ss >> neighIndex;
	  if (!ss) break;
	  neighbours[ibin][i].push_back(neighIndex);
	}
      }
    }
  }
  neighReader.close();
  
  ofstream cellWriter;
  cellWriter.open(cellOutFile);
  if (!cellWriter) {
    cout << "Error: cannot open the file " << cellOutFile << endl;
    return 1;
  }
  ofstream avgWriter;
  avgWriter.open(avgOutFile);
  if (!avgWriter) {
    cout << "Error: cannot open the file " << avgOutFile << endl;
    return 1;
  }

  // Compute the local hexatic order for each cell at each time frame
  // phi_6i = 1/(N_i)\sum_{j=1}^{N_i} exp(i6\theta_{ij})
  double** hexOrder = create2DArray<double>(npoints,2);
  double avgHexOrder[2];
  double x, y, dx, dy, theta, magHexOrder, avgMagHexOrder;
  int numOfNeighs, ineigh;
  for (ibin = 0; ibin < nbins; ibin++) {
    time = ibin*timeInc+startTime;
    avgHexOrder[0] = 0.0;
    avgHexOrder[1] = 0.0;
    avgMagHexOrder = 0.0;
    for (int i = 0; i < npoints; i++) {
      hexOrder[i][0] = 0.0;
      hexOrder[i][1] = 0.0;
      numOfNeighs = neighbours[ibin][i].size();
      x = pos[ibin][i][0];
      y = pos[ibin][i][1];
      for (int j = 0; j < numOfNeighs; j++) {
	ineigh = neighbours[ibin][i][j];
	dx = diff(lx, x, pos[ibin][ineigh][0]);
	dy = diff(ly, y, pos[ibin][ineigh][1]);
	theta = atan2(-dy,-dx)+M_PI;
	hexOrder[i][0] += cos(6.0*theta);
	hexOrder[i][1] += sin(6.0*theta);
      }
      hexOrder[i][0] /= numOfNeighs;
      hexOrder[i][1] /= numOfNeighs;
      magHexOrder = dist(hexOrder[i][0], hexOrder[i][1]);
      avgMagHexOrder += magHexOrder;
      avgHexOrder[0] += hexOrder[i][0];
      avgHexOrder[1] += hexOrder[i][1];  
    }
    avgHexOrder[0] /= npoints;
    avgHexOrder[1] /= npoints;
    avgMagHexOrder /= npoints; 
    
    double magAvgHexOrder = dist(avgHexOrder[0], avgHexOrder[1]);

    // Write individual cell data
    cellWriter << "Cells: " << npoints << "\n";
    cellWriter << "Timestep: " << time << "\n";
    for (int i = 0; i < npoints; i++) {
      // Compute projection to sample/global hexatic order
      magHexOrder = dist(hexOrder[i][0], hexOrder[i][1]);
      double projection = dot(hexOrder[i][0], hexOrder[i][1], avgHexOrder[0], 
			      avgHexOrder[1])/ (magHexOrder*magAvgHexOrder);
      cellWriter << hexOrder[i][0] << " " << hexOrder[i][1] << " " 
		 << magHexOrder << " " << projection << "\n";
    }
    
    // Write average data
    avgWriter << time << " " << avgHexOrder[0] << " " << avgHexOrder[1] << " " 
	      << magAvgHexOrder << " " << avgMagHexOrder << "\n";
  } // Close loop over ibin 

  cellWriter.close();
  avgWriter.close();

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
