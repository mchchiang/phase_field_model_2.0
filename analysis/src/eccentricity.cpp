// eccentricity.cpp
// A program to compute the average eccentricity using the gyration tensor

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

double eccent(double gxx, double gyy, double gxy);

int main(int argc, char* argv[]) {
  
  if (argc != 7) {
    cout << "Usage: eccentricity npoints startTime endTime timeInc "
	 << "gyrFile outFile" << endl;
    return 1;
  }

  int argi = 0;
  int npoints = stoi(string(argv[++argi]), nullptr, 10);
  long startTime = stoi(string(argv[++argi]), nullptr, 10);
  long endTime = stoi(string(argv[++argi]), nullptr, 10);
  long timeInc = stoi(string(argv[++argi]), nullptr, 10);
  string gyrFile (argv[++argi]);
  string outFile (argv[++argi]);
  
  ifstream reader;
  reader.open(gyrFile);
  if (!reader) {
    cout << "Error: cannot open the file " << gyrFile << endl;
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
  double gxx, gyy, gxy, b, eccentAvg, eccentAvgSq;
  while (getline(reader, line)) {
    // Read the two header lines and get time
    getline(reader, line);
    ss.clear();
    ss.str(line);
    ss >> str >> time;
    
    // Only use the data from the specified time period
    if (time > endTime) {
      break;
    } else if (time < startTime || (time-startTime)%timeInc != 0) {
      // Skip the data in that time frame
      for (int i = 0; i < npoints; i++) {
	getline(reader, line);
      }
    } else {
      // Compute eccentricity
      eccentAvg = 0.0;
      eccentAvgSq = 0.0;
      for (int i = 0; i < npoints; i++) {
	getline(reader, line);
	ss.clear();
	ss.str(line);
	ss >> gxx >> gyy >> gxy;
	b = eccent(gxx, gyy, gxy);
	eccentAvg += b;
	eccentAvgSq += b*b;
      }
      
      // Normalise
      eccentAvg /= npoints;
      eccentAvgSq /= npoints;
      double var = npoints/(npoints-1.0)*(eccentAvgSq-eccentAvg*eccentAvg);
      double stdev = sqrt(var);
      double stderr = stdev/sqrt(npoints);
      
      // Output results
      writer << time << " " << eccentAvg << " " << stdev << " " << stderr
	     << "\n";
    }
  }
  reader.close();
  writer.close();
}

double eccent(double gxx, double gyy, double gxy) {
  mat matrix = mat(2,2);
  vec eigenVal = vec(2);
  mat eigenVec = mat(2,2);
  matrix.at(0,0) = gxx;
  matrix.at(0,1) = gxy;
  matrix.at(1,0) = gxy;
  matrix.at(1,1) = gyy;
  // Note that eigenvectors are stored in columns, not rows
  eig_sym(eigenVal, eigenVec, matrix);
  double lam0 = eigenVal.at(0);
  double lam1 = eigenVal.at(1);
  return fabs(lam0-lam1)/(lam0+lam1);
}
