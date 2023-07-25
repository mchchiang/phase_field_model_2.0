// self_int_scatter.cpp
// A code that computes the self intermediate scattering function from the
// trajectory file

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <omp.h>
#include "position.hpp"
#include "array.hpp"

using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::stringstream;
using std::string;

int main (int argc, char* argv[]) {
  
  if (argc != 13) {
    cout << "Usage: self_int_scatter npoints lx ly nqvecs r0 startTime "
	 << "endTime timeInc endShiftTime posFile posBulkFile outFile" << endl;
    return 1;
  }
  
  int argi = 0;
  int npoints = stoi(string(argv[++argi]), nullptr, 10);
  int lx = stoi(string(argv[++argi]), nullptr, 10);
  int ly = stoi(string(argv[++argi]), nullptr, 10);
  int nqvecs = stoi(string(argv[++argi]), nullptr, 10);
  double r0 = stod(string(argv[++argi]), nullptr);
  long startTime = stoi(string(argv[++argi]), nullptr, 10);
  long endTime = stoi(string(argv[++argi]), nullptr, 10);
  long timeInc = stoi(string(argv[++argi]), nullptr, 10);
  long endShiftTime = stoi(string(argv[++argi]), nullptr, 10);
  string posFile (argv[++argi]);
  string posBulkFile (argv[++argi]);
  string outFile (argv[++argi]);

  double qmag = M_PI/r0;
  
  PositionReader reader;
  if (!reader.open(posFile, npoints, lx, ly)) {
    cout << "Error: cannot open the file " << posFile << endl;
    return 1;
  }

  // Read the position data
  int nbins = static_cast<int>((endTime-startTime)/timeInc)+1;
  double*** pos = create3DArray<double>(nbins, npoints, 2);
  long time;
  int ibin;
  cout << "Reading data ..." << endl;
  while (reader.nextFrame()) {
    time = reader.getTime();
    if (time < startTime) {
      continue;
    } else if (time > endTime) {
      break;
    } else if ((time-startTime) % timeInc == 0) {
      ibin = static_cast<int>((time-startTime)/timeInc);
      for (int i = 0; i < npoints; i++) {
	pos[ibin][i][0] = reader.getUnwrappedPosition(i, 0);
	pos[ibin][i][1] = reader.getUnwrappedPosition(i, 1);
      }
    }
  }
  cout << "Done reading data" << endl;
  reader.close();

  // Read bulk cm
  double** totCM = create2DArray<double>(nbins, 2);
  ifstream cmReader;
  cmReader.open(posBulkFile);
  if (!cmReader) {
    cout << "Error: cannot open the file " << posBulkFile << endl;
    return 1;
  }
  long t;
  double xcm, ycm;
  stringstream iss;
  string line;
  while (getline(cmReader, line)) {
    iss.clear();
    iss.str(line);
    iss >> t >> xcm >> ycm;
    if (t < startTime) {
      continue;
    } else if (t > endTime) {
      break;
    } else if ((t-startTime) % timeInc == 0) {
      ibin = static_cast<int>((t-startTime)/timeInc);
      totCM[ibin][0] = xcm;
      totCM[ibin][1] = ycm;
    }
  }
  cmReader.close();

  // Store a set of orietations for the q vectors to average over
  double** qvecs = create2DArray<double>(nqvecs, 2);
  double theta;
  for (int i = 0; i < nqvecs; i++) {
    theta = i*(2.0*M_PI/nqvecs);
    qvecs[i][0] = qmag*cos(theta);
    qvecs[i][1] = qmag*sin(theta);
  }

  // Compute the self intermediate scattering function
  int endShiftBin = static_cast<int>((endShiftTime-startTime)/timeInc)+1;
  if (endShiftBin >= nbins) endShiftBin = nbins-1;

#ifdef _OPENMP
  int nthreads = omp_get_max_threads();
#else
  int nthreads = 1;
#endif
  double** cosAvg = create2DArray<double>(nthreads, nbins);
  double** sinAvg = create2DArray<double>(nthreads, nbins);
  double*** dr = create3DArray<double>(nthreads, npoints, 2);
  long** count = create2DArray<long>(nthreads, nbins);

#pragma omp parallel default(none)					\
  shared(nbins, npoints, nqvecs, pos, cosAvg, sinAvg, dr, count, totCM),\
  shared(qvecs, endShiftBin) private(ibin)
  {
#ifdef _OPENMP
    int id = omp_get_thread_num();
#else
    int id = 0;
#endif
    double dxcm, dycm, qdr;
#pragma omp for schedule(dynamic,10)
    for (int i = 0; i < endShiftBin; i++) {
      for (int j = i+1; j < nbins; j++) {
	ibin = j-i;
	dxcm = totCM[j][0]-totCM[i][0];
	dycm = totCM[j][1]-totCM[i][1];
	for (int k = 0; k < npoints; k++) {
	  dr[id][k][0] = (pos[j][k][0]-pos[i][k][0])-dxcm;
	  dr[id][k][1] = (pos[j][k][1]-pos[i][k][1])-dycm;
	}
	for (int k = 0; k < nqvecs; k++) {
	  for (int l = 0; l < npoints; l++) {
	    qdr = qvecs[k][0]*dr[id][l][0]+qvecs[k][1]*dr[id][l][1];
	    cosAvg[id][ibin] += cos(qdr);
	    sinAvg[id][ibin] += sin(qdr);
	  }
	  count[id][ibin] += npoints;
	}
      }     
    }
  }

  // Combine results
  for (int i = 1; i < nthreads; i++) {
    for (int j = 0; j < nbins; j++) {
      cosAvg[0][j] += cosAvg[i][j];
      sinAvg[0][j] += sinAvg[i][j];
      count[0][j] += count[i][j];
    }
  }
  // Normalise results
  cosAvg[0][0] = 1.0;
  sinAvg[0][0] = 0.0;
  for (int i = 1; i < nbins; i++) {
    cosAvg[0][i] /= static_cast<double>(count[0][i]);
    sinAvg[0][i] /= static_cast<double>(count[0][i]);
  }

  // Output results
  ofstream writer;
  writer.open(outFile);
  if (!writer) {
    cout << "Error: cannot open the file " << outFile << endl;
    return 1;
  }
  
  for (int i = 0; i < nbins; i++) {
    writer << (i*timeInc) << " " << cosAvg[0][i] << " " << sinAvg[0][i]
	   << "\n";
  }
  writer.close();
  
  // Clean up
  deleteArray(pos);
  deleteArray(totCM);
  deleteArray(cosAvg);
  deleteArray(sinAvg);
  deleteArray(dr);
  deleteArray(count);
}
