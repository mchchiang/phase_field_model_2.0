// fourpt_pos_corr.cpp
// A code that computes the 4-point positional correlation function, which is
// defined as
// chi_4 = N*[<Q(r,t,r_c)^2>-<Q(r,t,r_c)>^2]  with
// Q(r,t,r_c) = 1/N*sum_i H(r_c-|r_i(t)-r_i(0)|)
// with H(x) being the Heaviside step function

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
  
  if (argc != 12) {
    cout << "Usage: fourpt_pos_corr npoints lx ly thres startTime "
	 << "endTime timeInc endShiftTime posFile posBulkFile outFile" << endl;
    return 1;
  }
  
  int argi = 0;
  int npoints = stoi(string(argv[++argi]), nullptr, 10);
  int lx = stoi(string(argv[++argi]), nullptr, 10);
  int ly = stoi(string(argv[++argi]), nullptr, 10);
  double thres = stod(string(argv[++argi]), nullptr);
  long startTime = stoi(string(argv[++argi]), nullptr, 10);
  long endTime = stoi(string(argv[++argi]), nullptr, 10);
  long timeInc = stoi(string(argv[++argi]), nullptr, 10);
  long endShiftTime = stoi(string(argv[++argi]), nullptr, 10);
  string posFile (argv[++argi]);
  string posBulkFile (argv[++argi]);
  string outFile (argv[++argi]);
  
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
  stringstream ss;
  string line;
  while (getline(cmReader, line)) {
    ss.clear();
    ss.str(line);
    ss >> t >> xcm >> ycm;
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
  
  // Compute the 4-point positional correlation function
  int endShiftBin = static_cast<int>((endShiftTime-startTime)/timeInc)+1;
  if (endShiftBin >= nbins) endShiftBin = nbins-1;

#ifdef _OPENMP
  int nthreads = omp_get_max_threads();
#else
  int nthreads = 1;
#endif
  double** qavg = create2DArray<double>(nthreads, nbins);
  double** qavgsq = create2DArray<double>(nthreads, nbins);
  long** count = create2DArray<long>(nthreads, nbins);

#pragma omp parallel default(none)					\
  shared(nbins, npoints, pos, qavg, qavgsq, count, totCM, endShiftBin)	\
  shared(thres) private(ibin)
  {
#ifdef _OPENMP
    int id = omp_get_thread_num();
#else
    int id = 0;
#endif
    double dx, dy, dxcm, dycm, q;
#pragma omp for schedule(dynamic,10)
    for (int i = 0; i < endShiftBin; i++) {
      for (int j = i+1; j < nbins; j++) {
	ibin = j-i;
	dxcm = totCM[j][0]-totCM[i][0];
	dycm = totCM[j][1]-totCM[i][1];
	q = 0.0;
	for (int n = 0; n < npoints; n++) {
	  dx = pos[j][n][0]-pos[i][n][0]-dxcm;
	  dy = pos[j][n][1]-pos[i][n][1]-dycm;
	  q += exp(-(dx*dx+dy*dy)/(2.0*thres*thres));
	}
	q /= npoints;
	qavg[id][ibin] += q;
	qavgsq[id][ibin] += q*q;
	count[id][ibin]++;
      }
    }
  }

  // Combine results
  for (int i = 1; i < nthreads; i++) {
    for (int j = 0; j < nbins; j++) {
      qavg[0][j] += qavg[i][j];
      qavgsq[0][j] += qavgsq[i][j];
      count[0][j] += count[i][j];
    }
  }
  // Normalise results
  qavg[0][0] = 0.0;
  qavgsq[0][0] = 0.0;
  for (int i = 1; i < nbins; i++) {
    qavg[0][i] /= static_cast<double>(count[0][i]);
    qavgsq[0][i] /= static_cast<double>(count[0][i]);
  }

  // Output results
  ofstream writer;
  writer.open(outFile);
  if (!writer) {
    cout << "Error: cannot open the file " << outFile << endl;
    return 1;
  }
  
  for (int i = 0; i < nbins; i++) {
    writer << (i*timeInc) << " "
	   << npoints * (qavgsq[0][i] - qavg[0][i]*qavg[0][i]) << "\n";
  }
  writer.close();
  
  // Clean up
  deleteArray(pos);
  deleteArray(totCM);
  deleteArray(qavg);
  deleteArray(qavgsq);
  deleteArray(count);
}
