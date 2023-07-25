// disc_defect_radial.cpp
// A program to measure the radial separation between structural
// disclination and nematic defects

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>
#include "position.hpp"
#include "array.hpp"

using std::cout;
using std::endl;
using std::string;
using std::ifstream;
using std::ofstream;
using std::stringstream;
using std::vector;

double pbcDiff(double len, double x1, double x2);
double pbcDist(int ndims, double* len, double* x1, double* x2);
double min(double a, double b);

int main(int argc, char* argv[]) {
  if (argc < 12) {
    cout << "Usage: disc_defect_polar npoints lx ly rinc "
	 << "seed normHist startTime endTime timeInc outFile "
	 << "[posFile neighnumFile defectFile ...]" << endl;
    return 1;
  }

  int argi = 0;
  int npoints = stoi(string(argv[++argi]), nullptr, 10);
  int lx = stoi(string(argv[++argi]), nullptr, 10);
  int ly = stoi(string(argv[++argi]), nullptr, 10);
  double rinc = stod(string(argv[++argi]), nullptr);
  long seed = stol(string(argv[++argi]), nullptr, 10);
  int normHist = stoi(string(argv[++argi]), nullptr, 10);
  long startTime = stol(string(argv[++argi]), nullptr, 10);
  long endTime = stol(string(argv[++argi]), nullptr, 10);
  long timeInc = stol(string(argv[++argi]), nullptr, 10);
  string outFile (argv[++argi]);

  const int ndims = 2;
  double dims[ndims] = {(double) lx, (double) ly};
  double hlx = lx*0.5;
  double hly = ly*0.5;
  double rmax = ceil(sqrt(hlx*hlx+hly*hly));
  int nrbins = static_cast<int>(rmax/rinc);

  double* pveDiscPveDftHist = create1DArray<double>(nrbins);
  double* nveDiscPveDftHist = create1DArray<double>(nrbins);  
  double* discPveDftHist = create1DArray<double>(nrbins);
  double* pveDiscNveDftHist = create1DArray<double>(nrbins);
  double* nveDiscNveDftHist = create1DArray<double>(nrbins);
  double* discNveDftHist = create1DArray<double>(nrbins);  
  double* discDftHist = create1DArray<double>(nrbins);
  double* rand6PveDftHist = create1DArray<double>(nrbins);
  double* rand6NveDftHist = create1DArray<double>(nrbins);
  double* rand6DftHist = create1DArray<double>(nrbins);  
  double* randPveDftHist = create1DArray<double>(nrbins);
  double* randNveDftHist = create1DArray<double>(nrbins);
  double* randDftHist = create1DArray<double>(nrbins);
  double* rand6Hist = create1DArray<double>(nrbins);  
  double* randHist = create1DArray<double>(nrbins);

  PositionReader posReader;
  ifstream neiReader;
  ifstream dftReader;
  ofstream writer;
  writer.open(outFile);
  if (!writer) {
    cout << "Error: cannot open the file " << outFile << endl;
    return 1;
  }
  
  string str, line;
  stringstream ss;
  int neighnum, ndfts;
  long t;
  double x, y, q;
  
  int* neigh = create1DArray<int>(npoints);
  double** posRef = create2DArray<double>(npoints,ndims);
  double** posDft = nullptr;
  double* qdft = nullptr;
  
  vector<int> pveDiscIdx, nveDiscIdx, randIdx;
  vector<int> ptIdx (npoints);
  std::iota(ptIdx.begin(), ptIdx.end(), 0);

  std::mt19937 rand(seed);
  std::uniform_real_distribution<double> randDouble(0.0,1.0);

  // For adding a point to the histogram
  auto addHistData =
    [rmax, rinc, nrbins] (double r, double* hist) -> bool {
      int ir = static_cast<int>(r/rinc);
      if (ir >= nrbins) {
	cout << "Warning: value " << r << " exceed " << rmax << endl;
	return false;
      }
      hist[ir] += 1.0;
      return true;
    };
  
  auto findNearestRefPt =
    [ndims, &dims, lx, ly, posRef, addHistData]
    (int ndiscs, double* pos, double* hist, vector<int>& refIdx) {
      double dist;
      double minDist = -1.0;
      for (int j = 0; j < ndiscs; j++) {
	dist = pbcDist(ndims, dims, pos, posRef[refIdx[j]]);
	if (j == 0 || dist < minDist) {
	  minDist = dist;
	}
      }
      addHistData(minDist, hist);
    };
  
  while ((argc-(argi+1)) > 0) {
    string posFile (argv[++argi]);
    string neighnumFile (argv[++argi]);
    string defectFile (argv[++argi]);
    cout << "Working on " << posFile << " ..." << endl;
    if (!posReader.open(posFile, npoints, lx, ly)) {
      cout << "Error: cannot open the file " << posFile << endl;
      return 1;
    }
    neiReader.open(neighnumFile);
    if (!neiReader) {
      cout << "Error: cannot open the file " << neighnumFile << endl;
      return 1;
    }
    dftReader.open(defectFile);
    if (!dftReader) {
      cout << "Error: cannot open the file " << defectFile << endl;
      return 1;
    }

    bool readPos = false;
    bool readNei = false;
    bool readDft = false;
    
    for (long time = startTime; time <= endTime; time += timeInc) {
      readPos = false;
      readNei = false;
      readDft = false;

      // Read position data
      while (posReader.nextFrame()) {
	t = posReader.getTime();
	if (t != time) continue;
	for (int n = 0; n < npoints; n++) {
	  posRef[n][0] = posReader.getPosition(n,0);
	  posRef[n][1] = posReader.getPosition(n,1);
	}
	readPos = true;
	break;
      }
      
      // Read neighnum data
      while (getline(neiReader, line)) {
	getline(neiReader, line);
	ss.clear();
	ss.str(line);
	ss >> str >> t;
	if (t != time) {
	  for (int n = 0; n < npoints; n++) {
	    getline(neiReader, line);
	  }
	} else {
	  for (int n = 0; n < npoints; n++) {
	    getline(neiReader, line);
	    ss.clear();
	    ss.str(line);
	    ss >> neighnum;
	    neigh[n] = neighnum;
	  }
	  readNei = true;
	  break;	  
	}
      }
      
      // Read defect data
      while (getline(dftReader, line)) {
	ss.clear();
	ss.str(line);
	ss >> str >> ndfts;
	getline(dftReader, line);
	ss.clear();
	ss.str(line);
	ss >> str >> t;
	if (t != time) {
	  for (int n = 0; n < ndfts; n++) {
	    getline(dftReader, line);
	  }
	} else {
	  posDft = create2DArray<double>(ndfts,ndims);
	  qdft = create1DArray<double>(ndfts);
	  for (int n = 0; n < ndfts; n++) {
	    getline(dftReader, line);
	    ss.clear();
	    ss.str(line);
	    ss >> x >> y >> q;
	    posDft[n][0] = x;
	    posDft[n][1] = y;
	    qdft[n] = q;
	  }
	  readDft = true;
	  break;
	}
      }      
      if (!readPos || !readNei || !readDft) {
	cout << "Error: not all required data read" << endl;
	return 1;
      }

      if (ndfts == 0) continue;

      // Determine disclinations
      pveDiscIdx.clear();
      nveDiscIdx.clear();
      for (int i = 0; i < npoints; i++) {
	if (neigh[i] > 6) pveDiscIdx.push_back(i);
	else if (neigh[i] < 6) nveDiscIdx.push_back(i);
      }
      int npdiscs = static_cast<int>(pveDiscIdx.size());
      int nndiscs = static_cast<int>(nveDiscIdx.size());
      int ndiscs = npdiscs + nndiscs;
      if (ndiscs == 0) continue;
      
      // Draw a random list of cells for disc
      std::shuffle(ptIdx.begin(), ptIdx.end(), rand);
      randIdx.clear();
      randIdx.reserve(ndiscs);
      int count = 0;
      for (int i = 0; i < npoints; i++) {
	if (neigh[ptIdx[i]] == 6) {
	  randIdx.push_back(ptIdx[i]);
	  count++;
	  if (count >= ndiscs) break;
	}
      }

      double randPos[2];
      for (int i = 0; i < ndfts; i++) {
	if (qdft[i] > 0.0) {
	  findNearestRefPt(npdiscs, posDft[i], pveDiscPveDftHist, pveDiscIdx);
	  findNearestRefPt(nndiscs, posDft[i], nveDiscPveDftHist, nveDiscIdx);
	  findNearestRefPt(ndiscs, posDft[i], rand6PveDftHist, randIdx);
	  findNearestRefPt(ndiscs, posDft[i], randPveDftHist, ptIdx);
	} else {
	  findNearestRefPt(npdiscs, posDft[i], pveDiscNveDftHist, pveDiscIdx);
	  findNearestRefPt(nndiscs, posDft[i], nveDiscNveDftHist, nveDiscIdx);
	  findNearestRefPt(ndiscs, posDft[i], rand6NveDftHist, randIdx);
	  findNearestRefPt(ndiscs, posDft[i], randNveDftHist, ptIdx);
	}
	randPos[0] = randDouble(rand)*lx;
	randPos[1] = randDouble(rand)*ly;
	findNearestRefPt(ndiscs, randPos, rand6Hist, randIdx);	
	findNearestRefPt(ndiscs, randPos, randHist, ptIdx);
      }
      
      // Clean up
      deleteArray(posDft);
      deleteArray(qdft);
      posDft = nullptr;
      qdft = nullptr;
    } // Close loop over time
    posReader.close();
    neiReader.close();
    dftReader.close();
  } // Close loop over files

  for (int i = 0; i < nrbins; i++) {
    discPveDftHist[i] = pveDiscPveDftHist[i] + nveDiscPveDftHist[i];
    discNveDftHist[i] = pveDiscNveDftHist[i] + nveDiscNveDftHist[i];
    discDftHist[i] = discPveDftHist[i] + discNveDftHist[i];
    rand6DftHist[i] = rand6PveDftHist[i] + rand6NveDftHist[i];      
    randDftHist[i] = randPveDftHist[i] + randNveDftHist[i];
  }
  
  // Normalise histograms
  if (normHist) {
    double discPveDftSum = 0.0;
    double discNveDftSum = 0.0;
    double pveDiscPveDftSum = 0.0;
    double nveDiscPveDftSum = 0.0;
    double pveDiscNveDftSum = 0.0;
    double nveDiscNveDftSum = 0.0;    
    double discDftSum = 0.0;
    double randPveDftSum = 0.0;
    double randNveDftSum = 0.0;
    double randDftSum = 0.0;
    double rand6PveDftSum = 0.0;
    double rand6NveDftSum = 0.0;
    double rand6DftSum = 0.0;
    double randSum = 0.0;
    double rand6Sum = 0.0;
    for (int i = 0; i < nrbins; i++) {
      pveDiscPveDftSum += pveDiscPveDftHist[i];
      nveDiscPveDftSum += nveDiscPveDftHist[i];
      discPveDftSum += discPveDftHist[i];
      pveDiscNveDftSum += pveDiscNveDftHist[i];
      nveDiscNveDftSum += nveDiscNveDftHist[i];	
      discNveDftSum += discNveDftHist[i];
      discDftSum += discDftHist[i];
      rand6PveDftSum += rand6PveDftHist[i];
      rand6NveDftSum += rand6NveDftHist[i];
      rand6DftSum += rand6DftHist[i];
      randPveDftSum += randPveDftHist[i];
      randNveDftSum += randNveDftHist[i];
      randDftSum += randDftHist[i];
      rand6Sum += rand6Hist[i];	
      randSum += randHist[i];
    }
    for (int i = 0; i < nrbins; i++) {
      pveDiscPveDftHist[i] /= pveDiscPveDftSum;
      nveDiscPveDftHist[i] /= nveDiscPveDftSum;	
      discPveDftHist[i] /= discPveDftSum;
      pveDiscNveDftHist[i] /= pveDiscNveDftSum;
      nveDiscNveDftHist[i] /= nveDiscNveDftSum;	
      discNveDftHist[i] /= discNveDftSum;
      discDftHist[i] /= discDftSum;
      rand6PveDftHist[i] /= rand6PveDftSum;
      rand6NveDftHist[i] /= rand6NveDftSum;
      rand6DftHist[i] /= rand6DftSum;
      randPveDftHist[i] /= randPveDftSum;
      randNveDftHist[i] /= randNveDftSum;
      randDftHist[i] /= randDftSum;
      rand6Hist[i] /= rand6Sum;
      randHist[i] /= randSum;
    }
  }
  
  // Output results to file
  writer << "# args: ";
  for (int i = 0; i < argc; i++) {
    writer << argv[i] << " ";
  }
  writer << "\n";
  writer << "# rbins (rmin,rmax,rinc,nrbins): "
	 << 0.0 << " " << rmax << " " << rinc << " " << nrbins << "\n";
  writer << "# data_columns: r_low r_mid r_high "
	 << "pve_disc_pve_dft nve_disc_pve_dft disc_pve_dft "
	 << "pve_disc_nve_dft nve_disc_nve_dft disc_nve_dft disc_dft"
	 << "rand6_pve_dft rand6_nve_dft rand6_dft rand_pve_dft rand_nve_dft "
	 << "rand_dft rand6 rand\n";  
  double r;
  for (int i = 0; i < nrbins; i++) {
    r = i*rinc;  
    writer << r << " " << r+0.5*rinc << " " << r+rinc << " "
	   << pveDiscPveDftHist[i] << " "
	   << nveDiscPveDftHist[i] << " " << discPveDftHist[i] << " "
	   << pveDiscNveDftHist[i] << " "
	   << nveDiscNveDftHist[i] << " " << discNveDftHist[i] << " "
	   << discDftHist[i] << " " << rand6PveDftHist[i] << " "
	   << rand6NveDftHist[i] << " " << rand6DftHist[i] << " "
	   << randPveDftHist[i] << " " << randNveDftHist[i] << " "
	   << randDftHist[i] << " " << rand6Hist[i] << " "
	   << randHist[i] << "\n";
  }
  writer.close();
  
  // Clean up
  deleteArray(neigh);
  deleteArray(posRef);
  deleteArray(pveDiscPveDftHist);
  deleteArray(nveDiscPveDftHist);
  deleteArray(discPveDftHist);
  deleteArray(pveDiscNveDftHist);
  deleteArray(nveDiscNveDftHist);  
  deleteArray(discNveDftHist);
  deleteArray(discDftHist);
  deleteArray(rand6PveDftHist);
  deleteArray(rand6NveDftHist);
  deleteArray(rand6DftHist);
  deleteArray(randPveDftHist);
  deleteArray(randNveDftHist);
  deleteArray(randDftHist);
  deleteArray(rand6Hist);  
  deleteArray(randHist);
}

inline double pbcDiff(double len, double x1, double x2) {
  double dx = x1-x2;
  return dx-round(dx/len)*len;
}

inline double pbcDist(int ndims, double* len, double* x1, double* x2) {
  double dx;
  double dr = 0.0;
  for (int i = 0; i < ndims; i++) {
    dx = pbcDiff(len[i],x1[i],x2[i]);
    dr += dx*dx;
  }
  return sqrt(dr);
}

inline double min(double a, double b) {
  return a < b ? a : b; 
}
