// disc_defect_polar.cpp
// A program to measure the radial and angular separation between structural
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
#include <armadillo>
#include "position.hpp"
#include "array.hpp"

using std::cout;
using std::endl;
using std::string;
using std::ifstream;
using std::ofstream;
using std::stringstream;
using std::vector;
using namespace arma;

struct Eigen {
  mat matrix = mat(2,2);
  vec eigenVal = vec(2);
  mat eigenVec = mat(2,2);
};

void getEigen(double mxx, double myy, double mxy, Eigen& eigen);
double pbcDiff(double len, double x1, double x2);
double pbcDist(int ndims, double* len, double* x1, double* x2);
double dot(int ndims, double* x1, double* x2);
double min(double a, double b);

int main(int argc, char* argv[]) {
  if (argc < 16) {
    cout << "Usage: disc_defect_polar npoints lx ly ntbins rinc coeff "
	 << "seed normHist startTime endTime timeInc outFile "
	 << "[posFile neighnumFile deformFile defectFile ...]" << endl;
    return 1;
  }

  int argi = 0;
  int npoints = stoi(string(argv[++argi]), nullptr, 10);
  int lx = stoi(string(argv[++argi]), nullptr, 10);
  int ly = stoi(string(argv[++argi]), nullptr, 10);
  int ntbins = stoi(string(argv[++argi]), nullptr, 10);
  double rinc = stod(string(argv[++argi]), nullptr);
  double coeff = stod(string(argv[++argi]), nullptr);
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
  double tmax = 2.0*M_PI;
  double tinc = tmax/static_cast<double>(ntbins);
  int nrbins = static_cast<int>(rmax/rinc);

  double** pveDiscPveDftHist = create2DArray<double>(nrbins, ntbins);
  double** nveDiscPveDftHist = create2DArray<double>(nrbins, ntbins);  
  double** discPveDftHist = create2DArray<double>(nrbins, ntbins);
  double** pveDiscNveDftHist = create2DArray<double>(nrbins, ntbins);
  double** nveDiscNveDftHist = create2DArray<double>(nrbins, ntbins);
  double** discNveDftHist = create2DArray<double>(nrbins, ntbins);  
  double** discDftHist = create2DArray<double>(nrbins, ntbins);
  double** rand6PveDftHist = create2DArray<double>(nrbins, ntbins);
  double** rand6NveDftHist = create2DArray<double>(nrbins, ntbins);
  double** rand6DftHist = create2DArray<double>(nrbins, ntbins);  
  double** randPveDftHist = create2DArray<double>(nrbins, ntbins);
  double** randNveDftHist = create2DArray<double>(nrbins, ntbins);
  double** randDftHist = create2DArray<double>(nrbins, ntbins);
  double** rand6Hist = create2DArray<double>(nrbins, ntbins);  
  double** randHist = create2DArray<double>(nrbins, ntbins);

  PositionReader posReader;
  ifstream neiReader;
  ifstream defReader;
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
  int axis;
  double x, y, q, sxx, syy, sxy, lam0, lam1;
  Eigen eigen;

  int* neigh = create1DArray<int>(npoints);
  double** posRef = create2DArray<double>(npoints,ndims);
  double** defAxis = create2DArray<double>(npoints,ndims);
  double** posDft = nullptr;
  double* qdft = nullptr;
  
  vector<int> pveDiscIdx, nveDiscIdx, randIdx;
  vector<int> ptIdx (npoints);
  std::iota(ptIdx.begin(), ptIdx.end(), 0);

  std::mt19937 rand(seed);
  std::uniform_real_distribution<double> randDouble(0.0,1.0);

  // For adding a point to the histogram
  auto addHistData =
    [rmax, rinc, tinc, nrbins]
    (double r, double theta, double** hist) -> bool {
      int ir = static_cast<int>(r/rinc);
      if (ir >= nrbins) {
	cout << "Warning: value " << r << " exceed " << rmax << endl;
	return false;
      }
      int it = static_cast<int>((theta)/tinc);
      hist[ir][it] += 1.0;
      it = static_cast<int>((M_PI-theta)/tinc);
      hist[ir][it] += 1.0;
      it = static_cast<int>((M_PI+theta)/tinc);
      hist[ir][it] += 1.0;
      it = static_cast<int>((2.0*M_PI-theta)/tinc);
      hist[ir][it] += 1.0;      
      return true;
    };
  
  auto findNearestRefPt =
    [ndims, &dims, lx, ly, posRef, defAxis, addHistData]
    (int ndiscs, double* pos, double** hist, vector<int>& refIdx) {
      double dist, theta;
      double dr[2];
      double minDist = -1.0;
      int minIdx = 0;
      for (int j = 0; j < ndiscs; j++) {
	dist = pbcDist(ndims, dims, pos, posRef[refIdx[j]]);
	if (j == 0 || dist < minDist) {
	  minDist = dist;
	  minIdx = refIdx[j];
	}
      }
      dr[0] = pbcDiff(lx, pos[0], posRef[minIdx][0]) / minDist;
      dr[1] = pbcDiff(ly, pos[1], posRef[minIdx][1]) / minDist;
      theta = acos(dot(ndims, defAxis[minIdx], dr));
      theta = min(theta, M_PI-theta);
      addHistData(minDist, theta, hist);
    };
  
  while ((argc-(argi+1)) > 0) {
    string posFile (argv[++argi]);
    string neighnumFile (argv[++argi]);
    string deformFile (argv[++argi]);
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
    defReader.open(deformFile);
    if (!defReader) {
      cout << "Error: cannot open the file " << deformFile << endl;
      return 1;
    }
    dftReader.open(defectFile);
    if (!dftReader) {
      cout << "Error: cannot open the file " << defectFile << endl;
      return 1;
    }

    bool readPos = false;
    bool readNei = false;
    bool readDef = false;
    bool readDft = false;
    
    for (long time = startTime; time <= endTime; time += timeInc) {
      readPos = false;
      readNei = false;
      readDef = false;
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

      // Read deform data
      while (getline(defReader, line)) {
	getline(defReader, line);
	ss.clear();
	ss.str(line);
	ss >> str >> t;
	if (t != time) {
	  for (int n = 0; n < npoints; n++) {
	    getline(defReader, line);
	  }
	} else {
	  for (int n = 0; n < npoints; n++) {
	    getline(defReader, line);
	    ss.clear();
	    ss.str(line);
	    ss >> sxx >> syy >> sxy;
	    sxx *= coeff;
	    syy *= coeff;
	    sxy *= coeff;
	    // Use armadillo to compute eigenvalues/eigenvectors to minimise
	    // numerical errors when cells are close to being a sphere
	    getEigen(sxx, syy, sxy, eigen);
	    lam0 = eigen.eigenVal(0);
	    lam1 = eigen.eigenVal(1);
	    // Always use the larger eigenvalue - this should correspond to the
	    // elongation axis (coeff must be set to ensure this is the case)
	    axis = lam0 < lam1 ? 1 : 0;
	    defAxis[n][0] = eigen.eigenVec.at(0,axis);
	    defAxis[n][1] = eigen.eigenVec.at(1,axis);	    
	  }
	  readDef = true;
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
      if (!readPos || !readNei || !readDef || !readDft) {
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
    defReader.close();
    dftReader.close();
  } // Close loop over files

  for (int i = 0; i < nrbins; i++) {
    for (int j = 0; j < ntbins; j++) {
      discPveDftHist[i][j] = pveDiscPveDftHist[i][j] + nveDiscPveDftHist[i][j];
      discNveDftHist[i][j] = pveDiscNveDftHist[i][j] + nveDiscNveDftHist[i][j];
      discDftHist[i][j] = discPveDftHist[i][j] + discNveDftHist[i][j];
      rand6DftHist[i][j] = rand6PveDftHist[i][j] + rand6NveDftHist[i][j];      
      randDftHist[i][j] = randPveDftHist[i][j] + randNveDftHist[i][j];
    }
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
      for (int j = 0; j < ntbins; j++) {
	pveDiscPveDftSum += pveDiscPveDftHist[i][j];
	nveDiscPveDftSum += nveDiscPveDftHist[i][j];
	discPveDftSum += discPveDftHist[i][j];
	pveDiscNveDftSum += pveDiscNveDftHist[i][j];
	nveDiscNveDftSum += nveDiscNveDftHist[i][j];	
	discNveDftSum += discNveDftHist[i][j];
	discDftSum += discDftHist[i][j];
	rand6PveDftSum += rand6PveDftHist[i][j];
	rand6NveDftSum += rand6NveDftHist[i][j];
	rand6DftSum += rand6DftHist[i][j];
	randPveDftSum += randPveDftHist[i][j];
	randNveDftSum += randNveDftHist[i][j];
	randDftSum += randDftHist[i][j];
	rand6Sum += rand6Hist[i][j];	
	randSum += randHist[i][j];
      }
    }
    for (int i = 0; i < nrbins; i++) {
      for (int j = 0; j < ntbins; j++) {
	pveDiscPveDftHist[i][j] /= pveDiscPveDftSum;
	nveDiscPveDftHist[i][j] /= nveDiscPveDftSum;	
	discPveDftHist[i][j] /= discPveDftSum;
	pveDiscNveDftHist[i][j] /= pveDiscNveDftSum;
	nveDiscNveDftHist[i][j] /= nveDiscNveDftSum;	
	discNveDftHist[i][j] /= discNveDftSum;
	discDftHist[i][j] /= discDftSum;
	rand6PveDftHist[i][j] /= rand6PveDftSum;
	rand6NveDftHist[i][j] /= rand6NveDftSum;
	rand6DftHist[i][j] /= rand6DftSum;
	randPveDftHist[i][j] /= randPveDftSum;
	randNveDftHist[i][j] /= randNveDftSum;
	randDftHist[i][j] /= randDftSum;
	rand6Hist[i][j] /= rand6Sum;
	randHist[i][j] /= randSum;
      }
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
  writer << "# tbins (tmin,tmax,tinc,ntbins): "
	 << 0.0 << " " << tmax << " " << tinc << " " << ntbins << "\n";
  writer << "# data_columns: r_low r_mid r_high t_low t_mid t_high "
	 << "pve_disc_pve_dft nve_disc_pve_dft disc_pve_dft "
	 << "pve_disc_nve_dft nve_disc_nve_dft disc_nve_dft disc_dft"
	 << "rand6_pve_dft rand6_nve_dft rand6_dft rand_pve_dft rand_nve_dft "
	 << "rand_dft rand6 rand\n";
  double r, theta;
  for (int i = 0; i < nrbins; i++) {
    r = i*rinc;
    for (int j = 0; j < ntbins; j++) {
      theta = j*tinc;
      writer << r << " " << r+0.5*rinc << " " << r+rinc << " "
	     << theta << " " << theta+0.5*tinc << " " << theta+tinc << " "
	     << pveDiscPveDftHist[i][j] << " "
	     << nveDiscPveDftHist[i][j] << " " << discPveDftHist[i][j] << " "
	     << pveDiscNveDftHist[i][j] << " "
	     << nveDiscNveDftHist[i][j] << " " << discNveDftHist[i][j] << " "
	     << discDftHist[i][j] << " " << rand6PveDftHist[i][j] << " "
	     << rand6NveDftHist[i][j] << " " << rand6DftHist[i][j] << " "
	     << randPveDftHist[i][j] << " " << randNveDftHist[i][j] << " "
	     << randDftHist[i][j] << " " << rand6Hist[i][j] << " "
	     << randHist[i][j] << "\n";
    }
    writer << "\n";
  }
  writer.close();
  
  // Clean up
  deleteArray(neigh);
  deleteArray(posRef);
  deleteArray(defAxis);
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

inline double dot(int ndims, double* x1, double* x2) {
  double sum = 0.0;
  for (int i = 0; i < ndims; i++) {
    sum += x1[i]*x2[i];
  }
  return sum;
}

inline double min(double a, double b) {
  return a < b ? a : b; 
}

void getEigen(double mxx, double myy, double mxy, Eigen& eigen) {
  eigen.matrix.at(0,0) = mxx;
  eigen.matrix.at(0,1) = mxy;
  eigen.matrix.at(1,0) = mxy;
  eigen.matrix.at(1,1) = myy;
  // Note that eigenvectors are stored in columns, not rows
  eig_sym(eigen.eigenVal, eigen.eigenVec, eigen.matrix);
}
