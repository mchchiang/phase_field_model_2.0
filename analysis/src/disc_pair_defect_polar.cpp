// disc_pair_defect_polar.cpp
// A program to measure the radial and angular separation between 5-7 pair
// structural disclination and nematic defects

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <set>
#include <map>
#include <utility>
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
using std::set;
using std::map;
using std::pair;
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
double pbcWrap(double len, double x);
double dot(int ndims, double* x1, double* x2);
double min(double a, double b);
pair<int,int> makePair(int a, int b);

int main(int argc, char* argv[]) {
  if (argc < 17) {
    cout << "Usage: disc_pair_defect_polar npoints lx ly ntbins rinc coeff "
	 << "seed normHist startTime endTime timeInc outFile "
	 << "[posFile neighFile neighnumFile deformFile defectFile ...]"
	 << endl;
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

  double** discDftHist = create2DArray<double>(nrbins, ntbins);
  double** rand6DftHist = create2DArray<double>(nrbins, ntbins);
  double** randDftHist = create2DArray<double>(nrbins, ntbins);
  double** rand6Hist = create2DArray<double>(nrbins, ntbins);
  double** randHist = create2DArray<double>(nrbins, ntbins);  

  PositionReader posReader;
  ifstream neiReader;
  ifstream nnmReader;
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
  int ineigh, nnum, ndfts;
  long t;
  int axis;
  double x, y, q, sxx, syy, sxy, lam0, lam1;
  Eigen eigen;

  vector<vector<int> > neigh (npoints, vector<int>());
  int* disc = create1DArray<int>(npoints);
  double** posRef = create2DArray<double>(npoints,ndims);
  double** defAxis = create2DArray<double>(npoints,ndims);
  double** posDft = nullptr;
  double* qdft = nullptr;

  set<pair<int,int> > discPairIdx, rand6PairIdx, randPairIdx;
  
  std::mt19937 rand(seed);
  std::uniform_real_distribution<double> randDoubleDist(0.0,1.0);
  auto randDouble = [&rand, &randDoubleDist] () -> double {
		      return randDoubleDist(rand);};
  auto randInt = [&rand, &randDoubleDist] (int n) -> int {
		   return static_cast<int>(randDoubleDist(rand)*n);};

  // For computing CM between pairs of points
  auto computeCMAngle =
    [ndims, lx, ly, posRef, defAxis] (set<pair<int,int> >& pairIdx) {
      int idx1, idx2;
      double xcm, ycm, theta;
      map<pair<int,int>,vector<double> > pairDist;
      for (auto& p : pairIdx) {
	// Compute CM
	idx1 = p.first;
	idx2 = p.second;
	xcm = posRef[idx1][0]*2.0;
	ycm = posRef[idx1][1]*2.0;
	xcm += pbcDiff(lx, posRef[idx2][0], posRef[idx1][0]);
	ycm += pbcDiff(ly, posRef[idx2][1], posRef[idx1][1]);
	xcm = pbcWrap(lx, xcm*0.5);
	ycm = pbcWrap(ly, ycm*0.5);
	// Compute angle
	theta = acos(dot(ndims, defAxis[idx1], defAxis[idx2]));
	theta = min(theta, M_PI-theta);
	pairDist[p] = {xcm,ycm,theta};
      }
      return pairDist;
    };
  
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
    [ndims, &dims, defAxis, addHistData]
    (double* pos, double** hist, set<pair<int,int> >& refIdx,
     map<pair<int,int>,vector<double> >& refCMAngle) {
      double dist;
      double rpos[2];
      double minDist = -1.0;
      pair<int,int> minIdx;
      for (auto& p : refIdx) {
	rpos[0] = refCMAngle[p][0];
	rpos[1] = refCMAngle[p][1];
	dist = pbcDist(ndims, dims, pos, rpos);
	if (minDist < 0.0 || dist < minDist) {
	  minDist = dist;
	  minIdx = p;
	}
      }
      addHistData(minDist, refCMAngle[minIdx][2], hist);
    };
  
  while ((argc-(argi+1)) > 0) {
    string posFile (argv[++argi]);
    string neighFile (argv[++argi]);
    string neighnumFile (argv[++argi]);
    string deformFile (argv[++argi]);
    string defectFile (argv[++argi]);
    cout << "Working on " << posFile << " ..." << endl;
    if (!posReader.open(posFile, npoints, lx, ly)) {
      cout << "Error: cannot open the file " << posFile << endl;
      return 1;
    }
    neiReader.open(neighFile);
    if (!neiReader) {
      cout << "Error: cannot open the file " << neighFile << endl;
      return 1;
    }    
    nnmReader.open(neighnumFile);
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
    bool readNnm = false;
    bool readDef = false;
    bool readDft = false;
    
    for (long time = startTime; time <= endTime; time += timeInc) {
      readPos = false;
      readNei = false;
      readNnm = false;
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

      // Read neighbours data
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
	    neigh[n].clear();
	    getline(neiReader, line);
	    ss.clear();
	    ss.str(line);
	    while (ss) {
	      ss >> ineigh;
	      neigh[n].push_back(ineigh);
	    }
	  }
	  readNei = true;
	  break;
	}
      }
      
      // Read neighnum data
      while (getline(nnmReader, line)) {
	getline(nnmReader, line);
	ss.clear();
	ss.str(line);
	ss >> str >> t;
	if (t != time) {
	  for (int n = 0; n < npoints; n++) {
	    getline(nnmReader, line);
	  }
	} else {
	  for (int n = 0; n < npoints; n++) {
	    getline(nnmReader, line);
	    ss.clear();
	    ss.str(line);
	    ss >> nnum;
	    disc[n] = nnum-6;
	  }
	  readNnm = true;
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
      if (!readPos || !readNei || !readNnm || !readDef || !readDft) {
	cout << "Error: not all required data read" << endl;
	return 1;
      }

      if (ndfts == 0) continue;

      // Determine 5-7 disclination pairs
      discPairIdx.clear();
      for (int i = 0; i < npoints; i++) {
	if (disc[i] != 0) {
	  for (auto& inei : neigh[i]) {
	    if (disc[inei] * disc[i] < 0) {
	      discPairIdx.insert(makePair(i,inei));
	    }
	  }
	}
      }
      int ndiscPairs = static_cast<int>(discPairIdx.size());
      if (ndiscPairs == 0) continue;
      auto discPairCMAngle = computeCMAngle(discPairIdx);
      
      // Draw a random list of cells for disc
      int idx;
      int count = 0;
      rand6PairIdx.clear();
      do {
	idx = randInt(npoints);
	nnum = disc[idx]+6;
	if (nnum != 6) continue;
	ineigh = neigh[idx][randInt(nnum)];
	auto p = makePair(idx,ineigh);
	if (rand6PairIdx.find(p) == rand6PairIdx.end()) {
	  rand6PairIdx.insert(p);
	  count++;
	}
      } while (count < ndiscPairs);
      auto rand6PairCMAngle = computeCMAngle(rand6PairIdx);
      
      randPairIdx.clear();
      count = 0;
      do {
	idx = randInt(npoints);
	nnum = disc[idx]+6;
	ineigh = neigh[idx][randInt(nnum)];	
	auto p = makePair(idx,ineigh);
	if (randPairIdx.find(p) == randPairIdx.end()) {
	  randPairIdx.insert(p);
	  count++;
	}
      } while (count < ndiscPairs);
      auto randPairCMAngle = computeCMAngle(randPairIdx);

      double randPos[2];
      for (int i = 0; i < ndfts; i++) {
	if (qdft[i] < 0.0) continue; // Only focus on +1/2 defects
	findNearestRefPt(posDft[i], discDftHist, discPairIdx, discPairCMAngle);
	findNearestRefPt(posDft[i], rand6DftHist, rand6PairIdx,
			 rand6PairCMAngle);	
	findNearestRefPt(posDft[i], randDftHist, randPairIdx, randPairCMAngle);
	randPos[0] = randDouble()*lx;
	randPos[1] = randDouble()*ly;
	findNearestRefPt(randPos, rand6Hist, rand6PairIdx, rand6PairCMAngle);
	findNearestRefPt(randPos, randHist, randPairIdx, randPairCMAngle);
      }
      
      // Clean up
      deleteArray(posDft);
      deleteArray(qdft);
      posDft = nullptr;
      qdft = nullptr;
    } // Close loop over time
    posReader.close();
    neiReader.close();
    nnmReader.close();
    defReader.close();
    dftReader.close();
  } // Close loop over files
  
  // Normalise histograms
  double discDftSum = 0.0;
  double rand6DftSum = 0.0;
  double randDftSum = 0.0;
  double rand6Sum = 0.0;
  double randSum = 0.0;
  if (normHist) {
    for (int i = 0; i < nrbins; i++) {
      for (int j = 0; j < ntbins; j++) {
	discDftSum += discDftHist[i][j];
	rand6DftSum += rand6DftHist[i][j];
	randDftSum += randDftHist[i][j];
	rand6Sum += rand6Hist[i][j];
	randSum += randHist[i][j];
      }
    }
    for (int i = 0; i < nrbins; i++) {
      for (int j = 0; j < ntbins; j++) {
	discDftHist[i][j] /= discDftSum;
	rand6DftHist[i][j] /= rand6DftSum;
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
	 << "disc_dft rand6_dft rand_dft rand6 rand\n";
  double r, theta;
  for (int i = 0; i < nrbins; i++) {
    r = i*rinc;
    for (int j = 0; j < ntbins; j++) {
      theta = j*tinc;
      writer << r << " " << r+0.5*rinc << " " << r+rinc << " "
	     << theta << " " << theta+0.5*tinc << " " << theta+tinc << " "
	     << discDftHist[i][j] << " " << rand6DftHist[i][j] << " "
	     << randDftHist[i][j] << " " << rand6Hist[i][j] << " "
	     << randHist[i][j] << "\n";
    }
    writer << "\n";
  }
  writer.close();
  
  // Clean up
  deleteArray(disc);
  deleteArray(posRef);
  deleteArray(defAxis);
  deleteArray(discDftHist);
  deleteArray(rand6DftHist);
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

inline double pbcWrap(double len, double x) {
  double remainder = fmod(x,len);
  return remainder >= 0.0 ? remainder : len + remainder;
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

inline pair<int,int> makePair(int a, int b) {
  return a < b ? std::make_pair(a,b) : std::make_pair(b,a);
}
