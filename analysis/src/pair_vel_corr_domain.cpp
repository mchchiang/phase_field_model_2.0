// pair_vel_corr_domain.cpp
// A program to compute pair velocity correlation

#include <iostream>
#include <fstream>
#include <sstream>
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

double ddiff(double len, double x1, double x2);
double ddist(int ndims, double* len, double* v1, double* v2);

int main(int argc, char* argv[]) {
  if (argc != 16) {
    cout << "Usage: pair_vel_corr_domain npoints lx ly ngridx ngridy rmin "
	 << "rmax rinc thres startTime endTime timeInc posFile velFile outFile"
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
  double thres = stod(string(argv[++argi]), nullptr);
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
  double*** sep = create3DArray<double>(ngridx, ngridy, npoints);
  double* gridx = create1DArray<double>(ngridx);
  double* gridy = create1DArray<double>(ngridy);

  // Compute grid positions
  double gridxWidth = lx/static_cast<double>(ngridx);
  double gridyWidth = ly/static_cast<double>(ngridy);
  for (int i = 0; i < ngridx; i++) {
    gridx[i] = (i+0.5)*gridxWidth;
  }
  for (int i = 0; i < ngridy; i++) {
    gridy[i] = (i+0.5)*gridyWidth;
  }
  
  int nbins = static_cast<int>((rmax-rmin)/rinc);
  int ntbins = static_cast<int>((endTime-startTime)/timeInc)+1;  
  double* velAdvDist = create1DArray<double>(nbins);
  double* velCMDist = create1DArray<double>(nbins);
  double* velAdvWeightedDist = create1DArray<double>(nbins);
  double* velCMWeightedDist = create1DArray<double>(nbins);  
  long t;
  double dims[2] = {(double) lx, (double) ly};
  double vax, vay, vcmx, vcmy;
  bool foundVelData = false;
  bool foundPosData = false;
  stringstream ss;
  string str, line;
  for (long time = startTime; time <= endTime; time += timeInc) {
    cout << "Working on t = " << time << endl;
    foundVelData = false;
    foundPosData = false;
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
	foundVelData = true;
	break;
      }
    }

    if (!foundVelData) {
      cout << "Error: cannot find the velocity data for time = " 
	   << time << endl;
      return 1;
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
	foundPosData = true;
	break;
      }
    }
    
    // Compute distance matrix
    double rg[2];
#pragma omp parallel for schedule(static),\
  shared(ngridx, ngridy, npoints, dims, pos, sep) private(rg)
    for (int i = 0; i < ngridx; i++) {
      rg[0] = gridx[i];
      for (int j = 0; j < ngridy; j++) {
	rg[1] = gridy[j];
	for (int k = 0; k < npoints; k++) {
	  sep[i][j][k] = ddist(2, dims, rg, pos[k]);
	}
      }
    }

    if (!foundPosData) {
      cout << "Error: cannot find the position data for time = " 
	   << time << endl;
      return 1;
    }    

    // Compute local alignment order for each box length
#pragma omp parallel default(none),\
  shared(rmin, rinc, nbins, npoints, ngridx, ngridy, sep, thres, velAdv), \
  shared(velCM, velAdvDist, velAdvWeightedDist, velCMDist, velCMWeightedDist)
    {
      int count, ngridptsWithData;
      double vadot, vcmdot, vaCorr, wvaCorr, vcmCorr, wvcmCorr;
      double vaCorrAll, wvaCorrAll, vcmCorrAll, wvcmCorrAll;
#pragma omp for schedule(dynamic)
      for (int n = 0; n < nbins; n++) {
	double r = n*rinc+rmin;
	vaCorrAll = 0.0;
	wvaCorrAll = 0.0;
	vcmCorrAll = 0.0;
	wvcmCorrAll = 0.0;
	ngridptsWithData = 0;
	for (int i = 0; i < ngridx; i++) {
	  for (int j = 0; j < ngridy; j++) {
	    vaCorr = 0.0;
	    vcmCorr = 0.0;
	    wvaCorr = 0.0;
	    wvcmCorr = 0.0;
	    count = 0;
	    for (int k = 0; k < npoints; k++) {
	      for (int l = k; l < npoints; l++) {
		if (sep[i][j][k] < r && sep[i][j][l] < r) {
		  if (velAdv[k][2] > thres && velAdv[l][2] > thres) {
		    vadot = (velAdv[k][0]*velAdv[l][0] +
			     velAdv[k][1]*velAdv[l][1]);
		  } else {
		    vadot = 0.0;
		  }
		  if (velCM[k][2] > thres && velCM[l][2] > thres) {
		    vcmdot = (velCM[k][0]*velCM[l][0] +
			      velCM[k][1]*velCM[l][1]);
		  } else {
		    vcmdot = 0.0;
		  }		  
		  vaCorr += vadot/(velAdv[k][2]*velAdv[l][2]);
		  wvaCorr += vadot;
		  vcmCorr += vcmdot/(velCM[k][2]*velCM[l][2]);
		  wvcmCorr += vcmdot;
		  count++;
		}
	      }
	    }
	    if (count > 0) {
	      vaCorr /= count;
	      wvaCorr /= count;
	      vcmCorr /= count;
	      wvcmCorr /= count;
	      vaCorrAll += vaCorr;
	      wvaCorrAll += wvaCorr;
	      vcmCorrAll += vcmCorr;
	      wvcmCorrAll += wvcmCorr;
	      ngridptsWithData++;
	    }
	  }
	} // Close loop over all grid points
	if (ngridptsWithData > 0) {
	  vaCorrAll /= ngridptsWithData;
	  wvaCorrAll /= ngridptsWithData;	  
	  vcmCorrAll /= ngridptsWithData;
	  wvcmCorrAll /= ngridptsWithData;	  
	  velAdvDist[n] += vaCorrAll;
	  velAdvWeightedDist[n] += wvaCorrAll;	  
	  velCMDist[n] += vcmCorrAll;
	  velCMWeightedDist[n] += wvcmCorrAll;
	}
      } // Close loop over different radial distances
    } // Close parallel region
  } // Close loop over time
  posReader.close();
  velReader.close();

  // Averge over time
  for (int n = 0; n < nbins; n++) {
    velAdvDist[n] /= ntbins;
    velAdvWeightedDist[n] /= ntbins;    
    velCMDist[n] /= ntbins;
    velCMWeightedDist[n] /= ntbins;    
  }
  
  // Normalise and output
  for (int n = 0; n < nbins; n++) {
    double r = rmin+rinc*n;
    writer << r+rinc*0.5 << " " << velAdvDist[n] << " "
	   << velAdvWeightedDist[n] << " " << velCMDist[n] << " "
	   << velCMWeightedDist[n] << "\n";
  }
  writer.close();
  
  // Clean up
  deleteArray(pos);
  deleteArray(velAdv);
  deleteArray(velCM);
  deleteArray(velAdvDist);
  deleteArray(velAdvWeightedDist);
  deleteArray(velCMDist);
  deleteArray(velCMWeightedDist);  
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

