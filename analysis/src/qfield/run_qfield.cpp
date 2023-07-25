// run_qfield.cpp
// A program to analyse the computed deformation tensor (field)

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>
#include <vector>
#include <map>
#include <algorithm>
#include "qfield.hpp"
#include "qfield_writer.hpp"

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::map;
using std::find;

int main(int argc, char* argv[]) {  
  if (argc != 2) {
    cout << "Usage: run_qfield paramsFile" << endl;
    return 1;
  }
  
  int argi = 0;
  string paramsFile (argv[++argi]);

  // Read arguments from params file
  FILE* fparams = fopen(paramsFile.c_str(), "r");
  if (fparams == NULL) {
    cout << "Error: cannot open the file " << paramsFile << endl;
    return 1;
  }
  const int nchr = 1000;
  map<string,int> params;
  char line[nchr], posFile[nchr], matFile[nchr], mode[nchr], shape[nchr];
  int npoints, lx, ly, boxx, boxy, cellLx, cellLy;
  int useTrace = 1;
  int useUniformShape = 0;
  double coeff = 1.0;
  double thickness = 1.0;
  long startTime, endTime, timeInc;
  while (fgets(line, sizeof(line), fparams) != NULL) {
    params["mode"] += sscanf(line, "mode = %s", mode);
    params["npoints"] += sscanf(line, "npoints = %d", &npoints);
    params["lx"] += sscanf(line, "lx = %d", &lx);
    params["ly"] += sscanf(line, "ly = %d", &ly);
    params["boxx"] += sscanf(line, "boxx = %d", &boxx);
    params["boxy"] += sscanf(line, "boxy = %d", &boxy);
    params["cellLx"] += sscanf(line, "cellLx = %d", &cellLx);
    params["cellLy"] += sscanf(line, "cellLy = %d", &cellLy);
    params["thickness"] += sscanf(line, "thickness = %lf", &thickness);
    params["startTime"] += sscanf(line, "startTime = %ld", &startTime);
    params["endTime"] += sscanf(line, "endTime = %ld", &endTime);
    params["timeInc"] += sscanf(line, "timeInc = %ld", &timeInc);
    params["shape"] += sscanf(line, "shape = %s", shape);
    params["coeff"] += sscanf(line, "coeff = %lf", &coeff);
    params["posFile"] += sscanf(line, "posFile = %s", posFile);
    params["matFile"] += sscanf(line, "matFile = %s", matFile);
    params["useTrace"] += sscanf(line, "useTrace = %d", &useTrace);
    params["useUniformShape"] += sscanf(line, "useUniformShape = %d",
					&useUniformShape);
  }
  fclose(fparams);
  
  string smode (mode);
  Qfield qfield;
  map<string,vector<string> > requiredParams;
  vector<string> modes = {"shape", "box", "smoothed"};
  requiredParams["shape"] =
    {"npoints", "lx", "ly", "cellLx", "cellLy", "startTime", "endTime",
     "timeInc", "shape", "posFile", "matFile"};
  requiredParams["box"] =
    {"lx", "ly", "boxx", "boxy", "startTime", "endTime",
     "timeInc", "matFile"};
  requiredParams["smoothed"] =
    {"lx", "ly", "startTime", "endTime", "timeInc", "matFile"};
  if (find(modes.begin(), modes.end(), smode) != modes.end()) {
    for (auto& p : requiredParams[smode]) {
      if (!params[p]) {
	cout << "Missing the parameter " << p << endl;
	return 1;
      }
    }
    if (smode == "shape") {
      qfield = QField::makeShapeSmoothQField
	(npoints, lx, ly, cellLx, cellLy, coeff, thickness, startTime, endTime,
	 timeInc, useUniformShape, useTrace, shape, posFile, matFile);
    } else if (smode == "box") {
      qfield = QField::makeBoxSmoothQField
	(lx, ly, boxx, boxy, coeff, startTime, endTime, timeInc, useTrace,
	 matFile);
    } else if (smode == "smoothed") {
      qfield = QField::makeSmoothQField
	(lx, ly, coeff, startTime, endTime, timeInc, useTrace, matFile);
    }
  } else {
    cout << "Error: invalid read mode" << endl;
    return 1;
  }
  
  if (!qfield->getStatus()) {
    return 1;
  }
  
  // Read dumps
  int ngridx, ngridy, minLen, maxLen, lenInc, overwrite;
  int contourDist, binsize;
  double minDist;
  long sampFreq;
  char outFile[nchr];
  vector<Qwrite> writers;
  fparams = fopen(paramsFile.c_str(), "r");
  while (fgets(line, sizeof(line), fparams) != NULL) {
    if (line[0] == '#') continue; // Skip comments
    if (sscanf(line, "dump_global_obs %ld %s", &sampFreq, outFile) == 2) {
      writers.push_back(QFieldWriter::makeGlobalObsWriter
			(sampFreq, outFile));
    }
    if (sscanf(line, "dump_smooth_qfield %d %ld %s",
	       &overwrite, &sampFreq, outFile) == 3) {
      writers.push_back(QFieldWriter::makeSmoothQFieldWriter
			(overwrite, sampFreq, outFile));
    }
    if (sscanf(line, "dump_director %d %d %ld %s",
	       &binsize, &overwrite, &sampFreq, outFile) == 4) {
      writers.push_back(QFieldWriter::makeDirectorWriter
			(lx, ly, binsize, overwrite, sampFreq, outFile));
    }
    if (sscanf(line, "dump_topo_charge %d %ld %s",
	       &overwrite, &sampFreq, outFile) == 3) {
      writers.push_back(QFieldWriter::makeTopoChargeWriter
			(overwrite, sampFreq, outFile));
    }
    if (sscanf(line, "dump_local_align %d %d %d %d %d %ld %s",
	       &ngridx, &ngridy, &minLen, &maxLen, &lenInc,
	       &sampFreq, outFile) == 7) {
      writers.push_back(QFieldWriter::makeLocalAlignWriter
			(lx, ly, ngridx, ngridy, minLen, maxLen, lenInc,
			 sampFreq, outFile));
    }
    if (sscanf(line, "dump_nematic_defect %lf %d %d %ld %s",
	       &minDist, &contourDist, &binsize, &sampFreq, outFile) == 5) {
      writers.push_back(QFieldWriter::makeNematicDefectWriter
			(lx, ly, minDist, contourDist, binsize, sampFreq,
			 outFile));
    }
  }
  cout << "Read all parameters" << endl;

  // Make measurements
  while (qfield->nextFrame()) {
    for (auto& writer : writers) {
      writer->sample(*qfield);
    }
  }
}
