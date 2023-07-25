// vel_tavg.cpp
// A program to compute the time average of the advected and centre of mass
// velocities of the cells

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::stringstream;
using std::string;

double normsq(double x, double y);

int main(int argc, char* argv[]) {
  if (argc != 7) {
    cout << "Usage: vel_tavg npoints startTime endTime timeInc velFile outFile"
	 << endl;
    return 1;
  }

  int argi = 0;
  int npoints = stoi(string(argv[++argi]), nullptr, 10);
  long startTime = stol(string(argv[++argi]), nullptr, 10);
  long endTime = stol(string(argv[++argi]), nullptr, 10);
  long timeInc = stol(string(argv[++argi]), nullptr, 10);
  string velFile (argv[++argi]);
  string outFile (argv[++argi]);

  ifstream reader;
  reader.open(velFile);
  if (!reader) {
    cout << "Error: cannot open the file " << velFile << endl;
    return 1;
  }

  ofstream writer;
  writer.open(outFile);
  if (!writer) {
    cout << "Error: cannot open the file " << outFile << endl;
    return 1;
  }

  long t;
  stringstream ss;
  string str, line;
  double vaAllAvg[3] = {0.0, 0.0, 0.0};
  double vcmAllAvg[3] = {0.0, 0.0, 0.0};
  double vaTimeAvg[3] = {0.0, 0.0, 0.0};
  double vcmTimeAvg[3] = {0.0, 0.0, 0.0};
  double vaSpaceVar = 0.0;
  double vcmSpaceVar = 0.0;  
  double vax, vay, vasq, vcmx, vcmy, vcmsq;  
  double vaxAvg, vayAvg, vasqAvg, vcmxAvg, vcmyAvg, vcmsqAvg;
  int nbins = static_cast<int>((endTime-startTime)/timeInc)+1;
  for (long time = startTime; time <= endTime; time += timeInc) {
    while (getline(reader,line)) {
      getline(reader,line);
      ss.clear();
      ss.str(line);
      ss >> str >> t;
      if (t > endTime) {
	break;
      } else if (t != time) {
	// Skip data for irrelevant frames
	for (int n = 0; n < npoints; n++) {
	  getline(reader,line);
	}
      } else {
	vaxAvg = 0.0;
	vayAvg = 0.0;
	vasqAvg = 0.0;
	vcmxAvg = 0.0;
	vcmyAvg = 0.0;
	vcmsqAvg = 0.0;
	for (int n = 0; n < npoints; n++) {
	  getline(reader,line);
	  ss.clear();
	  ss.str(line);
	  ss >> vax >> vay >> vcmx >> vcmy;
	  vasq = normsq(vax,vay);
	  vcmsq = normsq(vcmx,vcmy);
	  vaxAvg += vax;
	  vayAvg += vay;
	  vasqAvg += vasq;
	  vcmxAvg += vcmx;
	  vcmyAvg += vcmy;
	  vcmsqAvg += vcmsq;	 
	}

	vaAllAvg[0] += vaxAvg;
	vaAllAvg[1] += vayAvg;
	vaAllAvg[2] += vasqAvg;
	vcmAllAvg[0] += vcmxAvg;
	vcmAllAvg[1] += vcmyAvg;
	vcmAllAvg[2] += vcmsqAvg;

	vaxAvg /= npoints;
	vayAvg /= npoints;
	vasqAvg /= npoints;
	vcmxAvg /= npoints;
	vcmyAvg /= npoints;
	vcmsqAvg /= npoints;
	vasq = normsq(vaxAvg,vayAvg);
	vcmsq = normsq(vcmxAvg,vcmyAvg);	

	vaSpaceVar += vasqAvg - vasq;
	vcmSpaceVar += vcmsqAvg - vcmsq;
	
	vaTimeAvg[0] += vaxAvg;
	vaTimeAvg[1] += vayAvg;
	vaTimeAvg[2] += vasq;
	vcmTimeAvg[0] += vcmxAvg;
	vcmTimeAvg[1] += vcmyAvg;
	vcmTimeAvg[2] += vcmsq;
      }
    }
  }
  reader.close();

  // Normalise
  // Spatial fluctuations
  vaSpaceVar /= nbins;
  vcmSpaceVar /= nbins;
  
  // Temporal fluctuations
  for (int i = 0; i < 2; i++) {
    vaTimeAvg[i] /= nbins;
    vcmTimeAvg[i] /= nbins;    
  }
  double vaTimeVar = vaTimeAvg[2]-normsq(vaTimeAvg[0],vaTimeAvg[1]);
  double vcmTimeVar = vcmTimeAvg[2]-normsq(vcmTimeAvg[0],vcmTimeAvg[1]);

  // Ensemble fluctuations
  double n = npoints*nbins;
  for (int i = 0; i < 2; i++) {
    vaAllAvg[i] /= n;
    vcmAllAvg[i] /= n;
  }
  double vaAllVar = vaAllAvg[2]-normsq(vaAllAvg[0],vaAllAvg[1]);
  double vcmAllVar = vcmAllAvg[2]-normsq(vcmAllAvg[0],vcmAllAvg[1]);
  
  // Output results
  writer << vaAllAvg[0] << " " << vaAllAvg[1] << " "
	 << sqrt(normsq(vaAllAvg[0],vaAllAvg[1])) << " "
	 << vaSpaceVar << " " << vaTimeVar << " " << vaAllVar << " "
	 << vcmAllAvg[0] << " " << vcmAllAvg[1] << " "
	 << sqrt(normsq(vcmAllAvg[0],vcmAllAvg[1])) << " "
	 << vcmSpaceVar << " " << vcmTimeVar << " " << vcmAllVar << "\n";
  writer.close();
}

inline double normsq(double x, double y) {
  return x*x+y*y;
}
