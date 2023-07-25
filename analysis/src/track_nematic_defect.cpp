// track_nematic_defect.cpp

#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <string>
#include <vector>
#include <cmath>
#include <map>
#include <set>

using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::stringstream;
using std::string;
using std::vector;
using std::pair;
using std::map;
using std::set;

struct Defect {
  vector<double> pt;
  double q;
};

struct DefectTrajectory {
  vector<vector<double> > pos;
  long startTime;
  double q;
};

double ddiff(double len, double x1, double x2);
double ddistsq(int ndims, const vector<double>& len,
	       const vector<double>& v1, const vector<double>& v2);

int main(int argc, char* argv[]) {
  if (argc != 10) {
    cout << "Usage: track_nematic_defect lx ly thres minFrames startTime "
	 << "endTime timeInc defectFile trajFile" << endl;
    return 1;
  }

  int argi = 0;
  int lx = stoi(string(argv[++argi]), nullptr, 10);
  int ly = stoi(string(argv[++argi]), nullptr, 10);
  double thres = stod(string(argv[++argi]), nullptr);
  int minFrames = stoi(string(argv[++argi]), nullptr, 10);
  long startTime = stol(string(argv[++argi]), nullptr, 10);
  long endTime = stol(string(argv[++argi]), nullptr, 10);
  long timeInc = stol(string(argv[++argi]), nullptr, 10);
  string defectFile (argv[++argi]);
  string trajFile (argv[++argi]);

  const int ndims = 2;
  vector<double> boxsize = {(double) lx, (double) ly};
  double thressq = thres*thres;
  
  ifstream reader;
  reader.open(defectFile);
  if (!reader) {
    cout << "Error: cannot open the file " << defectFile << endl;
    return 1;
  }

  ofstream writer;
  writer.open(trajFile);
  if (!writer) {
    cout << "Error: cannot open the file " << trajFile << endl;
    return 1;
  }

  vector<Defect> defects;
  int trajCount = 0;
  map<int,DefectTrajectory> defectTraj;

  int ndefects = 0;
  long time;
  string line, str;
  stringstream ss;
  double x, y, q;
  while (getline(reader, line)) {
    // Read header lines
    ss.clear();
    ss.str(line);
    ss >> str >> ndefects;
    getline(reader, line);
    ss.clear();
    ss.str(line);
    ss >> str >> time;
    if (time > endTime) {
      break;
    } else if (time < startTime || (time-startTime) % timeInc != 0) {
      // Skip irrelevant frames
      for (int i = 0; i < ndefects; i++) {
	getline(reader, line);
      }
      continue;
    }
    cout << "Working on t = " << time << endl;
    // Read defect positions and charges
    defects.clear();
    defects.reserve(ndefects);
    for (int i = 0; i < ndefects; i++) {
      getline(reader, line);
      ss.clear();
      ss.str(line);
      ss >> x >> y >> q;
      defects.push_back({{x,y},q});
    }
    // Compute closest distances to previous defects
    vector<int> taken (ndefects,-1);
    double distsq;
    set<int> trajToRemove;
    for (auto& it : defectTraj) {
      int index = -1;
      double minDistSq = -1.0;
      auto& id = it.first;
      auto& traj = it.second;
      for (int i = 0; i < ndefects; i++) {
	if (taken[i] == -1) {
	  distsq = ddistsq(ndims, boxsize, traj.pos.back(), defects[i].pt);
	  if (fabs(traj.q-defects[i].q) < 1e-3 && distsq < thressq &&
	      (index == -1 || distsq < minDistSq)) {
	    minDistSq = distsq;
	    index = i;
	  }	  
	}
      }
      if (index != -1) { // Found new connection
	taken[index] = id;
	traj.pos.push_back(defects[index].pt);
      } else { // No new connection
	trajToRemove.insert(id);
      }
    }

    // Remove and output trajectories for those without new connection
    for (auto& id : trajToRemove) {
      // Only dump trajectories with more than minimum number of frames
      auto& traj = defectTraj[id];
      int nframes = traj.pos.size();
      if (nframes >= minFrames) {
	writer << traj.q << " " << traj.startTime << " " << time << " "
	       << timeInc << " ";
	for (int n = 0; n < nframes; n++) {
	  writer << traj.pos[n][0] << " " << traj.pos[n][1] << " ";
	}
	writer << "\n";
      }
      defectTraj.erase(id);
    }

    // Store new trajectories for defects not taken
    for (int i = 0; i < ndefects; i++) {
      if (taken[i] == -1) {
	auto traj = DefectTrajectory();
	traj.startTime = time;
	traj.pos.push_back(defects[i].pt);
	traj.q = defects[i].q;
	defectTraj[trajCount] = traj;
	trajCount++;
      }
    }
  }
  reader.close();
  writer.close();
}

double ddiff(double len, double x1, double x2) {
  double dx = x1-x2;
  return dx-round(dx/len)*len;
}

double ddistsq(int ndims, const vector<double>& len,
	       const vector<double>& v1, const vector<double>& v2) {
  double sum = 0.0;
  double dx;
  for (int i = 0; i < ndims; i++) {
    dx = ddiff(len[i], v1[i], v2[i]);
    sum += dx*dx;
  }
  return sum;
}
