// qfield.cpp

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <armadillo>
#include "array.hpp"
#include "qfield_util.hpp"
#include "qfield.hpp"

using std::cout;
using std::endl;
using std::ifstream;
using std::stringstream;
using std::string;
using namespace arma;

// Member functions of QField
QField::QField(int _lx, int _ly, long tstart, long tend, long tinc, bool tr) :
  lx(_lx), ly(_ly), startTime(tstart), endTime(tend), timeInc(tinc),
  trace(tr) {
  q = create3DArray<double>(lx, ly, ncomps);
}

QField::~QField() {deleteArray(q);}

int QField::getLx() const {return lx;}
int QField::getLy() const {return ly;}
long QField::getStartTime() const {return startTime;}
long QField::getTime() const {return time;}
bool QField::getStatus() const {return status;}
double*** QField::getField() const {return q;}

double& QField::operator()(int i, int j, int comp){
  return q[i][j][comp];
}

double QField::operator()(int i, int j, int comp) const {
  return q[i][j][comp];
}

void QField::resetField() {
  for (int i = 0; i < lx; i++) {
    for (int j = 0; j < ly; j++) {
      for (int k = 0; k < ncomps; k++) {
	q[i][j][k] = 0.0;
      }
    }
  }
}

Qfield QField::makeShapeSmoothQField
(int npts, int lx, int ly, int clx, int cly, double cf, double xi,
 long tstart, long tend, long tinc, bool uni, bool tr, const string& mode,
 const string& pfile, const string& mfile) {
  return Qfield(new ShapeSmoothQField
		(npts, lx, ly, clx, cly, cf, xi, tstart, tend, tinc, uni, tr,
		 mode, pfile, mfile));
}

Qfield QField::makeBoxSmoothQField
(int lx, int ly, int gridx, int gridy, double cf, long tstart, long tend,
 long tinc, bool tr, const string& mfile) {
  return Qfield(new BoxSmoothQField
		(lx, ly, gridx, gridy, cf, tstart, tend, tinc, tr, mfile));
}

Qfield QField::makeSmoothQField
(int lx, int ly, double cf, long tstart, long tend, long tinc, bool tr,
 const string& mfile) {
  return Qfield(new SmoothQField(lx, ly, cf, tstart, tend, tinc, tr, mfile));
}

// ============================================================================

// Member functions of ShapeSmoothQField
ShapeSmoothQField::ShapeSmoothQField
(int npts, int _lx, int _ly, int clx, int cly, double cf, double xi,
 long tstart, long tend, long tinc, bool uni, bool tr, const string& mode,
 const string& pfile, const string& mfile) :
  QField(_lx, _ly, tstart, tend, tinc, tr), npoints(npts), cellLx(clx),
  cellLy(cly), coeff(cf), thickness(xi), useUniformShape(uni), posFile(pfile),
  matFile(mfile) {
  status = true;
  if (!posReader.open(posFile, npoints, lx, ly)) {
    cout << "Error: cannot open the file " << posFile << endl;
    return;
  }
  matReader.open(matFile);
  if (!matReader) {
    cout << "Error: cannot open the file " << matFile << endl;
    return;
  }
  // Define shape function
  auto frect = [](double a, double b, double c, double dx, double dy)
	       {return 0.25*(tanh((dx+a)/c)-tanh((dx-a)/c))*
		   (tanh((dy+b)/c)-tanh((dy-b)/c));};
  auto fellipse = [](double a, double b, double c, double dx, double dy)
		  {return 0.5*(1.0+tanh((1.0-dx*dx/(a*a)-dy*dy/(b*b))/c));};
  auto fgauss = [](double a, double b, double _, double dx, double dy)
		{return exp(-(dx*dx+dy*dy)/(0.25*(a+b)*(a+b)));};
  if (mode == "rect") {
    shapeFunc = frect;
  } else if (mode == "ellipse") {
    shapeFunc = fellipse;
  } else { // mode == gauss
    shapeFunc = fgauss;
  }
  // Store deformation tensor data for each cell
  s = create2DArray<double>(npoints,5);
  status = true;
}

ShapeSmoothQField::~ShapeSmoothQField() {
  if (posReader.isOpen()) posReader.close();
  if (!matReader) matReader.close();
  deleteArray(s);
}

bool ShapeSmoothQField::nextFrame() {
  status = false;
  int x, y, ix, iy, axis;
  string line, str;
  stringstream ss;
  Eigen eigen;
  double xcm, ycm, sxx, syy, sxy, vx, vy;
  double lam0, lam1, alam0, alam1, r0, r1, r0avg, r1avg;
  double d, f;
  vec v (2);
  vec vr (2);

  // Read the two header lines and get time
  while (getline(matReader, line)) {
    getline(matReader, line);
    ss.clear();
    ss.str(line);
    ss >> str >> time;
    if (!posReader.nextFrame()) break;
    // Only use the data from the specified time period
    if (time < startTime || (time-startTime) % timeInc != 0) {
      // Skip the data in that time frame
      for (int i = 0; i < npoints; i++) {
	getline(matReader, line);
      }
    } else if (time > endTime) {
      break;
    } else {
      cout << "Working on " << time << endl;
      resetField();
      if (useUniformShape) {
	r0avg = 0.0;
	r1avg = 0.0;
      }
      // Read shape tensor data
      for (int n = 0; n < npoints; n++) {
	getline(matReader, line);
	ss.clear();
	ss.str(line);
	// Input tensor contains trace
	ss >> sxx >> syy >> sxy;
	sxx *= coeff;
	syy *= coeff;
	sxy *= coeff;
	getEigen(sxx, syy, sxy, eigen);
	// Compute deformability
	lam0 = eigen.eigenVal.at(0);
	lam1 = eigen.eigenVal.at(1);
	alam0 = fabs(lam0);
	alam1 = fabs(lam1);
	// Always use the larger eigenvalue - this should correspond to the
	// elongation axis (coeff must be set to ensure this is the case)
	axis = lam0 < lam1 ? 1 : 0;
	// r0 = semi-major axis (the elongation axis)
	// r1 = semi-minor axis
	// The rotated frame is set such that the semi-major axis is the x-axis
	if (alam1 > alam0) {
	  r0 = alam1;
	  r1 = alam0;
	} else {
	  r0 = alam0;
	  r1 = alam1;
	}
	vx = eigen.eigenVec.at(0,axis);
	vy = eigen.eigenVec.at(1,axis);
	d = fabs((lam0-lam1)/(lam0+lam1));
	sxx = vx*vx*d; // Keep trace
	sxy = vx*vy*d;
	syy = vy*vy*d;
	if (!trace) { // Make traceless if needed
	  sxx -= syy;
	  syy = -sxx;
	  sxy *= 2.0;
	}
	s[n][0] = sxx;
	s[n][1] = syy;
	s[n][2] = sxy;
	if (useUniformShape) {
	  r0avg += r0;
	  r1avg += r1;
	} else {
	  s[n][3] = r0;
	  s[n][4] = r1;
	}
      }
      if (useUniformShape) {
	r0avg /= npoints;
	r1avg /= npoints;
      }
      // Read position data and do smoothing
      for (int n = 0; n < npoints; n++) {
	xcm = posReader.getPosition(n,0);
	ycm = posReader.getPosition(n,1);
	// Get cm in local domain coords
	ix = static_cast<int>(xcm)-cellLx/2;
	iy = static_cast<int>(ycm)-cellLy/2;
	xcm -= ix;
	ycm -= iy;
	if (useUniformShape) {
	  for (int i = 0; i < cellLx; i++) {
	    v.at(0) = i+0.5-xcm;
	    x = iwrap(lx,ix+i);
	    for (int j = 0; j < cellLy; j++) {
	      v.at(1) = j+0.5-ycm;
	      y = iwrap(ly,iy+j);
	      vr = eigen.eigenVec.t()*v;
	      f = shapeFunc(r0avg,r1avg,thickness,vr.at(0),vr.at(1));
	      q[x][y][0] += f*s[n][0];
	      q[x][y][1] += f*s[n][1];
	      q[x][y][2] += f*s[n][2];
	    }
	  }
	} else {
	  for (int i = 0; i < cellLx; i++) {
	    v.at(0) = i+0.5-xcm;
	    x = iwrap(lx,ix+i);
	    for (int j = 0; j < cellLy; j++) {
	      v.at(1) = j+0.5-ycm;
	      y = iwrap(ly,iy+j);
	      vr = eigen.eigenVec.t()*v;
	      f = shapeFunc(s[n][3],s[n][4],thickness,vr.at(0),vr.at(1));
	      q[x][y][0] += f*s[n][0];
	      q[x][y][1] += f*s[n][1];
	      q[x][y][2] += f*s[n][2];
	    }
	  }
	}
      } // Close loop over cells
      status = true;
      break;
    } // Close if over time 
  } // Close loop over reading frames
  return status;
}

// ============================================================================

// Member functions of BoxSmoothQField
BoxSmoothQField::BoxSmoothQField
(int _lx, int _ly, int gx, int gy, double cf, long tstart, long tend,
 long tinc, bool tr, const string& mfile) :
  QField(_lx, _ly, tstart, tend, tinc, tr), gridx(gx), gridy(gy), coeff(cf),
  matFile(mfile) {
  status = true;
  time = -1;
  rawq = create3DArray<double>(lx, ly, ncomps);
}

BoxSmoothQField::~BoxSmoothQField() {
  deleteArray(rawq);
}

bool BoxSmoothQField::nextFrame() {
  ifstream matReader;
  stringstream ss;
  string line;
  status = false;
  if (time == -1) {
    time = startTime;
  } else if (time >= endTime) {
    return false;
  } else {
    time += timeInc;
  }
  ss.clear();
  ss.str("");
  ss << matFile << "." << time;
  cout << "Working on " << time << endl;
  // Read weighted deformation tensor data
  string file = ss.str();
  matReader.open(file);
  if (!matReader) {
    cout << "Error: cannot open the file " << file << endl;
    status = false;
    return false;
  }
  int x, y;
  double mxx, myy, mxy;
  while (getline(matReader, line)) {
    ss.clear();
    ss.str(line);
    if (line.size() == 0) continue; // Skip empty lines
    ss >> x >> y >> mxx >> myy >> mxy;
    mxx *= coeff;
    myy *= coeff;
    mxy *= coeff;
    if (!trace) { // Make traceless if needed
      mxx -= myy;
      myy = -mxx;
      mxy *= 2.0;
    }
    rawq[x][y][0] = mxx;
    rawq[x][y][1] = myy;
    rawq[x][y][2] = mxy;
  }
  matReader.close();

  // Do grid smoothing
  int xs, ys;
  const double ngridpts = static_cast<double>(gridx*gridy);
  for (int i = 0; i < lx; i++) {
    xs = i-gridx/2;
    for (int j = 0; j < ly; j++) {
      ys = j-gridy/2;
      mxx = 0.0;
      myy = 0.0;
      mxy = 0.0;
      for (int k = 0; k < gridx; k++) {
	x = iwrap(lx, xs+k);
	for (int l = 0; l < gridy; l++) {
	  y = iwrap(ly, ys+l);
	  mxx += rawq[x][y][0];
	  myy += rawq[x][y][1];
	  mxy += rawq[x][y][2];
	}
      }
      q[i][j][0] = mxx/ngridpts;
      q[i][j][1] = myy/ngridpts;
      q[i][j][2] = mxy/ngridpts;
    }
  }
  status = true;
  return status;
}

// ============================================================================

// Member functions of SmoothQField
SmoothQField::SmoothQField
(int _lx, int _ly, double cf, long tstart, long tend, long tinc, bool tr,
 const string& mfile) :
  QField(_lx, _ly, tstart, tend, tinc, tr), coeff(cf), matFile(mfile) {
  status = true;
  time = -1;
}

SmoothQField::~SmoothQField() {}

bool SmoothQField::nextFrame() {
  ifstream matReader;
  stringstream ss;
  string line;
  status = false;
  if (time == -1) {
    time = startTime;
  } else if (time >= endTime) {
    return false;
  } else {
    time += timeInc;
  }
  ss.clear();
  ss.str("");
  ss << matFile << "." << time;
  cout << "Working on " << time << endl;
  // Read weighted deformation tensor data
  string file = ss.str();
  matReader.open(file);
  if (!matReader) {
    cout << "Error: cannot open the file " << file << endl;
    status = false;
    return false;
  }
  int x, y;
  double mxx, myy, mxy;
  while (getline(matReader, line)) {
    ss.clear();
    ss.str(line);
    if (line.size() == 0) continue; // Skip empty lines
    ss >> x >> y >> mxx >> myy >> mxy;
    mxx *= coeff;
    myy *= coeff;
    mxy *= coeff;
    if (trace) {
      q[x][y][0] = mxx;
      q[x][y][1] = myy;
      q[x][y][2] = mxy;
    } else {
      q[x][y][0] = mxx-myy;
      q[x][y][1] = -q[x][y][0];
      q[x][y][2] = 2.0*mxy;
    }
  }
  matReader.close();
  status = true;
  return status;
}
