// qfield_writer.cpp

#include <iostream>
#include <sstream>
#include <string>
#include <cmath>
#include <vector>
#include <omp.h>
#include "array.hpp"
#include "qfield.hpp"
#include "qfield_util.hpp"
#include "qfield_writer.hpp"

using std::cout;
using std::endl;
using std::stringstream;
using std::string;
using std::vector;

// Helper functions
void smooth(int lx, int ly, int binsize, double*** field, double*** smoothed);
void error(int n, double avg, double avgSq, double& dev, double& err);

// Member functions of QFieldWriter
QFieldWriter::QFieldWriter(long freq, const string& file) :
  sampFreq(freq), outFile(file) {}
QFieldWriter::~QFieldWriter() {}
bool QFieldWriter::getStatus() {return status;}
void QFieldWriter::sample(const QField& qfield) {
  long time = qfield.getTime();
  if ((time-qfield.getStartTime()) % sampFreq == 0 && status) {
    _sample(time, qfield);
  }
}

Qwrite QFieldWriter::makeGlobalObsWriter(long freq, const string& file) {
  return Qwrite(new GlobalObsWriter(freq, file));
}

Qwrite QFieldWriter::makeSmoothQFieldWriter
(bool owrite, long freq, const string& file) {
  return Qwrite(new SmoothQFieldWriter(owrite, freq, file));
}

Qwrite QFieldWriter::makeDirectorWriter
(int lx, int ly, int bsize, bool owrite, long freq, const string& file) {
  return Qwrite(new DirectorWriter(lx, ly, bsize, owrite, freq, file));
}

Qwrite QFieldWriter::makeTopoChargeWriter
(bool owrite, long freq, const string& file) {
  return Qwrite(new TopoChargeWriter(owrite, freq, file));
}

Qwrite QFieldWriter::makeLocalAlignWriter
(int lx, int ly, int ngx, int ngy, int lmin, int lmax, int linc, long freq,
 const string& file) {
  return Qwrite(new LocalAlignWriter
		(lx, ly, ngx, ngy, lmin, lmax, linc, freq, file));
}

Qwrite QFieldWriter::makeNematicDefectWriter
(int lx, int ly, double mdist, int cdist, int bsize, long freq,
 const string& file) {
  return Qwrite(new NematicDefectWriter
		(lx, ly, mdist, cdist, bsize, freq, file));
}

// ============================================================================

// Member functions of GlobalObsWriter
GlobalObsWriter::GlobalObsWriter(long freq, const string& file) :
  QFieldWriter(freq, file) {
  writer.open(file);
  status = true;
  if (!writer) {
    cout << "Error: cannot open the file " << file << endl;
    status = false;
  }
}

GlobalObsWriter::~GlobalObsWriter() {
  writer.close();
}

void GlobalObsWriter::_sample(long time, const QField& qfield) {
  int axis;
  const int lx = qfield.getLx();
  const int ly = qfield.getLy();
  const double npts = static_cast<double>(lx*ly);
  double lam, lam0, lam1, vx, vy, cos2t, sin2t, nem, wnem;
  double cos2tavg = 0.0;
  double sin2tavg = 0.0;
  double wcos2tavg = 0.0;
  double wsin2tavg = 0.0;
  double lamavg = 0.0;
  Eigen eigen;
  for (int i = 0; i < lx; i++) {
    for (int j = 0; j < ly; j++) {
      getEigen(qfield(i,j,0), qfield(i,j,1), qfield(i,j,2), eigen);
      lam0 = eigen.eigenVal.at(0);
      lam1 = eigen.eigenVal.at(1);
      lam = lam0 < lam1 ? lam1 : lam0;
      axis = lam0 < lam1 ? 1 : 0;
      vx = eigen.eigenVec(0,axis);
      vy = eigen.eigenVec(1,axis);
      cos2t = vx*vx-vy*vy;
      sin2t = 2*vx*vy;
      cos2tavg += cos2t;
      sin2tavg += sin2t;
      wcos2tavg += lam*cos2t;
      wsin2tavg += lam*sin2t;
      lamavg += lam;
    }
  }
  // Normalise
  cos2tavg /= npts;
  sin2tavg /= npts;
  wcos2tavg /= npts;
  wsin2tavg /= npts;
  lamavg /= npts;
  nem = sqrt(cos2tavg*cos2tavg+sin2tavg*sin2tavg);
  wnem = sqrt(wcos2tavg*wcos2tavg+wsin2tavg*wsin2tavg);
  writer << time << " " << lamavg << " " << nem << " " << wnem << "\n";
}

// ============================================================================

// Member functions of SmoothQFieldWriter
SmoothQFieldWriter::SmoothQFieldWriter
(bool owrite, long freq, const string& file) :
  QFieldWriter(freq, file), overwrite(owrite) {
  status = true;
}

SmoothQFieldWriter::~SmoothQFieldWriter() {}

void SmoothQFieldWriter::_sample(long time, const QField& qfield) {
  stringstream ss;
  string file;
  if (!overwrite) {
    ss.clear();
    ss.str("");
    ss << outFile << "." << time;
    file = ss.str();
  }
  writer.open(file);
  if (!writer) {
    cout << "Error: cannot open the file " << file << endl;
    return;
  }
  int lx = qfield.getLx();
  int ly = qfield.getLy();
  double qxx, qyy, qxy;
  for (int i = 0; i < lx; i++) {
    for (int j = 0; j < ly; j++) {
      qxx = qfield(i,j,0);
      qyy = qfield(i,j,1);
      qxy = qfield(i,j,2);
      writer << i << " " << j << " " << qxx << " " << qyy << " " << qxy
	     << "\n";
    }
    writer << "\n";
  }
  writer.close();
}

// ============================================================================

// Member functions of DirectorFieldWriter
DirectorWriter::DirectorWriter
(int _lx, int _ly, int bsize, bool owrite, long freq, const string& file) :
  QFieldWriter(freq, file), lx(_lx), ly(_ly), binsize(bsize),
  overwrite(owrite) {
  status = true;
  if (binsize > 1) {
    smoothedField = create3DArray<double>(lx,ly,3);
  }
}

DirectorWriter::~DirectorWriter() {
  deleteArray(smoothedField);
}

void DirectorWriter::_sample(long time, const QField& qfield) {
  stringstream ss;
  string file;
  if (!overwrite) {
    ss.clear();
    ss.str("");
    ss << outFile << "." << time;
    file = ss.str();
  }
  writer.open(file);
  if (!writer) {
    cout << "Error: cannot open the file " << file << endl;
    return;
  }
  double*** field;
  if (binsize > 1) {
    field = smoothedField;
    smooth(lx, ly, binsize, qfield.getField(), field);
  } else {
    field = qfield.getField();
  }
  int axis;
  double lam0, lam1, lam, vx, vy;
  Eigen eigen;
  for (int i = 0; i < lx; i++) {
    for (int j = 0; j < ly; j++) {
      getEigen(field[i][j][0], field[i][j][1], field[i][j][2], eigen);
      lam0 = eigen.eigenVal.at(0);
      lam1 = eigen.eigenVal.at(1);
      lam = lam0 < lam1 ? lam1 : lam0;
      axis = lam0 < lam1 ? 1 : 0;
      vx = eigen.eigenVec(0,axis);
      vy = eigen.eigenVec(1,axis);
      writer << i << " " << j << " " << vx << " " << vy << " " << lam << "\n";
    }
    writer << "\n";
  }
  writer.close();
}

// ============================================================================

// Member functions of TopoChargeWriter
TopoChargeWriter::TopoChargeWriter
(bool owrite, long freq, const string& file) :
  QFieldWriter(freq, file), overwrite(owrite) {
  status = true;
}

TopoChargeWriter::~TopoChargeWriter() {}

void TopoChargeWriter::_sample(long time, const QField& qfield) {
  stringstream ss;
  string file;
  if (!overwrite) {
    ss.clear();
    ss.str("");
    ss << outFile << "." << time;
    file = ss.str();
  }
  writer.open(file);
  if (!writer) {
    cout << "Error: cannot open the file " << file << endl;
    return;
  }
  int lx = qfield.getLx();
  int ly = qfield.getLy();
  int iu, iuu, id, idd, ju, juu, jd, jdd;
  double*** q = qfield.getField();
  for (int i = 0; i < lx; i++) {
    iu = iup(lx,i);
    iuu = iup(lx,iu);
    id = idown(lx,i);
    idd = idown(lx,id);
    for (int j = 0; j < ly; j++) {
      ju = iup(ly,j);
      juu = iup(ly,ju);
      jd = idown(ly,j);
      jdd = idown(ly,jd);
      writer << i << " " << j << " " <<	1.0/(2.0*M_PI) *
	(cgrad4(i,j,iuu,iu,id,idd,0,0,q) * cgrad4(i,j,juu,ju,jd,jdd,1,1,q) -
	 cgrad4(i,j,iuu,iu,id,idd,1,0,q) * cgrad4(i,j,juu,ju,jd,jdd,0,1,q))
	     << "\n";
    }
    writer << "\n";
  }
  writer.close();
}

// ============================================================================

// Member functions of LocalAlignWriter
LocalAlignWriter::LocalAlignWriter
(int _lx, int _ly, int ngx, int ngy, int lmin, int lmax, int linc, long freq,
 const string& file) :
  QFieldWriter(freq, file), lx(_lx), ly(_ly), ngridx(ngx), ngridy(ngy),
  minLen(lmin), maxLen(lmax), lenInc(linc) {
  nbins = static_cast<int>((maxLen-minLen)/lenInc)+1;
  double xwidth = lx/static_cast<double>(ngridx);
  double ywidth = ly/static_cast<double>(ngridy);
  gridx = create1DArray<int>(ngridx);
  gridy = create1DArray<int>(ngridy);
  for (int i = 0; i < ngridx; i++) {
    gridx[i] = static_cast<int>((i+0.5)*xwidth);
  }
  for (int i = 0; i < ngridy; i++) {
    gridy[i] = static_cast<int>((i+0.5)*ywidth);
  }  
  alignData = create3DArray<double>(lx,ly,3);
  alignOrderAvg = create2DArray<double>(nbins,3);
  alignOrderAvgSq = create2DArray<double>(nbins,3);
  ntimepts = 0;
  status = true;
}

LocalAlignWriter::~LocalAlignWriter() {
  writer.open(outFile);
  if (!writer) {
    cout << "Error: cannot open the file " << outFile << endl;
    status = false;
  } else {
    // Normalise
    for (int n = 0; n < nbins; n++) {
      for (int i = 0; i < norders; i++) {
	alignOrderAvg[n][i] /= ntimepts;
	alignOrderAvgSq[n][i] /= ntimepts;
      }
    }
    // Output results
    double stdev, stderr;
    for (int n = 0; n < nbins; n++) {
      writer << n*lenInc+minLen << " ";
      for (int i = 0; i < norders; i++) {
	error(ntimepts, alignOrderAvg[n][i], alignOrderAvgSq[n][i],
	      stdev, stderr);
	writer << alignOrderAvg[n][i] << " " << stdev << " " << stderr << " ";
      }
      writer << "\n";
    }
  }
  writer.close();
  
  // Clean up
  deleteArray(gridx);
  deleteArray(gridy);
  deleteArray(alignData);
  deleteArray(alignOrderAvg);
  deleteArray(alignOrderAvgSq);
}

void LocalAlignWriter::_sample(long time, const QField& qfield) {
  int len, x, y, axis, npts, ngridpts;
  double lam, lam0, lam1, vx, vy, cos2tavg, sin2tavg, wcos2tavg, wsin2tavg;
  double wavg, wavgAll, nemavgAll, wnemavgAll;
  Eigen eigen;
  // Compute the alignment order
  for (int i = 0; i < lx; i++) {
    for (int j = 0; j < ly; j++) {
      getEigen(qfield(i,j,0), qfield(i,j,1), qfield(i,j,2), eigen);
      lam0 = eigen.eigenVal.at(0);
      lam1 = eigen.eigenVal.at(1);
      lam = lam0 < lam1 ? lam1 : lam0;
      axis = lam0 < lam1 ? 1 : 0;
      vx = eigen.eigenVec(0,axis);
      vy = eigen.eigenVec(1,axis);
      alignData[i][j][0] = 2.0*vx*vx-1.0; // cos2t
      alignData[i][j][1] = 2.0*vx*vy; // sin2t
      alignData[i][j][2] = lam;
    }
  }
  // Compute average local alignment profile
  double sin2t, cos2t;
  for (int n = 0; n < nbins; n++) {
    len = n*lenInc+minLen;
    npts = static_cast<double>(len*len);
    ngridpts = ngridx*ngridy;
    nemavgAll = 0.0;
    wnemavgAll = 0.0;
    wavgAll = 0.0;
    for (int i = 0; i < ngridx; i++) {
      for (int j = 0; j < ngridy; j++) {
	cos2tavg = 0.0;
	sin2tavg = 0.0;
	wcos2tavg = 0.0;
	wsin2tavg = 0.0;
	wavg = 0.0;
	for (int k = 0; k < len; k++) {
	  x = iwrap(lx,gridx[i]-len/2+k);
	  for (int l = 0; l < len; l++) {
	    y = iwrap(ly,gridy[j]-len/2+l);
	    cos2t = alignData[x][y][0];
	    sin2t = alignData[x][y][1];
	    lam = alignData[x][y][2];
	    cos2tavg += cos2t;
	    sin2tavg += sin2t;
	    wcos2tavg += lam*cos2t;
	    wsin2tavg += lam*sin2t;
	    wavg += lam;
	  }
	}
	cos2tavg /= npts;
	sin2tavg /= npts;
	wcos2tavg /= npts;
	wsin2tavg /= npts;
	wavg /= npts;
	nemavgAll += sqrt(cos2tavg*cos2tavg+sin2tavg*sin2tavg);
	wnemavgAll += sqrt(wcos2tavg*wcos2tavg+wsin2tavg*wsin2tavg);
	wavgAll += wavg;
      } // Close loop over j (gridy)
    } // Close loop over i (gridx)
    nemavgAll /= ngridpts;
    wnemavgAll /= ngridpts;
    wavgAll /= ngridpts;
    alignOrderAvg[n][0] += wavgAll;
    alignOrderAvg[n][1] += nemavgAll;
    alignOrderAvg[n][2] += wnemavgAll;
    alignOrderAvgSq[n][0] += wavgAll*wavgAll;
    alignOrderAvgSq[n][1] += nemavgAll*nemavgAll;
    alignOrderAvgSq[n][2] += wnemavgAll*wnemavgAll;
  } // Close loop over bins
  ntimepts++;
}

inline void error(int n, double avg, double avgSq, double& dev, double& err) {
  if (n > 1) {
    dev = sqrt(n/(n-1.0)*(avgSq-avg*avg));
    err = dev/sqrt(n);
  } else {
    dev = 0.0;
    err = 0.0;
  }
}

// ============================================================================

// Member functions of NematicDefectWriter
NematicDefectWriter::NematicDefectWriter
(int _lx, int _ly, double mdist, int cdist, int bsize, long freq,
 const string& file) :
  QFieldWriter(freq, file), minDist(mdist), contourDist(cdist), binsize(bsize),
  lx(_lx), ly(_ly) {
  
  writer.open(file);
  if (!writer) {
    cout << "Error: cannot open the file " << file << endl;
    status = false;
  }
  
  // A square contour for summing angles when calculating defect charge
  for (int i = -contourDist; i < contourDist; i++) {
    contour.push_back({contourDist,i});
  }
  for (int i = contourDist; i > -contourDist; i--) {
    contour.push_back({i,contourDist});
  }
  for (int i = contourDist; i > -contourDist; i--) {
    contour.push_back({-contourDist,i});
  }
  for (int i = -contourDist; i < contourDist; i++) {
    contour.push_back({i,-contourDist});
  }

  // For storing the smoothed Q field and eigenvalues and eigenvectors
  smoothedField = create3DArray<double>(lx,ly,3);
  eigenVal = create2DArray<double>(lx,ly);
  eigenVec = create3DArray<double>(lx,ly,ndims);
  defectCharges = create2DArray<double>(lx,ly);

  status = true;
}

NematicDefectWriter::~NematicDefectWriter() {
  writer.close();
  deleteArray(smoothedField);
  deleteArray(eigenVal);
  deleteArray(eigenVec);
  deleteArray(defectCharges);
}

void NematicDefectWriter::_sample(long time, const QField& qfield) {
  defects.clear();

  // Smooth the field
  smooth(lx, ly, binsize, qfield.getField(), smoothedField);
  
  // Compute eigenvalues and eigenvectors
  Eigen eigen;
  double lam0, lam1;
  int axis;
  for (int i = 0; i < lx; i++) {
    for (int j = 0; j < ly; j++) {
      getEigen(smoothedField[i][j][0], smoothedField[i][j][1],
	       smoothedField[i][j][2], eigen);
      lam0 = eigen.eigenVal.at(0);
      lam1 = eigen.eigenVal.at(1);
      eigenVal[i][j] = lam0 < lam1 ? lam1 : lam0;
      axis = lam0 < lam1 ? 1 : 0;
      eigenVec[i][j][0] = eigen.eigenVec(0,axis);
      eigenVec[i][j][1] = eigen.eigenVec(1,axis);      
    }
  }

#pragma omp parallel for default(none),\
  shared(lx, ly, contour, defectCharges), schedule(static)
  for (int i = 0; i < lx; i++) {
    for (int j = 0; j < ly; j++) {
      // Check if it is a true defect by calculating topological charge
      // around the contour
      double angleSum = 0.0;
      int ic, jc, ip, jp, kup;
      double angle, pangle, dangle, xc, yc, xp, yp, q, dot;
      ip = iwrap(lx,i+contour[0][0]);
      jp = iwrap(ly,j+contour[0][1]);
      xp = eigenVec[ip][jp][0];
      yp = eigenVec[ip][jp][1];
      pangle = yp < 0.0 ? twopi-acos(xp) : acos(xp);
      for (size_t k = 0; k < contour.size(); k++) {
	kup = iup(contour.size(),k);
	ic = iwrap(lx,i+contour[kup][0]);
	jc = iwrap(ly,j+contour[kup][1]);
	xc = eigenVec[ic][jc][0];
	yc = eigenVec[ic][jc][1];
	dot = xc*xp+yc*yp;
	if (dot < 0.0) {
	  xc *= -1.0;
	  yc *= -1.0;
	}
	angle = yc < 0.0 ? twopi-acos(xc) : acos(xc);
	dangle = angle-pangle;
	if (dangle > pi) {
	  angle -= twopi;
	  dangle -= twopi;
	} else if (dangle < -pi) {
	  angle += twopi;
	  dangle += twopi;
	}
	angleSum += dangle;
	pangle = angle;
	ip = ic;
	jp = jc;
	xp = xc;
	yp = yc;
      } // Close loop over the contour
      q = angleSum / twopi;
      defectCharges[i][j] = q;
    }
  }
  
  for (int i = 0; i < lx; i++) {
    for (int j = 0; j < ly; j++) {
      if (fabs(defectCharges[i][j]) > minDefectCharge) {
	defects.push_back({i+0.5,j+0.5,defectCharges[i][j]});
      }
    }
  }

  // Group defects that are close together
  const double defectDistThresSq = minDist*minDist;
  defectClustStartIndex.clear();
  // Reset link list
  defectClustList = vector<size_t>(defects.size());
  for (size_t i = 0; i < defects.size(); i++) {
    defectClustList[i] = i;
  }
  // Do clustering
  double vj[ndims];
  double vk[ndims];
  double boxsize[ndims] = {(double) lx, (double) ly};
  for (size_t i = 0; i < defects.size(); i++) {
    if (i == defectClustList[i]) {
      size_t j = i;
      defectClustStartIndex.push_back(i);
      do {
	vj[0] = defects[j].x;
	vj[1] = defects[j].y;
	for (size_t k = i+1; k < defects.size(); k++) {
	  if (k != defectClustList[k]) continue;
	  vk[0] = defects[k].x;
	  vk[1] = defects[k].y;
	  if (ddistsq(ndims, boxsize, vj, vk) < defectDistThresSq &&
	      fabs(defects[j].q-defects[k].q) < 1e-3) {
	    size_t temp = defectClustList[j];
	    defectClustList[j] = defectClustList[k];
	    defectClustList[k] = temp;
	  }
	}
	j = defectClustList[j];
      } while (j != i);
    }
  }
  
  // Output the results
  writer << "Defects: " << defectClustStartIndex.size() << "\n";
  writer << "Timestep: " << time << "\n";
  
  // Average positions of defects in the same cluster
  double halflx = lx*0.5;
  double halfly = ly*0.5;
  size_t idx, count;
  double xcm, ycm, xshift, yshift, qavg;
  for (size_t i = 0; i < defectClustStartIndex.size(); i++) {
    idx = defectClustStartIndex[i];
    xcm = defects[idx].x;
    ycm = defects[idx].y;
    xshift = ddiff(lx,halflx,xcm);
    yshift = ddiff(ly,halfly,ycm);
    count = 1;
    qavg = defects[idx].q;
    idx = defectClustList[idx];
    while (idx != defectClustStartIndex[i]) {
      xcm = dwrap(lx,xcm+xshift);
      ycm = dwrap(ly,ycm+yshift);
      xcm *= count;
      ycm *= count;
      xcm += dwrap(lx,defects[idx].x+xshift);
      ycm += dwrap(ly,defects[idx].y+yshift);
      count++;
      xcm /= count;
      ycm /= count;
      xcm = dwrap(lx,xcm-xshift);
      ycm = dwrap(ly,ycm-yshift);
      xshift = ddiff(lx,halflx,xcm);
      yshift = ddiff(ly,halfly,ycm);
      qavg += defects[idx].q;
      idx = defectClustList[idx];
    }
    writer << xcm << " " << ycm << " " << qavg/count << "\n";
  }  
}

void smooth(int lx, int ly, int binsize, double*** field, double*** smoothed) {
  int start = -binsize/2;
  int end = start+binsize;
  int n = binsize*binsize;
#pragma omp parallel for default(none) \
  shared(lx, ly, start, end, n, field, smoothed)
  for (int i = 0; i < lx; i++) {
    for (int j = 0; j < ly; j++) {
      double sumxx = 0.0;
      double sumyy = 0.0;
      double sumxy = 0.0;
      int ki, lj;
      for (int k = start; k < end; k++) {
	ki = iwrap(lx,i+k);
	for (int l = start; l < end; l++) {
	  lj = iwrap(ly,j+l);
	  sumxx += field[ki][lj][0];
	  sumyy += field[ki][lj][1];
	  sumxy += field[ki][lj][2];
	}
      }
      smoothed[i][j][0] = sumxx/n;
      smoothed[i][j][1] = sumyy/n;
      smoothed[i][j][2] = sumxy/n;
    }
  }
}
