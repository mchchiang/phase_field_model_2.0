// qfield_writer.hpp

#ifndef QFIELD_WRITER_HPP
#define QFIELD_WRITER_HPP

#include <fstream>
#include <string>
#include <vector>
#include <memory>

// For minimisation and interpolation
/*#include <gsl/gsl_math.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>*/

#include "qfield.hpp"

class QFieldWriter;
using Qwrite = std::shared_ptr<QFieldWriter>;

class QFieldWriter {
protected:
  long sampFreq;
  std::string outFile;
  std::ofstream writer;
  bool status = false;
public:
  QFieldWriter(long sampFreq, const std::string& outFile);
  virtual ~QFieldWriter();
  bool getStatus();
  void sample(const QField& qfield); // This checks dump freq
  virtual void _sample(long time, const QField& qfield) = 0;

  // Static factory methods
  static Qwrite makeGlobalObsWriter
  (long sampFreq, const std::string& outFile);
  static Qwrite makeSmoothQFieldWriter
  (bool overwrite, long sampFreq, const std::string& outFile);
  static Qwrite makeDirectorWriter
  (int lx, int ly, int binsize, bool overwrite, long sampFreq,
   const std::string& outFile);
  static Qwrite makeTopoChargeWriter
  (bool overwrite, long sampFreq, const std::string& outFile);
  static Qwrite makeLocalAlignWriter
  (int lx, int ly, int ngridx, int ngridy, int minLen, int maxLen, int lenInc,
   long sampFreq, const std::string& outFile);
  static Qwrite makeNematicDefectWriter
  (int lx, int ly, double minDist, int contourDist, int binsize, long sampFreq,
   const std::string& outFile);
};

class GlobalObsWriter : public QFieldWriter {
public:
  GlobalObsWriter(long sampFreq, const std::string& outFile);
  ~GlobalObsWriter();
  void _sample(long time, const QField& qfield);
};

class SmoothQFieldWriter : public QFieldWriter {
private:
  bool overwrite;
public:
  SmoothQFieldWriter(bool overwrite, long sampFreq,
		     const std::string& outFile);
  ~SmoothQFieldWriter();
  void _sample(long time, const QField& qfield);
};

class DirectorWriter : public QFieldWriter {
  int lx, ly, binsize; // For smoothing the field
  bool overwrite;
  double*** smoothedField = nullptr;
public:
  DirectorWriter(int lx, int ly, int binsize, bool overwrite, long sampFreq,
		 const std::string& outFile);
  ~DirectorWriter();
  void _sample(long time, const QField& qfield);
};

class TopoChargeWriter : public QFieldWriter {
  bool overwrite;
public:
  TopoChargeWriter(bool overwrite, long sampFreq, const std::string& outFile);
  ~TopoChargeWriter();
  void _sample(long time, const QField& qfield);
};

class LocalAlignWriter : public QFieldWriter {
private:
  int lx, ly, ngridx, ngridy, minLen, maxLen, lenInc, nbins, ntimepts;
  int* gridx = nullptr;
  int* gridy = nullptr;
  double*** alignData = nullptr;
  double** alignOrderAvg = nullptr;
  double** alignOrderAvgSq = nullptr;
  const int norders = 3; // Number of alignment observables
public:
  LocalAlignWriter
  (int lx, int ly, int ngridx, int ngridy, int minLen, int maxLen, int lenInc,
   long sampFreq, const std::string& outFile);
  ~LocalAlignWriter();
  void _sample(long time, const QField& qfield);
};

class NematicDefectWriter : public QFieldWriter {
private:
  /*struct FuncParams {
    const int nparams = 3;
    gsl_interp_accel* xacc;
    gsl_interp_accel* yacc;
    gsl_spline2d* spline;
    };*/
  struct Defect {
    double x;
    double y;
    double q;
  };
  
  std::vector<Defect> defects; // Store the defect locations
  
  // Variables for 2d inteporlation and minimisation
  int ngridpts; // Number of gridpts for finding minima
  int minDist; // Minimum separation between defects
  int contourDist; // (Half) size of the contour for calculating topo charge
  int binsize; // Smoothing window size for the Q field
  /*const int maxIter = 1000;
  const double minDelta = 1e-10;
  const double startStepsize = 0.01;
  std::vector<double> gridpt;
  std::vector<double> gridVal;  
  std::vector<double> startpt;
  std::vector<double> startstep;
  const gsl_interp2d_type* interpType = gsl_interp2d_bicubic;
  gsl_spline2d* spline;
  gsl_interp_accel* xacc;
  gsl_interp_accel* yacc;
  const gsl_multimin_fminimizer_type* minimiseType =
    gsl_multimin_fminimizer_nmsimplex2;
  gsl_multimin_fminimizer* minimiser;
  gsl_vector* minpt;
  gsl_vector* step;
  gsl_multimin_function minexFunc;
  FuncParams minexFuncParams;*/
  
  const int ndims = 2;
  const double pi = M_PI;
  const double twopi = 2.0*pi;
  const double minDefectCharge = 0.495;

  // A square contour for calculating defect charge
  std::vector<std::vector<int> > contour;

  // Store the field and smooothed field
  int lx, ly;
  double** eigenVal = nullptr;
  double*** eigenVec = nullptr;
  double*** smoothedField = nullptr;
  double** defectCharges = nullptr;
  std::vector<size_t> defectClustStartIndex;
  std::vector<size_t> defectClustList;
  
public:
  NematicDefectWriter(int lx, int ly, double minDist, int contourDist,
		      int binsize, long sampFreq, const std::string& outFile);
  ~NematicDefectWriter();
  void _sample(long time, const QField& qfield);
  void cluster();
};

#endif
