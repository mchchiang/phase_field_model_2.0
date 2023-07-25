// qfield.hpp

#ifndef QFIELD_HPP
#define QFIELD_HPP

#include <fstream>
#include <string>
#include <memory>
#include <functional>
#include "position.hpp"

class QField;
using Qfield = std::shared_ptr<QField>;

class QField {
protected:
  int lx, ly;
  double*** q = nullptr;
  long startTime, endTime, timeInc;
  long time = 0;
  bool trace = true;
  bool status = false;
  const int ncomps = 3; // qxx qyy qxy (lam0-lam1)/(lam0+lam1) (q is traceless)
public:
  QField(int lx, int ly, long startTime, long endTime, long timeInc,
	 bool useTrace);
  virtual ~QField();
  double& operator() (int i, int j, int comp);
  double operator() (int i, int j, int comp) const;
  void resetField();
  long getStartTime() const;
  long getTime() const;
  bool getStatus() const;
  int getLx() const;
  int getLy() const;
  double*** getField() const;
  virtual bool nextFrame() = 0;

  // Static factory methods
  static Qfield makeShapeSmoothQField
  (int npoints, int lx, int ly, int cellLx, int cellLy, double coeff,
   double thickness, long startTime, long endTime, long timeInc,
   bool useUniformShape, bool useTrace, const std::string& mode,
   const std::string& posFile, const std::string& matFile);
  static Qfield makeBoxSmoothQField
  (int lx, int ly, int gridx, int gridy, double coeff, long startTime,
   long endTime, long timeInc, bool useTrace, const std::string& matFile);
  static Qfield makeSmoothQField
  (int lx, int ly, double coeff, long startTime, long endTime, long timeInc,
   bool useTrace, const std::string& matFile);
};

class ShapeSmoothQField : public QField {
private:
  int npoints, cellLx, cellLy;
  double coeff, thickness;
  bool useUniformShape = false;
  std::string posFile, matFile;
  std::ifstream matReader;
  PositionReader posReader;
  double** s = nullptr;
  std::function<double(double,double,double,double,double)> shapeFunc;
public:
  ShapeSmoothQField
  (int npoints, int lx, int ly, int cellLx, int cellLy, double coeff,
   double thickness, long startTime, long endTime, long timeInc,
   bool useUniformShape, bool useTrace, const std::string& mode,
   const std::string& posFile, const std::string& matFile);
  virtual ~ShapeSmoothQField();
  bool nextFrame();
};

class BoxSmoothQField : public QField {
private:
  int gridx, gridy;
  double coeff;
  std::string matFile;
  double*** rawq = nullptr;
public:
  BoxSmoothQField
  (int lx, int ly, int gridx, int gridy, double coeff, long startTime,
   long endTime, long timeInc, bool useTrace, const std::string& matFile);
  virtual ~BoxSmoothQField();
  bool nextFrame();
};


class SmoothQField : public QField {
private:
  double coeff;
  std::string matFile;
public:
  SmoothQField(int lx, int ly, double coeff, long startTime, long endTime,
	       long timeInc, bool useTrace, const std::string& matFile);
  virtual ~SmoothQField();
  bool nextFrame();
};

#endif
