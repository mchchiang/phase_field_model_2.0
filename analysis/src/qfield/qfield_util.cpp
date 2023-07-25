// qfield_util.cpp

#include "qfield_util.hpp"

int iup(int len, int i) {
  return (i+1 >= len) ? 0 : i+1;
}

int idown(int len, int i) {
  return (i-1 < 0) ? len-1 : i-1;
}

int iwrap(int len, int i) {
  int remainder = i % len;
  return remainder >= 0 ? remainder : len + remainder;
}

double ddiff(double len, double x1, double x2) {
  double dx = x1-x2;
  return dx-round(dx/len)*len;
}

double dwrap(double len, double x) {
  double remainder = fmod(x,len);
  return remainder >= 0.0 ? remainder : len + remainder;
}

double ddistsq(int ndims, double* len, double* v1, double* v2) {
  double sum = 0.0;
  double dx;
  for (int i = 0; i < ndims; i++) {
    dx = ddiff(len[i], v1[i], v2[i]);
    sum += dx*dx;
  }
  return sum;
}

double cgrad4(int i, int j, int uu, int u, int d, int dd, int ic, int oc,
	      double*** field) {
  switch (oc) {
  case 0: return (-field[uu][j][ic] +
                  8.0 * (field[u][j][ic] - field[d][j][ic]) +
                  field[dd][j][ic]) / 12.0;
  case 1: return (-field[i][uu][ic] +
                  8.0 * (field[i][u][ic] - field[i][d][ic]) +
                  field[i][dd][ic]) / 12.0;
  default: return 0.0;
  }
}

void getEigen(double mxx, double myy, double mxy, Eigen& eigen) {
  eigen.matrix.at(0,0) = mxx;
  eigen.matrix.at(0,1) = mxy;
  eigen.matrix.at(1,0) = mxy;
  eigen.matrix.at(1,1) = myy;
  // Note that eigenvectors are stored in columns, not rows
  eig_sym(eigen.eigenVal, eigen.eigenVec, eigen.matrix);
}
