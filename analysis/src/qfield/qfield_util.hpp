// qfield_util.hpp
// Some useful helper structures and functions

#ifndef QFIELD_UTIL_HPP
#define QFIELD_UTIL_HPP

#include <armadillo>

struct Eigen {
  arma::mat matrix = arma::mat(2,2);
  arma::vec eigenVal = arma::vec(2);
  arma::mat eigenVec = arma::mat(2,2);
};

int iup(int len, int i);
int idown(int len, int i);
int iwrap(int len, int i);
double ddiff(double len, double x1, double x2);
double dwrap(double len, double x);
double ddistsq(int ndims, double* len, double* v1, double* v2);
double cgrad4(int i, int j, int uu, int u, int d, int dd, int ic, int oc,
	      double*** field);
void getEigen(double mxx, double myy, double mxy, Eigen& eigen);

#endif
