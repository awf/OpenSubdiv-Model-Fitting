#define _wassert wassert_awf
#include <cassert>

#include <iostream>
#include <iomanip>
#include <ctime>

#include <future>

#include <random>

#include <Eigen/Eigen>
#include <Eigen/SparseQR>

#include <Eigen/SparseCore>
#include <Eigen/src/Core/util/DisableStupidWarnings.h>
#include <suitesparse/SuiteSparseQR.hpp>
#include <Eigen/src/CholmodSupport/CholmodSupport.h>
#include "Eigen_ext/SuiteSparseQRSupport_Ext.h"

#include "Eigen_ext/eigen_extras.h"
#include "Eigen_ext/SparseQR_Ext.h"
#include "Eigen_ext/BlockSparseQR_Ext.h"
#include "Eigen_ext/BlockDiagonalSparseQR_Ext.h"
#include "Eigen_ext/SparseSubblockQR_Ext.h"
#include "Eigen_ext/SparseBandedBlockedQR_Ext.h"
#include "Eigen_ext/SparseBandedBlockedQR_Ext3.h"

#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/LevenbergMarquardt>
#include <unsupported/Eigen/SparseExtra>
#include "unsupported/Eigen/src/SparseExtra/BlockSparseQR.h"
#include "unsupported/Eigen/src/SparseExtra/BlockDiagonalSparseQR.h"

#include "Logger.h"

using namespace Eigen;

//#define OUTPUT_MAT 1

int main() {
	Logger::createLogger("runtime_log.log");
	Logger::instance()->log(Logger::Info, "QR Banded Test STARTED!");

	std::default_random_engine gen;
	std::uniform_real_distribution<double> dist(0.5, 5.0);

	Eigen::Index numVars = 1024;
	Eigen::Index numParams = numVars * 2;
	Eigen::Index numResiduals = numVars * 3 + numVars + numVars * 3;

	clock_t begin;

	typedef SparseMatrix<Scalar, ColMajor, SuiteSparse_long> JacobianType;
	typedef SparseMatrix<Scalar, RowMajor, int> JacobianTypeRowMajor;
	typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
	typedef SparseQR_Ext<JacobianType, NaturalOrdering<int> > GeneralQRSolver;
	//typedef SparseBandedBlockedQR_Ext<JacobianType, NaturalOrdering<int> > BandedBlockedQRSolver;
	typedef SparseBandedBlockedQR_Ext3<JacobianType, NaturalOrdering<int> > BandedBlockedQRSolver;
	typedef SPQR<JacobianType> SPQRSolver;

	/*
	 * Set-up the problem to be solved
	*/
	int stride = 7;
	JacobianType spJ;
	Eigen::TripletArray<Scalar, typename JacobianType::Index> jvals(stride * numParams);
	for (int i = 0; i < numParams; i++) {
		for (int j = i * 2; j < (i * 2) + 2 && j < numParams; j++) {
			jvals.add(i * stride, j, dist(gen));
			jvals.add(i * stride + 1, j, dist(gen));
			jvals.add(i * stride + 2, j, dist(gen));
			jvals.add(i * stride + 3, j, dist(gen));
			jvals.add(i * stride + 4, j, dist(gen));
			jvals.add(i * stride + 5, j, dist(gen));
			jvals.add(i * stride + 6, j, dist(gen));
			if (j < numParams - 2) {
				jvals.add(i * stride + 6, j + 2, dist(gen));
			}
		}
	}
	spJ.resize(numResiduals, numParams);
	spJ.setFromTriplets(jvals.begin(), jvals.end());
	spJ.makeCompressed();

	JacobianType I(spJ.rows(), spJ.rows());
	I.setIdentity();

#if !defined(_DEBUG) && defined(OUTPUT_MAT)
	Logger::instance()->logMatrixCSV(spJ.toDense(), "slvrJ.csv");
#endif

	std::cout << "Problem size (r x c): " << spJ.rows() << " x " << spJ.cols() << std::endl;
	std::cout << "####################################################" << std::endl;

	/*
	 * Solve the problem using SuiteSparse QR.
	*/
	/*
	std::cout << "Solver: SPQR" << std::endl;
	std::cout << "---------------------- Timing ----------------------" << std::endl;
	SPQRSolver spqr;
	begin = clock();
	spqr.compute(spJ);
	std::cout << "Factorization:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";

	begin = clock();
	SPQRSolver::MatrixType QtSP(spJ.rows(), spJ.rows());
	QtSP.setIdentity();
	QtSP = spqr.matrixQ().transpose() * QtSP;
	std::cout << "matrixQ().T * I: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";

	begin = clock();
	SPQRSolver::MatrixType QSP(spJ.rows(), spJ.rows());
	QSP.setIdentity();
	QSP = spqr.matrixQ() * QSP;
	std::cout << "matrixQ()   * I: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";

	std::cout << "---------------------- Errors ----------------------" << std::endl;
	std::cout << "||Q    * R - J||_2 = " << (QSP * spqr.matrixR() - spJ).norm() << std::endl;
	std::cout << "||Q.T  * J - R||_2 = " << (QSP.transpose() * spJ - spqr.matrixR()).norm() << std::endl;
	//std::cout << "||Qt.T * R - J||_2 = " << (QtSP.transpose() * spqr.matrixR() - spJ).norm() << std::endl;
	//std::cout << "||Qt   * J - R||_2 = " << (QtSP * spJ - spqr.matrixR()).norm() << std::endl;
	//std::cout << "||Qt * Q - I||_2 = " << (QSP.transpose() * QSP - I).norm() << std::endl;
	std::cout << "####################################################" << std::endl;
	*/
	/*
	* Solve the problem using special banded QR solver.
	*/
	const Index blockRows = 35;//14;//21;//105;//35;
	const Index blockCols = 12;//6;//8;//32;//12;
	const Index blockOverlap = 2;
	std::cout << "Solver: Banded Blocked QR (r = " << blockRows << ", c = " << blockCols << ", o = " << blockOverlap << ")" << std::endl;
	std::cout << "---------------------- Timing ----------------------" << std::endl;
	BandedBlockedQRSolver slvr(blockRows, blockCols, blockOverlap);
	slvr.setRoundoffEpsilon(1e-16);

	begin = clock();
	slvr.compute(spJ);
	std::cout << "Factorization:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";

	begin = clock();
	JacobianType slvrQ(spJ.rows(), spJ.rows());
	slvrQ.setIdentity();
	slvrQ = slvr.matrixQ() * slvrQ;
	std::cout << "matrixQ()   * I: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";

	begin = clock();
	JacobianType slvrQt(spJ.rows(), spJ.rows());
	slvrQt.setIdentity();
	slvrQt = slvr.matrixQ().transpose() * slvrQt;
	std::cout << "matrixQ().T * I: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";


	/*
	JacobianType slvrVec(spJ.rows(), 384);
	for (int i = 0; i < slvrVec.rows(); i++) {
		for (int j = 0; j < slvrVec.cols(); j++) {
			slvrVec.coeffRef(i, j) = dist(gen);
		}
	}
	begin = clock();
	slvrVec = slvrQ * slvrVec;
	std::cout << "Slvr vec elapsed: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	*/

	std::cout << "---------------------- Errors ----------------------" << std::endl;
	std::cout << "||Q    * R - J||_2 = " << (slvrQ * slvr.matrixR() - spJ).norm() << std::endl;
	std::cout << "||Q.T  * J - R||_2 = " << (slvrQ.transpose() * spJ - slvr.matrixR()).norm() << std::endl;
	std::cout << "||Qt.T * R - J||_2 = " << (slvrQt.transpose() * slvr.matrixR() - spJ).norm() << std::endl;
	std::cout << "||Qt   * J - R||_2 = " << (slvrQt * spJ - slvr.matrixR()).norm() << std::endl;
	//std::cout << "||Qt * Q - I||_2 = " << (slvrQ.transpose() * slvrQ - I).norm() << std::endl;
	std::cout << "####################################################" << std::endl;

#if !defined(_DEBUG) && defined(OUTPUT_MAT)
	Logger::instance()->logMatrixCSV(slvrQ.toDense(), "slvrQ.csv");
	Logger::instance()->logMatrixCSV(slvr.matrixY().toDense(), "slvrY.csv");
	Logger::instance()->logMatrixCSV(slvr.matrixT().toDense(), "slvrT.csv");
	Logger::instance()->logMatrixCSV(slvr.matrixR().toDense(), "slvrR.csv");
#endif

	return 0;
}

// Override system assert so one can set a breakpoint in it rather than clicking "Retry" and "Break"
void __cdecl _wassert(_In_z_ wchar_t const* _Message, _In_z_ wchar_t const* _File, _In_ unsigned _Line)
{
	std::wcerr << _File << "(" << _Line << "): ASSERT FAILED [" << _Message << "]\n";

	abort();
}
