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
#include "Eigen_ext/SparseBandedBlockedQR_General.h"

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

	std::random_device rd;
	std::mt19937 genmt(rd());
	const int numIdxDists = 256;
	const int overlap = 2;
	int step = numParams / numIdxDists;
	std::uniform_int_distribution<int> idxDists[numIdxDists];
	for (int i = 0; i < numIdxDists; i++) {
		if(i < numIdxDists - 1) {
			idxDists[i] = std::uniform_int_distribution<int>(i * step, (i + 1) * step - 1 + overlap);
		}
		else {
			idxDists[i] = std::uniform_int_distribution<int>(i * step, (i + 1) * step - 1);
		}
 	}
	std::uniform_int_distribution<int> drawIdxDist(0, numIdxDists - 1);

	clock_t begin;

	typedef SparseMatrix<Scalar, ColMajor, SuiteSparse_long> JacobianType;
	typedef SparseMatrix<Scalar, RowMajor, int> JacobianTypeRowMajor;
	typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
	typedef SparseQR_Ext<JacobianType, NaturalOrdering<int> > GeneralQRSolver;
	//typedef SparseBandedBlockedQR_Ext<JacobianType, NaturalOrdering<int> > BandedBlockedQRSolver;
	typedef SparseBandedBlockedQR_General<JacobianType, NaturalOrdering<int>, 8> BandedBlockedQRSolver;
	typedef SPQR<JacobianType> SPQRSolver;
 
	/*
	 * Set-up the problem to be solved
	*/
	/*
	int nnzPerRow = 12;
	JacobianType spJ;
	Eigen::TripletArray<Scalar, typename JacobianType::Index> jvals(nnzPerRow * numResiduals);
	for (int i = 0; i < numResiduals; i++) {
		std::uniform_int_distribution<int> currDist = idxDists[drawIdxDist(gen)];
		for (int j = 0; j < nnzPerRow; j++) {
			jvals.add(i, currDist(genmt), dist(gen));
		}
	}
	*/
	///*
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
	//*/
	/*
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
		}
	}
	*/
	spJ.resize(numResiduals, numParams);
	spJ.setFromTriplets(jvals.begin(), jvals.end());
	spJ.makeCompressed();

	JacobianType I(spJ.rows(), spJ.rows());
	I.setIdentity();
	///*
	// Permute Jacobian rows (we want to see how our QR handles a general matrix)	
	PermutationMatrix<Dynamic, Dynamic, SuiteSparse_long> perm(spJ.rows());
	perm.setIdentity();
	std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
	spJ = perm * spJ;

#if !defined(_DEBUG) && defined(OUTPUT_MAT)
	Logger::instance()->logMatrixCSV(spJ.toDense(), "slvrJ.csv");
#endif

//#if !defined(_DEBUG) && defined(OUTPUT_MAT)
//	Logger::instance()->logMatrixCSV(spJ.toDense(), "slvrJ_perm.csv");
//#endif
	//*/

	std::cout << "Problem size (r x c): " << spJ.rows() << " x " << spJ.cols() << std::endl;
	std::cout << "####################################################" << std::endl;

	int nVecEvals = 1000;

	/*
	 * Solve the problem using SuiteSparse QR.
	*/
	std::cout << "Solver: SPQR" << std::endl;
	std::cout << "---------------------- Timing ----------------------" << std::endl;
	SPQRSolver spqr;
	begin = clock();
	spqr.compute(spJ);
	std::cout << "Factorization:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	
	std::cout << "Express full Q: " << std::endl;
	begin = clock();
	SPQRSolver::MatrixType QSP(spJ.rows(), spJ.rows());
	QSP = spqr.matrixQ() * I;
	std::cout << "matrixQ()   * I: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";

	begin = clock();
	SPQRSolver::MatrixType QtSP(spJ.rows(), spJ.rows());
	QtSP = spqr.matrixQ().transpose() * I;
	std::cout << "matrixQ().T * I: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";

	std::cout << "Solve LS: " << std::endl;
	JacobianType slvrVecSP(spJ.rows(), 1);
	for (int i = 0; i < slvrVecSP.rows(); i++) {
		for (int j = 0; j < slvrVecSP.cols(); j++) {
			slvrVecSP.coeffRef(i, j) = dist(gen);
		}
	}
	begin = clock();
	JacobianType resSP;
	for(int i = 0; i < nVecEvals; i++) {
		resSP = spqr.matrixQ() * slvrVecSP;
	}
	std::cout << "matrixQ()   * v: " << double(clock() - begin) / CLOCKS_PER_SEC << "s (eval " << nVecEvals << "x) \n";

	begin = clock();
	VectorXd resDenseSP = resSP.toDense();
	VectorXd solvedSP;
	for (int i = 0; i < nVecEvals; i++) {
		solvedSP = spqr.matrixR().template triangularView<Upper>().solve(resDenseSP);
	}
	std::cout << "matrixR() \\ res: " << double(clock() - begin) / CLOCKS_PER_SEC << "s (eval " << nVecEvals << "x) \n";

	std::cout << "Q non-zeros: " << QSP.nonZeros() << " (" << (QSP.nonZeros() / double(QSP.rows() * QSP.cols())) * 100 << "%)" << std::endl;
	std::cout << "---------------------- Errors ----------------------" << std::endl;
	std::cout << "||Q    * R - J||_2 = " << (QSP * spqr.matrixR() - spJ).norm() << std::endl;
	std::cout << "||Q.T  * J - R||_2 = " << (QSP.transpose() * spJ - spqr.matrixR()).norm() << std::endl;
	//std::cout << "||Qt.T * R - J||_2 = " << (QtSP.transpose() * spqr.matrixR() - spJ).norm() << std::endl;
	//std::cout << "||Qt   * J - R||_2 = " << (QtSP * spJ - spqr.matrixR()).norm() << std::endl;
	//std::cout << "||Qt * Q - I||_2 = " << (QSP.transpose() * QSP - I).norm() << std::endl;
	std::cout << "####################################################" << std::endl;
	
	/*
	* Solve the problem using special banded QR solver.
	*/
	std::cout << "Solver: General Banded Blocked QR" << std::endl;
	std::cout << "---------------------- Timing ----------------------" << std::endl;
	BandedBlockedQRSolver slvr;

	/*
	// Only analyze pattern test
	slvr.analyzePattern(spJ);

	Logger::instance()->logMatrixCSV(spJ.toDense(), "slvrJ_perm_input.csv");
	spJ = (slvr.rowsPermutation() * spJ);
	Logger::instance()->logMatrixCSV(spJ.toDense(), "slvrJ_perm_rowonly.csv");
	spJ = spJ * slvr.colsPermutation();
	Logger::instance()->logMatrixCSV(spJ.toDense(), "slvrJ_perm_output.csv");

	return 0;
	*/

	begin = clock();
	slvr.compute(spJ);
	std::cout << "Factorization:   " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	spJ = slvr.rowsPermutation() * spJ;
	
	std::cout << "Express full Q: " << std::endl;
	begin = clock();
	JacobianType slvrQ(spJ.rows(), spJ.rows());
	slvrQ = slvr.matrixQ() * I;
	std::cout << "matrixQ()   * I: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";

	begin = clock();
	JacobianType slvrQt(spJ.rows(), spJ.rows());
	slvrQt = slvr.matrixQ().transpose() * I;
	std::cout << "matrixQ().T * I: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";

	std::cout << "Solve LS: " << std::endl;
	Eigen::VectorXd slvrVecDense = Eigen::VectorXd::Random(spJ.rows());
	JacobianType slvrVec(spJ.rows(), 1);
	for (int i = 0; i < slvrVec.rows(); i++) {
		for (int j = 0; j < slvrVec.cols(); j++) {
			slvrVec.coeffRef(i, j) = dist(gen);
		}
	}
	begin = clock();
	Eigen::VectorXd slvrResDense;
	for(int i = 0; i < nVecEvals; i++) {
		slvrResDense = slvr.matrixQ() * slvrVecDense;//slvrVec;
	}
	std::cout << "matrixQ()   * v: " << double(clock() - begin) / CLOCKS_PER_SEC << "s (eval " << nVecEvals << "x) \n";

	begin = clock();
	VectorXd solved;
	for (int i = 0; i < nVecEvals; i++) {
		solved = slvr.matrixR().template triangularView<Upper>().solve(slvrResDense);
	}
	std::cout << "matrixR() \\ res: " << double(clock() - begin) / CLOCKS_PER_SEC << "s (eval " << nVecEvals << "x) \n";

	std::cout << "Q non-zeros: " << slvrQ.nonZeros() << " (" << (slvrQ.nonZeros() / double(slvrQ.rows() * slvrQ.cols())) * 100 << "%)" << std::endl;
	std::cout << "---------------------- Errors ----------------------" << std::endl;
	std::cout << "||Q    * R - J||_2 = " << (slvrQ * slvr.matrixR() - spJ).norm() << std::endl;
	std::cout << "||Q.T  * J - R||_2 = " << (slvrQ.transpose() * spJ - slvr.matrixR()).norm() << std::endl;
	std::cout << "||Qt.T * R - J||_2 = " << (slvrQt.transpose() * slvr.matrixR() - spJ).norm() << std::endl;
	std::cout << "||Qt   * J - R||_2 = " << (slvrQt * spJ - slvr.matrixR()).norm() << std::endl;
	//std::cout << "||Qt * Q - I||_2 = " << (slvrQ.transpose() * slvrQ - I).norm() << std::endl;
	std::cout << "####################################################" << std::endl;

	return 0;

#if !defined(_DEBUG) && defined(OUTPUT_MAT)
	Logger::instance()->logMatrixCSV(slvrQ.toDense(), "slvrQ.csv");
	//Logger::instance()->logMatrixCSV(slvr.matrixY().toDense(), "slvrY.csv");
	//Logger::instance()->logMatrixCSV(slvr.matrixT().toDense(), "slvrT.csv");
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
