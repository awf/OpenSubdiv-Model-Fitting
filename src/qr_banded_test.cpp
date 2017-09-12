#define _wassert wassert_awf
#include <cassert>

#include <iostream>
#include <iomanip>
#include <ctime>

#include <random>

#include <Eigen/Eigen>
#include <Eigen/SparseQR>

#include <Eigen/SparseCore>

#include "Eigen_ext/eigen_extras.h"
#include "Eigen_ext/SparseQR_Ext.h"
#include "Eigen_ext/BlockSparseQR_Ext.h"
#include "Eigen_ext/BlockDiagonalSparseQR_Ext.h"
#include "Eigen_ext/SparseSubblockQR_Ext.h"
#include "Eigen_ext/SparseBandedQR_Ext.h"

#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/LevenbergMarquardt>
#include <unsupported/Eigen/SparseExtra>
#include "unsupported/Eigen/src/SparseExtra/BlockSparseQR.h"
#include "unsupported/Eigen/src/SparseExtra/BlockDiagonalSparseQR.h"

#include "Logger.h"

using namespace Eigen;

int main() {
	Logger::createLogger("runtime_log.log");
	Logger::instance()->log(Logger::Info, "QR Diagonalization Test STARTED!");

	std::default_random_engine gen;
	std::uniform_real_distribution<double> dist(0.0, 1.0);

	Eigen::Index numVars = 1024;
	Eigen::Index numParams = numVars * 2;
	Eigen::Index numResiduals = numVars * 3 + numVars;

	typedef SparseMatrix<Scalar, ColMajor, int> JacobianType;
	typedef SparseMatrix<Scalar, RowMajor, int> JacobianTypeRowMajor;
	typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
	//typedef SparseQR_Ext<JacobianType, COLAMDOrdering<int> > GeneralQRSolver;
	typedef SparseQR_Ext<JacobianType, NaturalOrdering<int> > GeneralQRSolver;
	typedef SparseBandedQR_Ext<JacobianType, NaturalOrdering<int> > BandedQRSolver;

	JacobianTypeRowMajor spJ;
	Eigen::TripletArray<Scalar, typename JacobianType::Index> jvals(64);
	///*
	for (int i = 0; i < numParams; i++) {
		for (int j = i * 2; j < (i * 2) + 2 && j < numParams; j++) {
			jvals.add(i * 4, j, dist(gen));
			jvals.add(i * 4 + 1, j, dist(gen));
			jvals.add(i * 4 + 2, j, dist(gen));
			jvals.add(i * 4 + 3, j, dist(gen));
			if (j < numParams - 2) {
				jvals.add(i * 4 + 3, j + 2, dist(gen));
			}
		}
	}
	spJ.resize(numResiduals, numParams);
	//*/
	/*
	jvals.add(0, 0, 1); jvals.add(0, 1, 2);
	jvals.add(1, 0, 5); jvals.add(1, 1, 1);
	jvals.add(2, 0, 9); jvals.add(2, 1, 7);
	jvals.add(3, 0, 1); jvals.add(3, 1, 3); jvals.add(3, 2, 4); jvals.add(3, 3, 4);
	jvals.add(4, 2, 6); jvals.add(4, 3, 7);
	jvals.add(5, 2, 8); jvals.add(5, 3, 2);
	jvals.add(6, 2, 5); jvals.add(6, 3, 3);
	jvals.add(7, 2, 6); jvals.add(7, 3, 9); jvals.add(7, 4, 1); jvals.add(7, 5, 1);
	jvals.add(8, 4, 3); jvals.add(8, 5, 2);
	jvals.add(9, 4, 7); jvals.add(9, 5, 9);
	jvals.add(10, 4, 1); jvals.add(10, 5, 4);
	jvals.add(11, 4, 6); jvals.add(11, 5, 9); jvals.add(11, 6, 7); jvals.add(11, 7, 8);
	jvals.add(12, 6, 3); jvals.add(12, 7, 2);
	jvals.add(13, 6, 7); jvals.add(13, 7, 9);
	jvals.add(14, 6, 1); jvals.add(14, 7, 4);
	jvals.add(15, 6, 4); jvals.add(15, 7, 6); jvals.add(15, 8, 4); jvals.add(15, 9, 1);
	spJ.resize(20, 10);
	*/
	spJ.setFromTriplets(jvals.begin(), jvals.end());
	spJ.makeCompressed();

	Logger::instance()->logMatrixCSV(spJ.toDense(), "spJ.csv");

	/*
	GeneralQRSolver slvr;
	slvr.compute(spJ);
	JacobianType slvrQ(spJ.rows(), spJ.rows());
	slvrQ.setIdentity();
	slvrQ = slvr.matrixQ() * slvrQ;
	Logger::instance()->logMatrixCSV(slvrQ.toDense(), "slvrQ.csv");
	Logger::instance()->logMatrixCSV(slvr.matrixQ2().toDense(), "slvrQ2.csv");
	Logger::instance()->logMatrixCSV(slvr.matrixR().toDense(), "slvrR.csv");

	*/

	BandedQRSolver slvr;
	slvr.setPruningEpsilon(1e-12);
	slvr.setBlockParams(4, 2);	
	clock_t begin = clock();
	slvr.compute(spJ);
	std::cout << "Slvr compute elapsed: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	begin = clock();
	JacobianType slvrQ(spJ.rows(), spJ.rows());
	slvrQ.setIdentity();
	slvrQ = slvr.matrixQ() * slvrQ;
	std::cout << "Slvr Q elapsed: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	Logger::instance()->logMatrixCSV(slvrQ.toDense(), "slvrQ.csv");
	Logger::instance()->logMatrixCSV(slvr.matrixR().toDense(), "slvrR.csv");

	return 0;

	// Perform banded QR decomposition
	begin = clock();
	Eigen::TripletArray<Scalar, typename JacobianType::Index> Qvals(spJ.rows() * 2);
	Eigen::TripletArray<Scalar, typename JacobianType::Index> Rvals(spJ.rows() * spJ.cols() / 2);

	MatrixType hCoeffs(spJ.rows(), 1);

	Eigen::HouseholderQR<MatrixType> houseqr;
	//Eigen::HouseholderQR<MatrixType> houseqr;
	Index blockRows = 128;
	Index blockCols = 64;
	Index numBlocks = spJ.cols() / blockCols;
	MatrixType Jtmp;
	MatrixType Ji = spJ.block(0, 0, blockRows, blockCols);
	MatrixType Ji2;
	MatrixType tmp;

	for (Index i = 0; i < numBlocks; i++) {
		// Where does the current block start
		Index bs = i * blockCols;

		// Solve dense block using Householder QR
		houseqr.compute(Ji);

		//Eigen::MatrixXd fq = houseqr.householderQ();
		/*std::cout << fq << std::endl;
		std::cout << " ------ " << std::endl;
		std::cout << houseqr.householderQ() * MatrixType::Identity(blockRows, blockCols) << std::endl;
		std::cout << " ---#--- " << std::endl;
		*/
	
		//std::cout << houseqr.householderQ().essentialVector(0) << std::endl;
		//std::cout << "----" << std::endl;
		//std::cout << houseqr.householderQ().essentialVector(1) << std::endl;
		//std::cout << "--#--" << std::endl;
	
		for(int bc = 0; bc < blockCols; bc++) {
			Qvals.add(bs + bc, bs + bc, 1.0);
			for (int r = 0; r <  houseqr.householderQ().essentialVector(bc).rows(); r++) {
				Qvals.add(bs + r + 1 + bc, bs + bc, houseqr.householderQ().essentialVector(bc)(r));
			}		
			hCoeffs(bs + bc) = houseqr.hCoeffs()(bc);
		}

		// Update R
		if (i == numBlocks - 1) {
			tmp = houseqr.householderQ().transpose() * Ji;// spJ.block(bs, bs, blockRows, blockCols).toDense();
			for (int br = 0; br < blockCols; br++) {
				for (int bc = 0; bc < blockCols; bc++) {
					Rvals.add(bs + br, bs + bc, tmp(br, bc));
				}
			}
			// Rvals.add(bs, bs + 2, tmp(0, 2)); Rvals.add(bs, bs + 3, tmp(0, 3));
			//Rvals.add(bs + 1, bs, tmp(1, 0)); Rvals.add(bs + 1, bs + 1, tmp(1, 1)); //Rvals.add(bs + 1, bs + 2, tmp(1, 2)); Rvals.add(bs + 1, bs + 3, tmp(1, 3));
		} else {
			Ji2 = MatrixType::Zero(blockRows, blockCols * 2);
			Ji2 << Ji, spJ.block(bs, bs + blockCols, blockRows, blockCols).toDense();
			//Ji2.leftCols(2) = Ji;
			//Ji2.rightCols(2) = spJ.block(bs, bs + blockCols, blockRows, blockCols).toDense();
			tmp = houseqr.householderQ().transpose() * Ji2;// spJ.block(bs, bs, blockRows, blockCols * 2).toDense();
			for (int br = 0; br < blockCols; br++) {
				for (int bc = 0; bc < blockCols * 2; bc++) {
					Rvals.add(bs + br, bs + bc, tmp(br, bc));
				}
			}

			// Update block rows
			blockRows += blockCols;

			// Update Ji
			Ji = spJ.block(bs + blockCols, bs + blockCols, blockRows, blockCols);
			Ji.block(0, 0, blockRows - blockCols * 2, blockCols) = tmp.block(blockCols, blockCols, blockRows - blockCols * 2, blockCols);
		}

	}

	JacobianType Q(spJ.rows(), spJ.cols());
	Q.setFromTriplets(Qvals.begin(), Qvals.end());
	Q.makeCompressed();
	Q.prune(Scalar(1e-8), 1e-8);

	// Create final matrix R
	JacobianType R(spJ.rows(), spJ.cols());
	R.setFromTriplets(Rvals.begin(), Rvals.end());// , [](const Scalar&, const Scalar &b) { return b; });
	R.makeCompressed();
	R.prune(Scalar(1e-8), 1e-8);

	std::cout << "Elapsed: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";

	begin = clock();

	typedef Eigen::SparseVector<Scalar> SparseVector;
	Index m = R.rows();
	Index n = R.cols();
	Index diagSize = (std::min)(m, n);
	SparseVector resColJ;
	const Scalar Zero = Scalar(0);
	Scalar tau = Scalar(0);
	/*
	JacobianType res = spJ;
	for (Index j = 0; j < res.cols() - 2; j++) {
		res.col(j) -= hCoeffs.at(j) * Q.col(j);
	}*/

	/*
	// Compute Q as product of Householder vectors
	JacobianType I(spJ.rows(), spJ.rows());
	I.setIdentity();
	JacobianType QQ(spJ.rows(), spJ.rows());
	QQ.setIdentity();
	for (Index j = 0; j < res.cols() - 2; j++) {
		//std::cout << (I - (hCoeffs.at(j) * Q.col(j) * Q.col(j).transpose())).toDense() << std::endl;
		//std::cout << "----" << std::endl;
		QQ = QQ * (I - (hCoeffs.at(j) * Q.col(j) * Q.col(j).transpose()));
		QQ.prune(Scalar(0));
		//I.coeffRef(j, j) = 1;
	}*/
	
	//JacobianType res = R;
	JacobianType res(Q.rows(), Q.rows());
	res.setIdentity();
	// Compute res = Q * other column by column
	for (Index j = 0; j < res.cols(); j++) {
		// Use temporary vector resColJ inside of the for loop - faster access
		resColJ = res.col(j).pruned(Scalar(1e-8), 1e-8);
		/*Index start = diagSize - 1 - j;
		Index end = start - 128;
		start += 128;
		start = (start > diagSize - 1) ? diagSize - 1 : start;
		end = (end < 0) ? 0 : end;*/
		for (Index k = diagSize - 1; k >= 0; k--) {
			tau = Q.col(k).dot(resColJ.pruned(Scalar(1e-8), 1e-8));
			if (tau == Zero)
				continue;
			tau = tau * hCoeffs(k);
			resColJ -= tau * Q.col(k);
		}
		// Write the result back to j-th column of res
		res.col(j) = resColJ.pruned(Scalar(1e-8), 1e-8);
	}
	/*
	
	
	JacobianType res = spJ;
	//JacobianType res(Q.rows(), Q.rows());
	//res.setIdentity();
	for (Index j = 0; j < res.cols(); j++) {
		// Use temporary vector resColJ inside of the for loop - faster access
		resColJ = res.col(j).pruned(Scalar(1e-8), 1e-8);
		for (Index k = 0; k < diagSize; k++) {
			// Need to instantiate this to tmp to avoid various runtime fails (error in insertInnerOuter, mysterious collapses to zero)
			tau = Q.col(k).dot(resColJ.pruned(Scalar(1e-8), 1e-8));
			if (tau == Zero)
				continue;
			tau = tau * hCoeffs(k);
			resColJ -= tau * Q.col(k);
		}
		// Write the result back to j-th column of res
		res.col(j) = resColJ.pruned(Scalar(1e-8), 1e-8);
	}
	*/

	std::cout << "Elapsed: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";

	Logger::instance()->logMatrixCSV(R.toDense(), "R.csv");
	Logger::instance()->logMatrixCSV(Q.toDense(), "Q.csv");
	//Logger::instance()->logMatrixCSV(QQ.toDense(), "QQ.csv");

	Logger::instance()->logMatrixCSV((res).toDense(), "QR.csv");
	//Logger::instance()->logMatrixCSV((Q.transpose() * spJ).toDense(), "QtJ.csv");
	
	Logger::instance()->log(Logger::Info, "QR Diagonalization Test DONE!");

	return 0;
}

// Override system assert so one can set a breakpoint in it rather than clicking "Retry" and "Break"
void __cdecl _wassert(_In_z_ wchar_t const* _Message, _In_z_ wchar_t const* _File, _In_ unsigned _Line)
{
	std::wcerr << _File << "(" << _Line << "): ASSERT FAILED [" << _Message << "]\n";

	abort();
}
