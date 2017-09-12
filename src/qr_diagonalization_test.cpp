#define _wassert wassert_awf
#include <cassert>

#include <iostream>
#include <iomanip>
#include <ctime>

#include <random>

#include <Eigen/Eigen>
#include <Eigen/SparseQR>

//#include <Eigen/SPQRSupport>
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
	
	Eigen::Index numVars = 1000;
	Eigen::Index numParams = numVars * 2;
	Eigen::Index numResiduals = numVars * 3 + numVars;
	Eigen::Index numRightParams = 105;

	typedef SparseMatrix<Scalar, ColMajor, int> JacobianType;
	typedef SparseMatrix<Scalar, ColMajor, SuiteSparse_long> JacobianTypeSPQR;
	typedef SparseQR_Ext<JacobianType, COLAMDOrdering<int> > GeneralQRSolver;
	//typedef SparseQR<JacobianType, NaturalOrdering<Eigen::Index> > GeneralQRSolver;
	typedef ColPivHouseholderQR<Matrix<Scalar, 3, 2> > DenseQRSolverSmallBlock;
	typedef SPQR<JacobianType> SPQRSolver;
	typedef GeneralQRSolver SparseSuperblockSolver;
	typedef BlockDiagonalSparseQR_Ext<JacobianType, DenseQRSolverSmallBlock> DiagonalSubblockSolver;
	typedef SparseSubblockQR_Ext<JacobianType, DiagonalSubblockSolver, SparseSuperblockSolver> SpecializedSparseSolver;

	// The right block would be assumed dense
	JacobianType rightBlock;
	Eigen::TripletArray<Scalar, typename JacobianType::Index> rbvals(numRightParams * numResiduals);
	for (int i = 0; i < numResiduals; i++) {
		for (int j = 0; j < numRightParams; j++) {
			rbvals.add(i, j, dist(gen));
		}
	}
	rightBlock.resize(numResiduals, numRightParams);
	rightBlock.setFromTriplets(rbvals.begin(), rbvals.end());
	rightBlock.makeCompressed();

	JacobianType spJ;
	Eigen::TripletArray<Scalar, typename JacobianType::Index> jvals(64);
/*
	for (int i = 0; i < numResiduals; i++) {


		for (int j = i * 2; j < i * 2 + 4 && j < numParams; j++) {
			jvals.add(i, j, dist(gen));
		}
	}*/
	//for (int i = 0; i < numVars; i++) {
	for (int i = 0; i < numParams; i++) {
		
			for (int j = numParams - 1; j >= i; j--) {
				//jvals.add(i + numVars * 0, j, dist(gen));
				//jvals.add(i * 2, j, dist(gen));
				//jvals.add(i * 2 + 1, j, dist(gen));
				//jvals.add(i + numParams * 2, j, dist(gen));
			}

			for (int j = i * 2; j < (i * 2) + 2 && j < numParams; j++) {
				jvals.add(i * 4, j, dist(gen));
				jvals.add(i * 4 + 1, j, dist(gen));
				jvals.add(i * 4 + 2, j, dist(gen));
				/**/
				///*
				jvals.add(i * 4 + 3, j, dist(gen));
				if (j < numParams - 2) {
				jvals.add(i * 4 + 3, j + 2, dist(gen));
				}
				//*/

				/*
				jvals.add(i + numVars * 3, j, dist(gen));
				if (j < numParams - 2) {
					jvals.add(i + numVars * 3, j + 2, dist(gen));
				}
				*/
			}

		}
	
	/*for (int i = numResiduals - numVars; i < numResiduals; i++) {
		

		for (int j = (i - (numResiduals - numVars)) * 2; j < (i - (numResiduals - numVars)) * 2 + 4 && j < numParams; j++) {
			jvals.add(i, j, dist(gen));
		}
	}*/

	spJ.resize(numResiduals, numParams);
	spJ.setFromTriplets(jvals.begin(), jvals.end());
	spJ.makeCompressed();

	clock_t begin;
//	Logger::instance()->logMatrixCSV(spJ.toDense(), "spJ.csv");

	//GeneralQRSolver solver;
	/*
	SpecializedSparseSolver solver;
	solver.setDiagBlockParams(numVars * 3, numVars * 2);
	solver.getDiagSolver().setSparseBlockParams(3, 2);
	begin = clock();
	solver.compute(spJ);

	std::cout << "Compute elapsed: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	JacobianType R;
	begin = clock();

	JacobianType Q;
	//Q2 = solver.matrixQ2();// .transpose();
	//JacobianType Q(spJ.rows(), spJ.rows());
	//Q.setIdentity();
	Q = solver.matrixQ();
	//R = solver.matrixR();
	//Q = (solver.matrixQ().transpose() * spJ);
	
	std::cout << "MatrixQ elapsed: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	std::cout << "MatrixQ nnz: " << Q.nonZeros() << std::endl;

	// Compute errors in reconstruction of Jacobian and R and output matrices
	R = solver.matrixR();
	solver.colsPermutation().applyThisOnTheRight(R);
	std::cout << "J err norm: " << ((Q * R).toDense() - spJ.toDense()).norm() << std::endl;

	R.resize(spJ.rows(), spJ.rows());
	R.setIdentity();
	R = R * solver.matrixR();
	solver.colsPermutation().applyThisOnTheRight(spJ);
	std::cout << "R err norm: " << ((Q.transpose() * spJ).toDense() - R.toDense()).norm() << std::endl;
	*/
	//Logger::instance()->logMatrixCSV((Q.transpose() * spJ).toDense(), "QtJ.csv");
	//Logger::instance()->logMatrixCSV((Q * R).toDense(), "QR.csv");
//	Logger::instance()->logMatrixCSV(Q.toDense(), "Q.csv");
	//Logger::instance()->logMatrixCSV(Q.cwiseAbs().toDense(), "Qabs.csv");
//	Logger::instance()->logMatrixCSV(R.toDense(), "R.csv");


	// Evaluate SPQR solver
	JacobianTypeSPQR spJSP;
	spJSP.resize(numResiduals, numParams);
	spJSP.setFromTriplets(jvals.begin(), jvals.end());
	//spJSP.makeCompressed();

//	Logger::instance()->logMatrixCSV(spJSP.toDense(), "spJSP.csv");

	SPQRSolver spqr;
	begin = clock();
	spqr.compute(spJSP);
	std::cout << "SPQR Compute elapsed: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";

	begin = clock();
	SPQRSolver::MatrixType QSP(spJSP.rows(), spJSP.rows());
	QSP.setIdentity();
	//Eigen::MatrixXd q(spJSP.rows(), spJSP.rows());
	//q.setIdentity();
	//Eigen::MatrixXd res;
	QSP = spqr.matrixQ() * QSP;
	QSP.prune(Scalar(0));
	std::cout << "SPQR MatrixQ elapsed: " << double(clock() - begin) / CLOCKS_PER_SEC << "s\n";
	
	JacobianTypeSPQR RSP(spJSP.rows(), spJSP.cols());
	RSP = spqr.matrixR();
	spqr.colsPermutation().applyThisOnTheRight(RSP);
	std::cout << "SPQR J err norm: " << ((QSP * RSP).toDense() - spJSP.toDense()).norm() << std::endl;
	
	RSP.resize(spJSP.rows(), spJSP.rows());
	RSP.setIdentity();
	RSP = RSP * spqr.matrixR();
	spqr.colsPermutation().applyThisOnTheRight(spJSP);
	std::cout << "SPQR R err norm: " << ((QSP.transpose() * spJSP).toDense() - RSP.toDense()).norm() << std::endl;

//	Logger::instance()->logMatrixCSV(QSP.toDense(), "Qs.csv");
//	Logger::instance()->logMatrixCSV(RSP.toDense(), "Rs.csv");
//	Logger::instance()->logMatrixCSV((QSP.transpose() * spJSP).toDense(), "QJRs.csv");

	return 0;

	/*
	// Create testing matrix
	MatrixXX J = MatrixXX::Zero(6, 4);
	J.row(0) << 3, 1, 0, 0;
	J.row(1) << 0, 1, 0, 0;
	J.row(2) << 4, 3, 2, 2;
	J.row(3) << 0, 0, 5, 7;
	J.row(4) << 0, 0, 8, 6;
	J.row(5) << 0, 0, 9, 1;
	*/
	// Use "inverse" Givens rotations to zero out the last two elements of row(2) -> it is corrupting the diagonal
	/*
	 * | c  -s | * | x_i | = |           0           |
	 * | s   c |   | x_j |   | (x_i^2 + x_j^2)^(1/2) |
	 *
	 * Solution can be found as:
	 * c = x_j / (x_i^2 + x_j^2)^(1/2)
	 * s = x_i / (x_i^2 + x_j^2)^(1/2)
	 *
	 */
	/*
	MatrixXX R = J;

	std::cout << "R = \n" << R << std::endl;

	Scalar c = R(1, 0) / sqrt(R(1, 0) * R(1, 0) + R(2, 0) * R(2, 0));
	Scalar s = R(2, 0) / sqrt(R(1, 0) * R(1, 0) + R(2, 0) * R(2, 0));
	MatrixXX G22 = MatrixXX::Identity(6, 6);
	G22(1, 1) = c; G22(1, 2) = -s;
	G22(2, 1) = s; G22(2, 2) = c;
	R = G22 * R;

	std::cout << "G22 = \n" << G22 << std::endl;
	std::cout << "R = \n" << R << std::endl;

	Scalar c2 = R(3, 3) / sqrt(R(2, 3) * R(2, 3) + R(3, 3) * R(3, 3));
	Scalar s2 = R(2, 3) / sqrt(R(2, 3) * R(2, 3) + R(3, 3) * R(3, 3));
	MatrixXX G23 = MatrixXX::Identity(6, 6);
	G23(2, 2) = c2; G23(2, 3) = -s2;
	G23(3, 2) = s2; G23(3, 3) = c2;
	R = G23 * R;

	std::cout << "G23 = \n" << G23 << std::endl;
	std::cout << "R = \n" << R << std::endl;

	MatrixXX Q = (G22.transpose() * G23.transpose());
	 
	std::cout << "Q = \n" << Q << std::endl;
	std::cout << "R = \n" << R << std::endl;
	*/
	Logger::instance()->log(Logger::Info, "QR Diagonalization Test DONE!");

	return 0;
}

// Override system assert so one can set a breakpoint in it rather than clicking "Retry" and "Break"
void __cdecl _wassert(_In_z_ wchar_t const* _Message, _In_z_ wchar_t const* _File, _In_ unsigned _Line)
{
	std::wcerr << _File << "(" << _Line << "): ASSERT FAILED [" << _Message << "]\n";

	abort();
}
