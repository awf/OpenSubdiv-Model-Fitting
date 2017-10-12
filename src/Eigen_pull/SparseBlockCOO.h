// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef SPARSE_BLOCK_COO_H
#define SPARSE_BLOCK_COO_H

#include <Eigen/Eigen>

/*
* Each block YTY is devided into diagonal part and subdiagonal part, which is "running away" from the main diagonal
* as we go along the columns of the matrix.
*/
// Take the full column vector and compose dense vector from the main diagonal and subdiagonal part of the block, skipping zero elements in the middle
#define FULL_TO_BLOCK_VEC(fullVec, blockVec, yRows, yCols, blockRowIdx, blockNumZeros, numSubdiagElems) \
	blockVec = VectorXd(yRows); \
	blockVec.segment(0, yCols) = fullVec.segment(blockRowIdx, yCols); \
	if(numSubdiagElems > 0) { \
		blockVec.segment(yCols, numSubdiagElems) = fullVec.segment(blockRowIdx + yCols + blockNumZeros, numSubdiagElems); \
	} 

// Take the dense vector composed from the main diagonal and subdiagonal part of the block, skipping zero elements in the middle, and fill it back into the full column vector
#define BLOCK_VEC_TO_FULL(fullVec, blockVec, yRows, yCols, blockRowIdx, blockNumZeros, numSubdiagElems) \
	fullVec.segment(blockRowIdx, yCols) = blockVec.segment(0, yCols); \
	fullVec.segment(blockRowIdx + yCols + blockNumZeros, numSubdiagElems) = blockVec.segment(yCols, numSubdiagElems);

/*
* A dense block of the compressed WY representation (YTY) of the Householder product.
* Stores matrices Y (m x n) and T (n x n) and number of zeros between main diagonal and subdiagonal parts of the block YTY.
* Provides overloaded multiplication operator (*) allowing to easily perform the multiplication with a dense vector (Y * (T * (Y' * v)))
*/
template <typename ScalarType, typename IndexType>
class BlockYTY {
	typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> MatrixType;
	typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> VectorType;
public:
	BlockYTY() {
	}

	BlockYTY(const MatrixType &Y, const MatrixType &T, const IndexType numZeros)
		: matY(Y), matT(T), nzrs(numZeros) {
	}

	MatrixType& Y() {
		return this->matY;
	}

	MatrixType& T() {
		return this->matT;
	}

	IndexType rows() const {
		return this->matY.rows();
	}
	IndexType cols() const {
		return this->matY.cols();
	}

	IndexType numZeros() const {
		return this->nzrs;
	}

	BlockYTY transpose() const {
		return BlockYTY(matY, matT.transpose(), nzrs);
	}

	// FixMe: Is there a better way? (This is faster than calling .transpose() *
	VectorType multTransposed(const VectorType &other) const {
		return (this->matY * (this->matT.transpose() * (this->matY.transpose() * other)));
	}

	VectorType operator*(const VectorType &other) const {
		return (this->matY * (this->matT * (this->matY.transpose() * other)));
	}

private:
	MatrixType matY;
	MatrixType matT;

	IndexType nzrs;
};

/*
* Storage type for general sparse matrix with block structure.
* Each element holds block position (row index, column index) and the values in the block stored in ValueType.
* ValueType is a template type and can generally represent any datatype, both default and user defined.
*/
template <typename ValueType, typename IndexType>
class SparseBlockCOO {
public:
	struct Element {
		IndexType row;
		IndexType col;

		ValueType value;

		Element()
			: row(0), col(0) {
		}

		Element(const IndexType row, const IndexType col, const ValueType &val)
			: row(row), col(col), value(val) {
		}
	};
	typedef std::vector<Element> ElementsVec;

	SparseBlockCOO()
		: nRows(0), nCols(0) {
	}

	SparseBlockCOO(const IndexType &rows, const IndexType &cols)
		: nRows(rows), nCols(cols) {
	}

	void insert(const Element &elem) {
		this->elems.push_back(elem);
	}

	IndexType size() const {
		return this->elems.size();
	}

	void clear() {
		this->elems.clear();
	}

	Element& operator[](IndexType i) {
		return this->elems[i];
	}
	const Element& operator[](IndexType i) const {
		return this->elems[i];
	}

private:
	ElementsVec elems;

	IndexType nRows;
	IndexType nCols;
};


#endif

