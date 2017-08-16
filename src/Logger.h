#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>

#include "eigen_extras.h"

class Logger {
public:
	enum Level {
		Info = 0,
		Warning = 1,
		Error = 2,
		Debug = 3
	};

	std::string levelToString(Level level) const {
		switch (level) {
		case Level::Info:
			return "INFO";
		case Level::Warning:
			return "WARNING";
		case Level::Error:
			return "ERROR";
		case Level::Debug:
			return "DEBUG";
		default:
			return "Unknown";
		}
	}
	
	Logger() {
	}
	Logger(const std::string &_fileName);
	~Logger();

	/**
		Logging related functions
	*/
	void log(Logger::Level level, const std::string &msg);

	template <typename T, int _Options, typename _Index>
	void logSparseMatrix(Eigen::SparseMatrix<T, _Options, _Index> const& J, char const* filename) {
		std::ofstream f(filename);
		if (!f.good()) {
			std::stringstream ss;
			ss << "Failed to open [" << filename << "] for writing";
			this->instance()->log(Logger::Error, ss.str());
			return;
		}

		std::stringstream ss;
		ss << "Writing " << J.rows() << "x" << J.cols() << " sparse to \"" << filename << "\"";
		this->instance()->log(Logger::Debug, ss.str());

		for (int k = 0; k < J.outerSize(); ++k)
			for (Eigen::SparseMatrix<T, _Options, _Index>::InnerIterator it(J, k); it; ++it)
				f << "(" << it.row() << ", " << it.col() << ") = " << it.value() << "; ";
	}

	template <typename Derived>
	void logMatrix(Eigen::MatrixBase<Derived> const& J, char const* filename) {
		std::ofstream f(filename);
		if (!f.good()) {
			std::stringstream ss;
			ss << "Failed to open [" << filename << "] for writing";
			this->instance()->log(Logger::Error, ss.str());
			return;
		}

		std::stringstream ss;
		std::cout << "Writing " << J.rows() << "x" << J.cols() << " dense to \"" << filename << "\"" << std::endl;
		this->instance()->log(Logger::Debug, ss.str());
		f << J;
	}

	void setLevel(Logger::Level level) {
		this->level = level;
	}
	
	// Singleton initializer
	static void createLogger(const std::string &_fileName);
	// Singleton getter
	static Logger* instance(); 

private:
	std::ofstream outFile;
	std::mutex mutex;

	Level level;

	std::string timestamp() const;

	// Singleton instance
	static Logger *s_instance;
};

#endif