#include "Logger.h"
#include <ctime>

Logger *Logger::s_instance = NULL;

Logger::Logger(const std::string &_fileName) 
	: outFile(std::ofstream(_fileName)), level(Level::Info) {
}

Logger::~Logger() {
	outFile.close();
}

std::string Logger::timestamp() const {
	time_t rawtime;
	struct tm * timeinfo;
	char buffer[80];

	std::time(&rawtime);
	timeinfo = std::localtime(&rawtime);
	std::strftime(buffer, sizeof(buffer), "%d-%m-%Y %I:%M:%S", timeinfo);

	return std::string(buffer);
}

void Logger::log(Logger::Level level, const std::string &msg) {
	this->mutex.lock();
	if (this->level <= level) {
		this->outFile << this->levelToString(level).c_str() << " [" << this->timestamp().c_str() << "]: " << msg.c_str() << std::endl;
	}
	this->mutex.unlock();
}

void Logger::createLogger(const std::string &_fileName) {
	if (s_instance == NULL) {
		s_instance = new Logger(_fileName);
	}
}

// Singleton getter
Logger* Logger::instance() {
	if (s_instance == NULL) {
		s_instance = new Logger("output.log");
	}
	return s_instance;
}