#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
typedef unsigned int uint32;
typedef unsigned long long uint64;

enum BYTES {
  _4BYTE = 32,
  _8BYTE = 64,
};

enum TYPE {
  UINT32 = 0,
  UINT64 = 1,
  DOUBLE = 2,
};

void usage(const char *program_name) {
  std::cout << "Usage: " << program_name
            << " <inputFile> <numVertices> <n_ignored_lines> [-d] [-w] [-p]"
            << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  --input : Specify the graph input " << std::endl;
  std::cout << "  --algo : Specify the algorithm {'bfs', 'cc', 'pr', 'sssp'} "
               "(default = 'bfs')"
            << std::endl;
  std::cout << "  --type : Specify the edge size {32, 64} (default = 32)"
            << std::endl;
  std::cout
      << "  --type : Specify the source vertex {0, 1, 2, ...} (default = 1)"
      << std::endl;
  std::cout << "  --runs : Specify the heuristic for memory usage [0.0f - 1.0] "
               "(default = 0.5f)"
            << std::endl;
  std::cout
      << "  --runs : Specify the number of runs {0, 1, 2, ...} (default = 1)"
      << std::endl;
  exit(0);
}

void compareValuesUINT32(uint64 numVertices, std::string filePath1,
                         std::string filePath2) {
  std::cout << "Comparing UINT32" << std::endl;
  std::vector<uint32> values1(numVertices);
  std::vector<uint32> values2(numVertices);

  std::ifstream file1(filePath1, std::ios::binary);
  if (!file1.is_open()) {

    std::cout << "Could not open file: " << filePath1 << std::endl;
    return;
  }

  file1.read(reinterpret_cast<char *>(values1.data()),
             numVertices * sizeof(uint64));

  file1.close();

  std::ifstream file2(filePath2, std::ios::binary);
  if (!file2.is_open()) {
    std::cout << "Could not open file: " << filePath2 << std::endl;
    return;
  }

  file2.read(reinterpret_cast<char *>(values2.data()),
             numVertices * sizeof(uint32));

  file2.close();

  int nErrors = 0;
  for (uint32 i = 0; i < numVertices; i++) {
    if (values1.at(i) != values2.at(i)) {
      std::cout << "Arrays do not match at: " << i << std::endl;
      std::cout << values1.at(i) << " | " << values2.at(i) << std::endl;
      nErrors++;
      if (nErrors > 20)
        break;
    }
  }
}

void compareValuesUINT64(uint64 numVertices, std::string filePath1,
                         std::string filePath2) {

  std::cout << "Comparing UINT64" << std::endl;
  std::vector<uint64> values1(numVertices);
  std::vector<uint64> values2(numVertices);

  std::ifstream file1(filePath1, std::ios::binary);
  if (!file1.is_open()) {

    std::cout << "Could not open file: " << filePath1 << std::endl;
    return;
  }

  file1.read(reinterpret_cast<char *>(values1.data()),
             numVertices * sizeof(uint64));

  file1.close();

  std::ifstream file2(filePath2, std::ios::binary);
  if (!file2.is_open()) {
    std::cout << "Could not open file: " << filePath2 << std::endl;
    return;
  }

  file2.read(reinterpret_cast<char *>(values2.data()),
             numVertices * sizeof(uint64));

  file2.close();

  int nErrors = 0;
  for (uint32 i = 0; i < numVertices; i++) {
    if (values1[i] != values2[i]) {
      std::cout << "Arrays do not match at: " << i << std::endl;
      std::cout << values1[i] << " | " << values2[i] << std::endl;
      nErrors++;
      if (nErrors > 20)
        break;
    }
  }
}

void compareValuesDOUBLES(uint64 numVertices, std::string filePath1,
                          std::string filePath2) {

  std::cout << "Comparing DOUBLES" << std::endl;
  std::vector<double> values1(numVertices);
  std::vector<double> values2(numVertices);

  std::ifstream file1(filePath1, std::ios::binary);
  if (!file1.is_open())
    return;

  file1.read(reinterpret_cast<char *>(values1.data()),
             numVertices * sizeof(double));

  file1.close();

  std::ifstream file2(filePath2, std::ios::binary);
  if (!file2.is_open())
    return;

  file2.read(reinterpret_cast<char *>(values2.data()),
             numVertices * sizeof(double));

  file2.close();

  uint32 nErrors = 0;
  double tolerance = 1e-3; // Adjust as needed
  for (uint32 i = 0; i < numVertices; i++) {
    double diff = std::abs(values1[i] - values2[i]);
    if (diff > tolerance) {
      std::cout << "Arrays do not match at: " << i << std::endl;
      std::cout << values1[i] << " | " << values2[i] << std::endl;
      nErrors++;
      if (nErrors > 20)
        break;
    }
  }
}

int main(int argc, char **argv) {
  std::string filePath1;
  std::string filePath2;
  uint64 numVertices = 0;
  uint32 type = 0;

  std::cout << "Number of arguments: " << argc << std::endl;
  for (int i = 0; i < argc; ++i) {
    std::cout << "Argument " << i << ": " << argv[i] << std::endl;
  }

  try {
    for (unsigned int i = 1; i < argc - 1; i = i + 2) {
      if (strcmp(argv[i], "--input1") == 0)
        filePath1 = std::string(argv[i + 1]);
      else if (strcmp(argv[i], "--input2") == 0)
        filePath2 = std::string(argv[i + 1]);
      else if (strcmp(argv[i], "--type") == 0)
        type = atoi(argv[i + 1]);
      else if (strcmp(argv[i], "--vertices") == 0) {
        numVertices = std::stoul(argv[i + 1]);
        std::cout << "Uen" << std::endl;
      }
    }
  } catch (...) {
    std::cerr << "An exception has occurred.\n";
    exit(0);
  }

  std::cout << "Number of Vertices: " << numVertices << std::endl;
  std::cout << "Values1 | Values2 " << std::endl;

  if (type == UINT32)
    compareValuesUINT32(numVertices, filePath1, filePath2);
  else if (type == UINT64)
    compareValuesUINT64(numVertices, filePath1, filePath2);
  else if (type == DOUBLE)
    compareValuesDOUBLES(numVertices, filePath1, filePath2);

  return 1;
}
