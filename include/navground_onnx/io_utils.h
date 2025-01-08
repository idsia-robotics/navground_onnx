/**
 * @author Jerome Guzzi - <jerome@idsia.ch>
 */

#ifndef NAVGROUND_ONNX_IO_UTILS_H_
#define NAVGROUND_ONNX_IO_UTILS_H_

#include <stdio.h>
#include <unistd.h>

class SuppressStdErr {
public:
  SuppressStdErr() {
    fflush(stderr);
    _fd = dup(STDERR_FILENO);
    freopen("/dev/null", "w", stderr);
  }

  ~SuppressStdErr() {
    fflush(stderr);
    dup2(_fd, fileno(stderr));
    close(_fd);
  }

private:
  int _fd;
};

#endif // NAVGROUND_ONNX_IO_UTILS_H_
