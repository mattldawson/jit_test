#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include "jit_test/Basic/Version.h"

int main(int argc, const char **argv) {
  llvm::InitLLVM X(argc, argv);
  llvm::outs() << "Hello, I am Jit Test " << jit_test::getJitTestVersion()
               << "\n";
}
