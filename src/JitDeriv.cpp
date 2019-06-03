//===-- src/JitTest.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019 Matthew Dawson
// Licensed under the GNU General Public License version 2 or (at your
// option) any later version. See the file COPYING for details
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief JIT compilation derivative calculation class
///
//===----------------------------------------------------------------------===//

#include "JitDeriv.h"
#include "ClassicDeriv.h"

namespace jit_test {

JitDeriv::JitDeriv(ClassicDeriv classicDeriv)
    : myJIT{std::move(*llvm::orc::leJIT::Create())} {
}
void JitDeriv::Solve(const double *const state, double *const deriv) {}
llvm::Value *JitDeriv::DerivCodeGen() {
  static llvm::LLVMContext myContext;
  static std::unique_ptr<llvm::Module> module;
  static llvm::IRBuilder<> builder(myContext);
  return llvm::ConstantFP::get(myContext, llvm::APFloat(12.2));
}
} // namespace jit_test
