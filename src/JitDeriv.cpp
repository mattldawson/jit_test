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

JitDeriv::JitDeriv(llvm::orc::JITTargetMachineBuilder JTMB, llvm::DataLayout DL,
                   ClassicDeriv classicDeriv)
    : ObjectLayer(
          ES, []() { return llvm::make_unique<llvm::SectionMemoryManager>(); }),
      CompileLayer(ES, ObjectLayer,
                   llvm::orc::ConcurrentIRCompiler(std::move(JTMB))),
      DL(std::move(DL)), Mangle(ES, this->DL),
      Ctx(llvm::make_unique<llvm::LLVMContext>()) {
  ES.getMainJITDylib().setGenerator(cantFail(
      llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(DL)));
}
void JitDeriv::Solve(const double *const state, double *const deriv) {}
} // namespace jit_test
