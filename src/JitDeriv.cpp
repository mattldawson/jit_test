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
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include <iostream>
#include <stdlib.h>

namespace jit_test {

JitDeriv::JitDeriv(ClassicDeriv classicDeriv)
    : myJIT{std::move(*llvm::orc::leJIT::Create())} {
}
void JitDeriv::Solve(const double *const state, double *const deriv) {}
llvm::Value *JitDeriv::DerivCodeGen() {
  static llvm::LLVMContext myContext;
  static llvm::IRBuilder<> builder(myContext);
  static std::unique_ptr<llvm::Module> myModule;
  static std::unique_ptr<llvm::legacy::FunctionPassManager> myFPM;

  // Create module for derivative code
  myModule = llvm::make_unique<llvm::Module>("deriv jit code", myContext);

  // Optimization settings //

  // Create pass manager for code optimization
  myFPM = llvm::make_unique<llvm::legacy::FunctionPassManager>(myModule.get());

  // Simple "peephole" optimizations and bit-twiddling options
  myFPM->add(llvm::createInstructionCombiningPass());
  // Reassociate expressions
  myFPM->add(llvm::createReassociatePass());
  // Eliminate Common SubExpressions
  myFPM->add(llvm::createGVNPass());
  // Simplify the control flow graph (deleting unreachable blocks, etc.)
  myFPM->add(llvm::createCFGSimplificationPass());

  myFPM->doInitialization();

  // Code generation //

  // Prototype
  std::vector<llvm::Type *> derivArgsV(1, llvm::Type::getDoubleTy(myContext));
  llvm::FunctionType *derivFunctionType = llvm::FunctionType::get(
      llvm::Type::getDoubleTy(myContext), derivArgsV, false);
  llvm::Function *derivFunction =
      llvm::Function::Create(derivFunctionType, llvm::Function::ExternalLinkage,
                             "deriv", myModule.get());
  for (auto &Arg : derivFunction->args())
    Arg.setName("state");

  // function body
  llvm::BasicBlock *BB =
      llvm::BasicBlock::Create(myContext, "entry", derivFunction);
  builder.SetInsertPoint(BB);
  llvm::Value *retVal = llvm::ConstantFP::get(myContext, llvm::APFloat(12.2));
  builder.CreateRet(retVal);
  std::fprintf(stderr, "Generated function definition:\n");
  derivFunction->print(llvm::errs());

  // Optimization //
  verifyFunction(*derivFunction);
  myFPM->run(*derivFunction);

  // input arguments
  std::vector<llvm::Value *> ArgsV;
  ArgsV.push_back(llvm::ConstantFP::get(myContext, llvm::APFloat(232.5)));
  llvm::Value *derivCall =
      builder.CreateCall(derivFunction, ArgsV, "call deriv");

  return derivCall;
}
} // namespace jit_test
