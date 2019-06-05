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

static llvm::AllocaInst *CreateEntryBlockAlloca(llvm::Function *TheFunction,
                                                llvm::Type *type,
                                                const std::string &VarName) {
  llvm::IRBuilder<> TmpB(&TheFunction->getEntryBlock(),
                         TheFunction->getEntryBlock().begin());
  return TmpB.CreateAlloca(type, 0, VarName.c_str());
}

JitDeriv::JitDeriv(ClassicDeriv classicDeriv)
    : myJIT{std::move(*llvm::orc::leJIT::Create())} {
}
void JitDeriv::Solve(const double *const state, double *const deriv) {}
void JitDeriv::DerivCodeGen() {
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

  // Types
  llvm::Type *dbl = llvm::Type::getDoubleTy(myContext);
  llvm::Type *dblPtr = llvm::Type::getDoubleTy(myContext)->getPointerTo();

  // Code generation //

  // Prototype
  std::vector<llvm::Type *> derivArgsV{dblPtr, dblPtr};
  llvm::FunctionType *derivFunctionType =
      llvm::FunctionType::get(dbl, derivArgsV, false);
  llvm::Function *derivFunction =
      llvm::Function::Create(derivFunctionType, llvm::Function::ExternalLinkage,
                             "deriv", myModule.get());
  llvm::Function::arg_iterator argIter = derivFunction->arg_begin();
  llvm::Value *state = argIter++;
  state->setName("state");
  llvm::Value *deriv = argIter++;
  deriv->setName("deriv");

  // function body
  llvm::BasicBlock *BB =
      llvm::BasicBlock::Create(myContext, "entry", derivFunction);
  builder.SetInsertPoint(BB);
  llvm::AllocaInst *allocaState =
      CreateEntryBlockAlloca(derivFunction, dblPtr, "state");
  llvm::AllocaInst *allocaDeriv =
      CreateEntryBlockAlloca(derivFunction, dblPtr, "deriv");
  builder.CreateStore(state, allocaState);
  builder.CreateStore(deriv, allocaDeriv);
  // llvm::Value *stateVal = builder.CreateExtractValue(state, 3);
  llvm::Value *retVal = llvm::ConstantFP::get(myContext, llvm::APFloat(12.2));
  builder.CreateRet(retVal);

  // Print llvm code
  std::fprintf(stderr, "Generated function definition:\n");
  derivFunction->print(llvm::errs());
  std::fprintf(stderr, "\n");

  // Optimization //
  verifyFunction(*derivFunction);
  myFPM->run(*derivFunction);

}
} // namespace jit_test
