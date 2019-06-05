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

JitDeriv::JitDeriv() : myJIT{std::move(*llvm::orc::leJIT::Create())} {}
void JitDeriv::Solve(double *state, double *deriv) {
  std::printf("\nDeriv func = %le\n", this->funcPtr(state, deriv));
}
void JitDeriv::DerivCodeGen(ClassicDeriv cd) {
  static llvm::LLVMContext myContext;
  static llvm::IRBuilder<> builder(myContext);
  static std::unique_ptr<llvm::Module> myModule;
  static std::unique_ptr<llvm::legacy::FunctionPassManager> myFPM;

  // Create module for derivative code
  myModule = llvm::make_unique<llvm::Module>("deriv jit code", myContext);
  myModule->setDataLayout(myJIT->getDataLayout());

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
  llvm::Type *vd = llvm::Type::getVoidTy(myContext);

  // Code generation //

  // Prototype
  std::vector<llvm::Type *> derivArgsV{dblPtr, dblPtr};
  llvm::FunctionType *derivFunctionType =
      llvm::FunctionType::get(dbl, derivArgsV, false);
  llvm::Function *derivFunction =
      llvm::Function::Create(derivFunctionType, llvm::Function::ExternalLinkage,
                             "derivFunc", myModule.get());
  llvm::Function::arg_iterator argIter = derivFunction->arg_begin();
  llvm::Value *state = argIter++;
  state->setName("state");
  llvm::Value *deriv = argIter++;
  deriv->setName("deriv");

  // function body //

  // set up array arguments
  llvm::BasicBlock *BB =
      llvm::BasicBlock::Create(myContext, "entry", derivFunction);
  builder.SetInsertPoint(BB);
  llvm::AllocaInst *allocaState =
      CreateEntryBlockAlloca(derivFunction, dblPtr, "state");
  llvm::AllocaInst *allocaDeriv =
      CreateEntryBlockAlloca(derivFunction, dblPtr, "deriv");
  builder.CreateStore(state, allocaState);
  builder.CreateStore(deriv, allocaDeriv);
  llvm::Value *statePtr = builder.CreateLoad(allocaState);
  llvm::Value *derivPtr = builder.CreateLoad(allocaDeriv);

  // derivative calculation variables
  llvm::Value *idxList[1]; // array index
  llvm::Value *stateElem;  // state array element
  llvm::Value *derivElem;  // derivative array element
  llvm::Value *tmpVal;     // temporary value from array
  llvm::Value *rate;       // working derivative

  int derivCont[NUM_SPEC] = {
      0}; // number of contributions to each species derivative

  // calculate derivative contributions from each reaction
  for (int i_rxn = 0; i_rxn < cd.numRxns; ++i_rxn) {
    rate = llvm::ConstantFP::get(myContext, llvm::APFloat(cd.rateConst[i_rxn]));
    for (int i_react = 0; i_react < cd.numReact[i_rxn]; ++i_react) {
      idxList[0] = llvm::ConstantInt::get(
          myContext, llvm::APInt(64, cd.reactId[i_rxn][i_react]));
      stateElem = builder.CreateGEP(statePtr, idxList, "stateElemPtr");
      tmpVal = builder.CreateLoad(stateElem, "stateElemVal");
      rate = builder.CreateFMul(rate, tmpVal, "multRate");
    }
    for (int i_react = 0; i_react < cd.numReact[i_rxn]; ++i_react) {
      int i_spec = cd.reactId[i_rxn][i_react];
      idxList[0] = llvm::ConstantInt::get(myContext, llvm::APInt(64, i_spec));
      derivElem = builder.CreateGEP(derivPtr, idxList, "derivElemPtr");
      if (derivCont[i_spec] > 0) {
        tmpVal = builder.CreateLoad(derivElem, "existingDerivVal");
      } else {
        tmpVal = llvm::ConstantFP::get(myContext, llvm::APFloat(0.0));
      }
      tmpVal = builder.CreateFSub(tmpVal, rate, "subRateReact");
      builder.CreateStore(tmpVal, derivElem);
      ++derivCont[i_spec];
    }
    for (int i_prod = 0; i_prod < cd.numProd[i_rxn]; ++i_prod) {
      int i_spec = cd.prodId[i_rxn][i_prod];
      idxList[0] = llvm::ConstantInt::get(myContext, llvm::APInt(64, i_spec));
      derivElem = builder.CreateGEP(derivPtr, idxList, "derivElemPtr");
      if (derivCont[i_spec] > 0) {
        tmpVal = builder.CreateLoad(derivElem, "existingDerivVal");
        tmpVal = builder.CreateFAdd(tmpVal, rate, "addRateProd");
        builder.CreateStore(tmpVal, derivElem);
      } else {
        builder.CreateStore(rate, derivElem);
      }
      ++derivCont[i_spec];
    }
  }
  builder.CreateRet(rate);

  // Print llvm code
  std::fprintf(stderr, "Generated function definition:\n");
  derivFunction->print(llvm::errs());
  std::fprintf(stderr, "\n");

  // Optimization //
  verifyFunction(*derivFunction);
  myFPM->run(*derivFunction);

  // Add the module to the JIT
  auto H = myJIT->addModule(std::move(myModule));

  // Find the function
  llvm::JITEvaluatedSymbol exprSymbol = myJIT->lookup("derivFunc").get();

  // Get a pointer to the function
  this->funcPtr =
      (double (*)(double *, double *))(intptr_t)exprSymbol.getAddress();
}
} // namespace jit_test
