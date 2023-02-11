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
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include <iostream>
#include <memory>
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

void JitDeriv::Solve(double *rateConst, double *state, double *deriv) {
  this->funcPtr(rateConst, state, deriv);
}

void JitDeriv::DerivCodeGen(ClassicDeriv cd) {
  static std::unique_ptr<llvm::LLVMContext> myContext;
  static std::unique_ptr<llvm::IRBuilder<>> builder;
  static std::unique_ptr<llvm::Module> myModule;
  static llvm::ExitOnError ExitOnErr;

  // Open a new context and module
  myContext = std::make_unique<llvm::LLVMContext>();
  myModule = std::make_unique<llvm::Module>("deriv jit code", *myContext);
  myModule->setDataLayout(myJIT->getDataLayout());

  // Create a new builder for the module
  builder = std::make_unique<llvm::IRBuilder<>>(*myContext);

  // Types
  llvm::Type *intTy = llvm::Type::getInt64Ty(*myContext);
  llvm::Type *dbl = llvm::Type::getDoubleTy(*myContext);
  llvm::Type *dblPtr = llvm::Type::getDoubleTy(*myContext)->getPointerTo();
  llvm::Type *vd = llvm::Type::getVoidTy(*myContext);

  // Code generation //

  // Prototype
  std::vector<llvm::Type *> derivArgsV{dblPtr, dblPtr, dblPtr};
  llvm::FunctionType *derivFunctionType =
      llvm::FunctionType::get(vd, derivArgsV, false);
  llvm::Function *derivFunction =
      llvm::Function::Create(derivFunctionType, llvm::Function::ExternalLinkage,
                             "derivFunc", myModule.get());
  llvm::Function::arg_iterator argIter = derivFunction->arg_begin();
  llvm::Value *rateConst = argIter++;
  rateConst->setName("rateConst");
  llvm::Value *state = argIter++;
  state->setName("state");
  llvm::Value *deriv = argIter++;
  deriv->setName("deriv");

  // function body //

  // set up entry block
  llvm::BasicBlock *entryBB = llvm::BasicBlock::Create(*myContext, "entry", derivFunction);
  builder->SetInsertPoint(entryBB);

  // set up array arguments
  llvm::AllocaInst *allocaRateConst =
      CreateEntryBlockAlloca(derivFunction, dblPtr, "rateConst");
  llvm::AllocaInst *allocaState =
      CreateEntryBlockAlloca(derivFunction, dblPtr, "state");
  llvm::AllocaInst *allocaDeriv =
      CreateEntryBlockAlloca(derivFunction, dblPtr, "deriv");
  builder->CreateStore(rateConst, allocaRateConst);
  builder->CreateStore(state, allocaState);
  builder->CreateStore(deriv, allocaDeriv);
  llvm::Value *rateConstPtr = builder->CreateLoad(dblPtr, allocaRateConst);
  llvm::Value *statePtr = builder->CreateLoad(dblPtr, allocaState);
  llvm::Value *derivPtr = builder->CreateLoad(dblPtr, allocaDeriv);

  // set up loop block
  llvm::BasicBlock *loopBB =
      llvm::BasicBlock::Create(*myContext, "loop", derivFunction);
  builder->CreateBr(loopBB);
  builder->SetInsertPoint(loopBB);
  llvm::PHINode *i_cell = builder->CreatePHI(intTy, 2, "i_cell");
  i_cell->addIncoming(llvm::ConstantInt::get(*myContext, llvm::APInt(64,0)),
                      entryBB);

  // derivative calculation variables
  llvm::Value *numRxns;        // number of reactions
  llvm::Value *numSpec;        // number of species
  llvm::Value *cellRxnOffset;  // offset for current cell in rateConst
  llvm::Value *cellSpecOffset; // offset for current cell in state/deriv
  llvm::Value *idxList[1];     // array index
  llvm::Value *rateConstElem;  // rate constant element
  llvm::Value *stateElem;      // state array element
  llvm::Value *derivElem;      // derivative array element
  llvm::Value *tmpVal;         // temporary value from array
  llvm::Value *rate;           // working derivative

  int derivCont[NUM_SPEC] = {
      0}; // number of contributions to each species derivative

  // calculate derivative contributions from each reaction
  numRxns = llvm::ConstantInt::get(*myContext, llvm::APInt(64, cd.numRxns));
  numSpec = llvm::ConstantInt::get(*myContext, llvm::APInt(64, cd.numSpec));
  cellRxnOffset = builder->CreateNSWMul(numRxns, i_cell, "cellRxnOffset");
  cellSpecOffset = builder->CreateNSWMul(numSpec, i_cell, "cellSpecOffset");
  for (int i_rxn = 0; i_rxn < cd.numRxns; ++i_rxn) {
    tmpVal = llvm::ConstantInt::get(*myContext, llvm::APInt(64, i_rxn));
    idxList[0] = builder->CreateNSWAdd(cellRxnOffset, tmpVal, "rxnIndex");
    rateConstElem = builder->CreateGEP(dbl, rateConstPtr, idxList, "rateConstPtr");
    rate = builder->CreateLoad(dbl, rateConstElem, "rateConstElemVal");
    for (int i_react = 0; i_react < cd.numReact[i_rxn]; ++i_react) {
      int i_spec = cd.reactId[i_rxn][i_react];
      tmpVal = llvm::ConstantInt::get(*myContext, llvm::APInt(64, i_spec));
      idxList[0] = builder->CreateNSWAdd(cellSpecOffset, tmpVal, "stateIndex");
      stateElem = builder->CreateGEP(dbl, statePtr, idxList, "stateElemPtr");
      tmpVal = builder->CreateLoad(dbl, stateElem, "stateElemVal");
      rate = builder->CreateFMul(rate, tmpVal, "multRate");
    }
    for (int i_react = 0; i_react < cd.numReact[i_rxn]; ++i_react) {
      int i_spec = cd.reactId[i_rxn][i_react];
      tmpVal = llvm::ConstantInt::get(*myContext, llvm::APInt(64, i_spec));
      idxList[0] = builder->CreateNSWAdd(cellSpecOffset, tmpVal, "stateIndex");
      derivElem = builder->CreateGEP(dbl, derivPtr, idxList, "derivElemPtr");
      if (derivCont[i_spec] > 0) {
        tmpVal = builder->CreateLoad(dbl, derivElem, "existingDerivVal");
      } else {
        tmpVal = llvm::ConstantFP::get(*myContext, llvm::APFloat(0.0));
      }
      tmpVal = builder->CreateFSub(tmpVal, rate, "subRateReact");
      builder->CreateStore(tmpVal, derivElem);
      ++derivCont[i_spec];
    }
    for (int i_prod = 0; i_prod < cd.numProd[i_rxn]; ++i_prod) {
      int i_spec = cd.prodId[i_rxn][i_prod];
      tmpVal = llvm::ConstantInt::get(*myContext, llvm::APInt(64, i_spec));
      idxList[0] = builder->CreateNSWAdd(cellSpecOffset, tmpVal, "stateIndex");
      derivElem = builder->CreateGEP(dbl, derivPtr, idxList, "derivElemPtr");
      if (derivCont[i_spec] > 0) {
        tmpVal = builder->CreateLoad(dbl, derivElem, "existingDerivVal");
        tmpVal = builder->CreateFAdd(tmpVal, rate, "addRateProd");
        builder->CreateStore(tmpVal, derivElem);
      } else {
        builder->CreateStore(rate, derivElem);
      }
      ++derivCont[i_spec];
    }
  }

  // Increment cell counter and check for end of loop
  llvm::Value *step = llvm::ConstantInt::get(*myContext, llvm::APInt(64, 1));
  llvm::Value *nextCell = builder->CreateNSWAdd(i_cell, step, "nextCell");
  llvm::Value *numCells = llvm::ConstantInt::get(*myContext, llvm::APInt(64,cd.numCell));
  llvm::Value *atEnd = builder->CreateICmpSGE(nextCell, numCells, "atEnd");

  // Branch to next interation or exit loop
  llvm::BasicBlock *loopEndBB = builder->GetInsertBlock();
  llvm::BasicBlock *afterBB = llvm::BasicBlock::Create(*myContext, "afterLoop", derivFunction);
  builder->CreateCondBr(atEnd, afterBB, loopBB);
  builder->SetInsertPoint(afterBB);
  i_cell->addIncoming(nextCell, loopEndBB);

  builder->CreateRetVoid();

#if 0
  // Print llvm code
  std::fprintf(stderr, "Generated function definition:\n");
  derivFunction->print(llvm::errs());
  std::fprintf(stderr, "\n");
#endif

  // Verify the function //
  verifyFunction(*derivFunction);

  // Create a ResourceTracker to track the JIT'd memory allocated to our
  // anonymous expression -- that way we can free it after executing
  // ( this should be a class member, so that the finalize function can
  //   call ExitOnErr(RT->remove()); )
  auto RT = myJIT->getMainJITDylib().createResourceTracker();

  // Add the module to the JIT
  auto TSM = llvm::orc::ThreadSafeModule(std::move(myModule), std::move(myContext));
  ExitOnErr(myJIT->addModule(std::move(TSM), RT));

  // Find the function
  auto exprSymbol = ExitOnErr(myJIT->lookup("derivFunc"));

  // Get a pointer to the function
  this->funcPtr =
      (double (*)(double *, double *, double *))(intptr_t)exprSymbol.getAddress();
}
} // namespace jit_test
