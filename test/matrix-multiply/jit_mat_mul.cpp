#include "leJIT.h"
#include "jit_mat_mul.h"
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

#if __cplusplus
extern "C" {
#endif

typedef void (*MatMulFunc)(int*, int*, int*);

MatMulFunc jited_func;

static llvm::AllocaInst *CreateEntryBlockAlloca(llvm::Function *TheFunction,
                                                llvm::Type *type,
                                                const std::string &VarName) {
  llvm::IRBuilder<> TmpB(&TheFunction->getEntryBlock(),
                         TheFunction->getEntryBlock().begin());
  return TmpB.CreateAlloca(type, 0, VarName.c_str());
}

void test_jited_mat_mul() {

    int rowA=2, colA=3, rowB=3, colB=2;
    int a[rowA][colA],b[rowB][colB],c[rowA][colB];
    int i,j;

    if(colA != rowB){
      printf("unmatched A_mat's column and B_mat's row\n");
      return;
    }

    printf("jit initialization...\n");
    for(i=0;i<rowA;i++)
       for(j=0;j<colA;j++)
          a[i][j]=i*colA+j;

    for(i=0;i<rowB;i++)
       for(j=0;j<colB;j++)
          b[i][j]=i+j;

  (*jited_func)((int*)a, (int*)b, (int*)c);

  //for printing result
    for(i=0;i<rowA;i++){
       for(j=0;j<colB;j++)
          printf("%d\t",c[i][j]);
       printf("\n");
    }
}

void create_mat_mul_function(const int rowA, const int colA, const int colB) {

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  static llvm::ExitOnError ExitOnErr;
  std::unique_ptr<llvm::orc::leJIT> myJIT = ExitOnErr(llvm::orc::leJIT::Create());
  static std::unique_ptr<llvm::LLVMContext> myContext;
  static std::unique_ptr<llvm::IRBuilder<>> builder;
  static std::unique_ptr<llvm::Module> myModule;

  // open a new context and module
  myContext = std::make_unique<llvm::LLVMContext>();
  myModule = std::make_unique<llvm::Module>("matrix multiply jit code", *myContext);
  myModule->setDataLayout(myJIT->getDataLayout());

  // create a new builder for the module
  builder = std::make_unique<llvm::IRBuilder<>>(*myContext);

  // types
  llvm::Type *intType = llvm::Type::getInt32Ty(*myContext);
  llvm::Type *intPtr = llvm::Type::getInt32Ty(*myContext)->getPointerTo();
  llvm::Type *vd = llvm::Type::getVoidTy(*myContext);

  // Code generation //

  // Prototype
  std::vector<llvm::Type *> matMulArgsV{intPtr, intPtr, intPtr};
  llvm::FunctionType *matMulFunctionType =
      llvm::FunctionType::get(vd, matMulArgsV, false);
  llvm::Function *matMulFunction =
      llvm::Function::Create(matMulFunctionType, llvm::Function::ExternalLinkage,
                             "matMulFunc", myModule.get());
  llvm::Function::arg_iterator argIter = matMulFunction->arg_begin();
  llvm::Value *a = argIter++;
  a->setName("a");
  llvm::Value *b = argIter++;
  b->setName("b");
  llvm::Value *c = argIter++;
  c->setName("c");

  // function body //

  // set up array arguments
  llvm::BasicBlock *BB =
      llvm::BasicBlock::Create(*myContext, "entry", matMulFunction);
  builder->SetInsertPoint(BB);
  llvm::AllocaInst *allocaA =
      CreateEntryBlockAlloca(matMulFunction, intPtr, "a");
  llvm::AllocaInst *allocaB =
      CreateEntryBlockAlloca(matMulFunction, intPtr, "b");
  llvm::AllocaInst *allocaC =
      CreateEntryBlockAlloca(matMulFunction, intPtr, "c");
  builder->CreateStore(a, allocaA);
  builder->CreateStore(b, allocaB);
  builder->CreateStore(c, allocaC);
  llvm::Value *aPtr = builder->CreateLoad(intPtr, allocaA);
  llvm::Value *bPtr = builder->CreateLoad(intPtr, allocaB);
  llvm::Value *cPtr = builder->CreateLoad(intPtr, allocaC);

  // matrix multiply calculation variables
  llvm::Value *idxList[1]; // matrix indices
  llvm::Value *aElem; // a value
  llvm::Value *bElem; // b value
  llvm::Value *cElem; // c value
  llvm::Value *tmpVal, *tmpSum, *tmpMul; // temporary values

  // do the matrix multiplication
  for (int i=0; i<rowA; ++i) {
    for (int j=0; j<colB; ++j) {
      idxList[0] = llvm::ConstantInt::get(*myContext, llvm::APInt(64,i*colB+j));
      tmpSum = llvm::ConstantInt::get(*myContext, llvm::APInt(32, 0));
      for (int k=0; k<colA; ++k) {
        idxList[0] = llvm::ConstantInt::get(*myContext, llvm::APInt(64,i*colA+k));
        aElem = builder->CreateGEP(intType, aPtr, idxList, "aElemPtr");
        tmpMul = builder->CreateLoad(intType, aElem, "aElemVal");
        idxList[0] = llvm::ConstantInt::get(*myContext, llvm::APInt(64,k*colB+j));
        bElem = builder->CreateGEP(intType, bPtr, idxList, "bElemPtr");
        tmpVal = builder->CreateLoad(intType, bElem, "bElemVal");
        tmpMul = builder->CreateMul(tmpMul, tmpVal, "multPair");
        tmpSum = builder->CreateAdd(tmpSum, tmpMul, "addPair");
      }
      idxList[0] = llvm::ConstantInt::get(*myContext, llvm::APInt(64,i*colB+j));
      cElem = builder->CreateGEP(intType, cPtr, idxList, "cElemPtr");
      builder->CreateStore(tmpSum, cElem);
    }
  }
  builder->CreateRetVoid();

  // print llvm code
#if 0
  std::fprintf(stderr, "Generated function definition:\n");
  matMulFunction->print(llvm::errs());
  std::fprintf(stderr, "\n");
#endif

  // Verify the function //
  verifyFunction(*matMulFunction);

  // Create a ResourceTracker to track the JIT'd memory allocated to our
  // anonymous expression -- that way we can free it after executing
  // ( this should be a class member, so that the finalize function can
  //   call ExitOnErr(RT->remove()); )
  auto RT = myJIT->getMainJITDylib().createResourceTracker();

  // Add the module to the JIT
  auto TSM = llvm::orc::ThreadSafeModule(std::move(myModule), std::move(myContext));
  ExitOnErr(myJIT->addModule(std::move(TSM), RT));

  // Find the function
  auto exprSymbol = ExitOnErr(myJIT->lookup("matMulFunc"));

  // return a pointer to the generated function
  jited_func = (MatMulFunc) exprSymbol.getAddress();

  // run the test of the jited function
  test_jited_mat_mul();

}

#if __cplusplus
}
#endif
