
extern "C" __global__                                                     
void solve_general_flipped(double *rateConst, double *state, double *deriv,                 
           int *numReact, int *numProd, int *reactId, int *prodId,          
           int numcell, int numrxn, int numspec, int maxreact, int maxprod) 
                                                                            
{                                                                           
  size_t tid;                                                               
  int i_spec, i_rxn, i_react, i_prod;                                       
  double rate;                                                              
                                                                            
  tid = blockIdx.x * blockDim.x + threadIdx.x;                              
  if (tid < numcell) {                                                      
     for (i_spec = 0; i_spec < numspec; ++i_spec)                           
         deriv[i_spec*numcell+tid] = 0.0;                                   
     for (i_rxn = 0; i_rxn < numrxn; ++i_rxn) {                             
         rate = rateConst[i_rxn*numcell+tid];                               
         for (i_react = 0; i_react < numReact[i_rxn]; ++i_react)            
             rate *= state[reactId[i_rxn*maxreact+i_react]*numcell+tid];    
         for (i_react = 0; i_react < numReact[i_rxn]; ++i_react)            
             deriv[reactId[i_rxn*maxreact+i_react]*numcell+tid] -= rate;    
         for (i_prod = 0; i_prod < numProd[i_rxn]; ++i_prod)                
             deriv[prodId[i_rxn*maxprod+i_prod]*numcell+tid] += rate;       
     }                                                                      
  }                                                                         
}                                                                           
