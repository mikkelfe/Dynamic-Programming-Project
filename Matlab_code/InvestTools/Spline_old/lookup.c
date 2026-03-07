#include "mex.h"
#include <math.h>
/* LOOKUP  
Finds the highest index in Table with a value less than X.  
Based on the HUNT algorithm in Press, et al, Numerical Recipes.
*/

void mexFunction(
   int nlhs, mxArray *plhs[],
   int nrhs, const mxArray *prhs[])
{
   mxArray *pBa;
   double *table, *x, *ind, xi;
   int i, j, jlo, jhi, inc, n, m, p, numfirst;
    
   if (nrhs<2)
       mexErrMsgTxt("Two arguments must be passed");
   if (!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]))           
       mexErrMsgTxt("Input arguments of inproper type");
   if (nrhs>2) p=(int)*mxGetPr(prhs[2]);
   else p=0;

   n=mxGetM(prhs[0])*mxGetN(prhs[0]);       /* # of elements in table */
   m = mxGetM(prhs[1])*mxGetN(prhs[1]);     /* # of elements in x */ 
   table=mxGetPr(prhs[0]);
   x=mxGetPr(prhs[1]);
   /* mxArray for output */
   plhs[0]=mxCreateDoubleMatrix(mxGetM(prhs[1]),mxGetN(prhs[1]),mxREAL);
   ind=mxGetPr(plhs[0]);                      /* pointer to output data */
   /* Lower endpoint adjustment */
   numfirst=0;
   if (p==1 || p==3)
   {
     numfirst=1;
     for (i=1;i<n;i++)
     {
       if (table[i]==table[0]) numfirst++;
       else break;
     }
   }
   /* Upper endpoint adjustment */
   if (p>=2) 
   {
     n--;
     while (table[n]==table[n-1]) n--;
   }

   jlo=0;
   for (i = 0; i < m; i++)
   {
     inc=1;
     xi=x[i];
     if (xi>=table[jlo])
     {
       jhi=jlo+1;
       while (xi>=table[jhi])
       {
         jlo=jhi;
         jhi+=inc;
         if (jhi>=n)
         {
           jhi=n;
           break;
         }
         else
         {
           inc=inc+inc;
         }
       }
     }
     else
     {
       jhi=jlo;
       jlo--;
       while (xi<table[jlo])
       {
         jhi=jlo;
         jlo-=inc;
         if (jlo<0)
         {
           jlo=-1;
           break;
         }
         else
         {
           inc=inc+inc;
         }
       }
     }
     while (jhi-jlo>1)
     {
       j=(jhi+jlo)/2;
       if (xi>=table[j]) jlo=j; 
       else jhi=j; 
     }
     ind[i]=jlo+1;
     if (jlo<0)
     {
       jlo=0; 
       if (p==1 || p==3) ind[i]=numfirst;
     }
     if (jlo==n-1) jlo=n-2;
   }
}