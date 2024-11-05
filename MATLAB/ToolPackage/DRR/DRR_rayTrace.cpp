/*=================================================================
AUTHOR: Michael M. Folkerts
 PICO Lab, CART, UCSD Department Radiation Medicine and Applied Science
 mmfolkerts@gmail.com
 http://bit.ly/folkerts
 Feb. 2012
DESCRIPTION: This is the MEX wrapper for the primative ray trace algorithm (Siddon's incremental)
USAGE:
 pathAttenuation = DRR_rayTrace(startPoint,endPoint,voxOffset,voxDimension,voxSize,voxelData);
NOTES:
 Input Variables:
  startPoint = [x,y,z]
  endPoint = [x,y,z]
  voxOffset = [x,y,z]
   - defines center of coord. sys. relavive to (0,0,0) corner of volume (usually mm)
  voxDimension = [xDim,yDim,zDim]
   - defines number of voxel elements along each axis
  voxSize = [xSize,ySize,zSize]
	- defines physical size of each voxel (usually mm)
  voxData = [...] <double>
   - the voxel array (usually 1/mm)
   - not to exceed 4,294,967,295 elements
 Output Variables:
  pathAttenuation <double>
   - the sum of the products of intersection lengths with attenuation values along given path
 *=================================================================*/

#include "mex.h"
//comment this line next line out when compiling with GNU Octave:
#include "matrix.h"

// my ray-trace library
#include "rayTrace.h" //should be in same folder
#include "rayTrace.cpp" //should be in same folder

extern void _main();

// mex: left-hand-side = function ( right-hand-side )
void mexFunction(
	int nlhs, 				// number of left-hand-side arguments
	mxArray *plhs[],		// pointers to each of the left-hand-side arguments
	int nrhs,				// number of right-hand-side arguments
	const mxArray *prhs[]	// pointers to each of the right-hand-side arguments
){
	//Input Varification
	//mexPrintf("nrhs = %d", nrhs);
	if(nrhs != 6)
		mexErrMsgTxt(
			"\nAUTHOR: Michael M. Folkerts"
			"\n PICO Lab, CART, UCSD Department Radiation Medicine and Applied Science"
			"\n mmfolkerts@gmail.com"
			"\n http://bit.ly/folkerts"
			"\n Feb. 2012"
			"\nUSAGE:"
			"\n pathAttenuation = DRR_rayTrace(startPoint,endPoint,voxOffset,voxDimension,voxSize,voxelData);"
			"\nNOTES:"
			"\n Input Variables (all of type <double>):"
			"\n  startPoint = [x,y,z]"
			"\n  endPoint = [x,y,z]"
			"\n  voxOffset = [x,y,z]"
			"\n   - defines center of coord. sys. relavive to (0,0,0) corner of volume (usually mm)"
			"\n  voxDimension = [xDim,yDim,zDim]"
			"\n   - defines number of voxel elements along each axis"
			"\n  voxSize = [xSize,ySize,zSize]"
			"\n   - defines physical size of each voxel (usually mm)"
			"\n  voxData = [...]"
			"\n   - the voxel array (usually 1/mm)"
			"\n   - not to exceed 4,294,967,295 elements"
			"\n Output Variables:"
			"\n  pathAttenuation"
			"\n   - the sum of the products of intersection lengths with attenuation values along given path"
		);
	// Input varification
	if(!mxIsDouble(prhs[0])) mexErrMsgTxt("startPoint must be of type double. (sorry)");
	if(!mxIsDouble(prhs[1])) mexErrMsgTxt("endPoint must be of type double. (sorry)");
	if(!mxIsDouble(prhs[2])) mexErrMsgTxt("voxOffset must be of type double. (sorry)");
	if(!mxIsDouble(prhs[3])) mexErrMsgTxt("voxDimension must be of type double. (sorry)");
	if(!mxIsDouble(prhs[4])) mexErrMsgTxt("voxSize must be of type double. (sorry)");
	if(!mxIsDouble(prhs[5])) mexErrMsgTxt("voxelData must be a matrix of type double");

	///TODO: check to make sure parameter 0-4 are of expected length (3-vectors)
	
	// Input pointers
	double *startPt =	(double*) mxGetData(prhs[0]);
	double *endPt	=	(double*) mxGetData(prhs[1]);
	double *offset	=	(double*) mxGetData(prhs[2]);
	//const int* voxDim = mxGetDimensions(prhs[5]); 	// could get dimensions automatically
	double *voxDim	=	(double*) mxGetData(prhs[3]);	// but the rayTrace function expects type double
	double *voxSize	=	(double*) mxGetData(prhs[4]);
	double *voxData	=	(double*) mxGetData(prhs[5]);

	//user feedback
	//mexPrintf("\nVoxel spacing (x, y, z) = (%g, %g, %g)\n",voxSize[0],voxSize[1],voxSize[2]);
	//mexPrintf("\tVolume dimension (x, y, z) = (%g, %g, %g)\n", voxDim[0], voxDim[1], voxDim[2]);

	// Call tracing function:
	double pathVal = rayTrace(startPt, endPt, offset, voxDim, voxSize, voxData); 

	// Output Pointers:
	plhs[0] = mxCreateDoubleScalar(pathVal);
	//could create array output if we were tracing more than one path
	//mxCreateNumericArray(3, d, mxDOUBLE_CLASS, mxREAL);
}
