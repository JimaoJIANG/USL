/**
 * This is the primative ray trace function
 * all 3-vectors should be an array of type <type> with length of 3
 */

#include <math.h>
#include <algorithm>
#include <stdio.h>

#define RAY_TOL 10e-20 //check for double vs float reqs.
#define STRIDE_TOL 10e-10 //check for double vs float reqs.

using namespace std;

template<class FP>
FP rayTrace(
	 const FP* startPt,	 /**< [in] a 3-vector defining the starting point of ray trace in  volume coord. sys. */
	 const FP* endPt,	   /**< [in] a 3-vector defining the ending point of the ray trace in volume coord. sys. */
	 const FP* volOffset,   /**< [in] a 3-vector defining the origin of the volume coord. sys. relative to (0,0,0) corner */
	 const FP* volDimension,/**< [in] a 3-vector defining the dimension of the volume array */
	 const FP* voxSize,	 /**< [in] a 3-vector defining the size of all volume elements */
	 const FP* volArray	 /**< [in] an array containing the values for each volume element */
	 ){

	FP voxPlanesMin[3]; // defines lower boundary for volume in each dimension
	FP voxPlanesMax[3]; // defines upper boundary for volume in each dimension
	FP ray[3];		  // the ray connecting starting and ending points
	FP tm[2][3];		// parametric plane intersection values (min and max for each dimension)
	int direction[3];   // defins the direction (+ or -) through the data in each dimension

	// compute volume boundaries:
	#pragma unroll
	for(int i=0;i<3;i++){
		voxPlanesMin[i] = - volOffset[i];
		voxPlanesMax[i] = - volOffset[i] + voxSize[i] * volDimension[i];
	}

	//Ray vector from source to pixel:
	#pragma unroll
	for(int i=0;i<3;i++)
		ray[i] = endPt[i] - startPt[i] + RAY_TOL; //ray to be traced
	//for debugging:
	//printf("ray: %f, %f, %f\t",ray[0],ray[1],ray[2]);

	//PLANE_INTERSECTION = P1 + t(P2-P1)
	//min and max parameteric values:
	#pragma unroll
	for(int i=0;i<3;i++){
		tm[0][i] = ( voxPlanesMin[i] - startPt[i] ) / ray[i];//tmin for each dimension
		tm[1][i] = ( voxPlanesMax[i] - startPt[i] ) / ray[i];//tmax for each dimension
	}
	//for debugging:
	//printf("\ttmin = %f,%f,%f, tmax=%f,%f,%f\n",tm[0][0],tm[0][1],tm[0][2],tm[1][0],tm[1][1],tm[1][2]);

	//find direction through index spcace of volArray:
	#pragma unroll
	for(int i=0;i<3;i++)
		direction[i] = endPt[i] < startPt[i] ? -1 : 1;

	//RL Siddon's way of sorting out parametric starting and ending points:
	FP tmax[] = {max(tm[0][0],tm[1][0]),max(tm[0][1],tm[1][1]),max(tm[0][2],tm[1][2]),1.0f};
	FP tmin[] = {min(tm[0][0],tm[1][0]),min(tm[0][1],tm[1][1]),min(tm[0][2],tm[1][2]),0.0f};
	FP* ti_ptr = max_element(tmin, tmin+3); //the max of all the min values (entrance point or start point)
	FP* tf_ptr = min_element(tmax, tmax+3); //the min  of all the max values (exit point or end point)
	FP ti = *ti_ptr;
	FP tf = *tf_ptr;

	if(ti < tf){	//if the ray intersects the volume
		FP start[3];
		FP sx,sy,sz;
		FP tx,ty,tz;
		int ix,iy,iz;

		tf -= STRIDE_TOL; //attempt to not overshoot boundary!

		//step sizes:
		sx = fabs( voxSize[0]/ray[0] );
		sy = fabs( voxSize[1]/ray[1] );
		sz = fabs( voxSize[2]/ray[2] );

		//starting point
		#pragma unroll
		for(int i=0;i<3;i++)
			start[i] = ray[i]*ti + startPt[i];

		//compute starting index and next parametric intersection values
		ix = int( floor( (start[0]-voxPlanesMin[0]) / voxSize[0] ) );
		iy = int( floor( (start[1]-voxPlanesMin[1]) / voxSize[1] ) );
		iz = int( floor( (start[2]-voxPlanesMin[2]) / voxSize[2] ) );
		tx = ( voxPlanesMin[0] + FP( ix + int(direction[0]>0) ) * voxSize[0] - startPt[0] ) / (ray[0]);
		ty = ( voxPlanesMin[1] + FP( iy + int(direction[1]>0) ) * voxSize[1] - startPt[1] ) / (ray[1]);
		tz = ( voxPlanesMin[2] + FP( iz + int(direction[2]>0) ) * voxSize[2] - startPt[2] ) / (ray[2]);

		//trick to find starting dimension:
		int startDim = int(ti_ptr - tmin);
		
		//account for start-on-boundary cases:
		if(startDim == 0){
			ix = (ti == tm[1][0]) ? int(volDimension[0]) - 1 : 0; //if a max boundary, index is dim - 1
			tx = ti + sx; //next intersection
		}else if(startDim == 1){
			iy = (ti == tm[1][1]) ? int(volDimension[1]) - 1 : 0; //if a max boundary, index is dim - 1
			ty = ti + sy; //next intersection
		}else if(startDim == 2){
			iz = (ti == tm[1][2]) ? int(volDimension[2]) - 1 : 0; //if a max boundary, index is dim - 1
			tz = ti + sz; //next intersection
		}
		//for debugging:
		//printf("vox Idx: %i %i %i\n",ix,iy,iz);
		
		// initialize tracing intensity
		FP pixIntensity = 0.0f;
		// initialize indexing modification
		int ijk = ix + iy*int(volDimension[0]) + iz*int(volDimension[0])*int(volDimension[1]);
		direction[1]*=int(volDimension[0]);
		direction[2]*=int(volDimension[0])*int(volDimension[1]);

		//for error checking
		int maxIdx = int(volDimension[0])*int(volDimension[1])*int(volDimension[2]);

		//start the incremental ray trace
		do{
			//avoid reading outside of voxel array
			if(ijk >= (maxIdx-1) || ijk < 0){
				// If you get this error, adjust your tolerances (defined at top of this file):
				printf("Warning: Voxel Index Over-run/Under-run @ ijk = %d, skipping ray...\n", ijk);
				break;
			}
			//arranged to encourage FMA+A
			else if( tx<=ty && tx<=tz){
				pixIntensity += (tx - ti)*volArray[ijk];
				ijk += direction[0];
				ti = tx;
				tx = tx + sx;
			}else if( ty<=tz ){
				pixIntensity += (ty - ti)*volArray[ijk];
				ijk += direction[1];
				ti = ty;
				ty = ty + sy;
			}else{
				pixIntensity += (tz - ti)*volArray[ijk];
				ijk += direction[2];
				ti = tz;
				tz = tz + sz;
			}
		}while(ti < tf);

		return pixIntensity * FP( sqrt( pow(ray[0],2) + pow(ray[1],2) + pow(ray[2],2) ) ); //pixIntesity*length
		//pixIntensity better be in inverse units(length)
	}//end if
	else{ //if no intersection with volume
		return FP(0.0f);
	}
}

