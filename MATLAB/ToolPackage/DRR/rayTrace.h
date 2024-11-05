#ifndef RAYTRACE_H
#define RAYTRACE_H

template<class FP>
FP rayTrace(
     const FP* startPt,     /**< [in] a 3-vector defining the starting point of ray trace in  volume coord. sys. */
     const FP* endPt,       /**< [in] a 3-vector defining the ending point of the ray trace in volume coord. sys. */
     const FP* volOffset,   /**< [in] a 3-vector defining the origin of the volume coord. sys. relative to (0,0,0) corner */
     const FP* volDimension,/**< [in] a 3-vector defining the dimension of the volume array */
     const FP* voxSize,     /**< [in] a 3-vector defining the size of all volume elements */
     const FP* volArray     /**< [in] an array containing the values for each volume element */
     );

#endif

