#define DAMPING 0.02f

#define G_ACCEL (float4)(0.f, -9.81f, 0.f, 0.f)

#define WEIGHT_ORTHO	0.138f
#define WEIGHT_DIAG		0.097f
#define WEIGHT_ORTHO_2	0.069f
#define WEIGHT_DIAG_2	0.048f


#define ROOT_OF_2 1.4142135f
#define DOUBLE_ROOT_OF_2 2.8284271f




///////////////////////////////////////////////////////////////////////////////
// The integration kernel
// Input data:
// width and height - the dimensions of the particle grid
// d_pos - the most recent position of the cloth particle while...
// d_prevPos - ...contains the position from the previous iteration.
// elapsedTime      - contains the elapsed time since the previous invocation of the kernel,
// prevElapsedTime  - contains the previous time step.
// simulationTime   - contains the time elapsed since the start of the simulation (useful for wind)
// All time values are given in seconds.
//
// Output data:
// d_prevPos - Input data from d_pos must be copied to this array
// d_pos     - Updated positions
///////////////////////////////////////////////////////////////////////////////

  __kernel void Integrate(unsigned int width,
						unsigned int height, 
						__global float4* d_pos,
						__global float4* d_prevPos,
						float elapsedTime,
						float prevElapsedTime,
						float simulationTime) {
							
	// Make sure the work-item does not map outside the cloth
	if (get_global_id(1) >= height || get_global_id(0) >= width)
	{
		return;
	}
		
	unsigned int particleID = get_global_id(0) + get_global_id(1) * width;
	
	// Read the positions
	float4 x0 = d_pos[particleID];
	float4 xP = d_prevPos[particleID];
	float4 xT = x0;

	float4 wind, a0, v0;
	
	// This is just to keep every 8th particle of the first row attached to the bar
	if ((particleID & (7)) != 0 || particleID > width - 1)
	{
		// Compute the new one position using the Verlet position integration, taking into account gravity and wind
		wind = (float4)(0.0f, 0.0f, 1.0f, 0.0f) * (0.5f * sin(0.5f * simulationTime));
		a0 = G_ACCEL + wind;
		v0 = (prevElapsedTime == 0.0f) ? 0.0f : ((x0 - xP) / prevElapsedTime);
		xT = x0 + v0 * elapsedTime + 0.5f * a0 * elapsedTime * elapsedTime;

	}
	
	// Move the value from d_pos into d_prevPos and store the new one in d_pos
	d_prevPos[particleID] = x0;
	d_pos[particleID] = xT;

	// ADD YOUR CODE HERE!
}



///////////////////////////////////////////////////////////////////////////////
// Input data:
// pos1 and pos2 - The positions of two particles
// restDistance  - the distance between the given particles at rest
//
// Return data:
// correction vector for particle 1
///////////////////////////////////////////////////////////////////////////////
  float4 SatisfyConstraint(float4 pos1,
						 float4 pos2,
						 float restDistance){
	float4 toNeighbor = pos2 - pos1;
	return (toNeighbor - normalize(toNeighbor) * restDistance);
}

///////////////////////////////////////////////////////////////////////////////
// Input data:
// width and height - the dimensions of the particle grid
// restDistance     - the distance between two orthogonally neighboring particles at rest
// d_posIn          - the input positions
//
// Output data:
// d_posOut - new positions must be written here
///////////////////////////////////////////////////////////////////////////////

#define TILE_X 16 
#define TILE_Y 16
#define HALOSIZE 2

#define INDEX(x,y) ((x) + ((y) * width))

  struct Tile
  {
	  bool isLeftEdge, isTopEdge, isRightEdge, isBottomEdge;
  };

  struct Tile getTilePosition(int x, int y, uint width, uint height, int radius)
  {
	  struct Tile result;

	  result.isLeftEdge = x < radius;
	  result.isTopEdge = y < radius;
	  result.isRightEdge = x >= width - radius;
	  result.isBottomEdge = y >= height - radius;

	  return result;
  }

  __kernel __attribute__((reqd_work_group_size(TILE_X, TILE_Y, 1)))
	  __kernel void SatisfyConstraints(unsigned int width,
		  unsigned int height,
		  float restDistance,
		  __global float4* d_posOut,
		  __global float4 const* d_posIn) {

	  if (get_global_id(0) >= width || get_global_id(1) >= height)
		  return;


	  // ADD YOUR CODE HERE!
	  // Satisfy all the constraints (structural, shear, and bend).
	  // You can use weights defined at the beginning of this file.

	  // A ping-pong scheme is needed here, so read the values from d_posIn and store the results in d_posOut

	  // Hint: you should use the SatisfyConstraint helper function in the following manner:
	  //SatisfyConstraint(pos, neighborpos, restDistance) * WEIGHT_XXX

	  //local memory
	  __local float4 tile[TILE_Y + 2 * HALOSIZE][TILE_X + 2 * HALOSIZE];
	  __local float4 tileOut[TILE_Y][TILE_X];

	  float4 restDistanceVec = (restDistance, restDistance, restDistance, 0);

	  uint2 GID = { get_global_id(0), get_global_id(1) };
	  uint2 LID = { get_local_id(0), get_local_id(1) };
	  uint2 TID = LID + HALOSIZE;
	  struct Tile GTD1 = getTilePosition(GID.x, GID.y, width, height, 1);
	  struct Tile GTD2 = getTilePosition(GID.x, GID.y, width, height, 2);
	  struct Tile LTD = getTilePosition(LID.x, LID.y, TILE_X, TILE_Y, 2);

	  tile[TID.y][TID.x] = d_posIn[INDEX(GID.x, GID.y)];
	  tileOut[LID.y][LID.x] = 0.0f;

	  //halo regions
	  if (LTD.isLeftEdge)
	  {
		  tile[LID.y + 2][LID.x + 0] = GTD2.isLeftEdge ? NAN : d_posIn[INDEX(GID.x - 2, GID.y)];
	  }
	  else if (LTD.isRightEdge)
	  {
		  tile[LID.y + 2][LID.x + 4] = GTD2.isRightEdge ? NAN : d_posIn[INDEX(GID.x + 2, GID.y)];
	  }

	  if (LTD.isTopEdge)
	  {
		  tile[LID.y + 0][LID.x + 2] = GTD2.isTopEdge ? NAN : d_posIn[INDEX(GID.x, GID.y - 2)];
	  }
	  else if (LTD.isBottomEdge)
	  {
		  tile[LID.y + 4][LID.x + 2] = GTD2.isBottomEdge ? NAN : d_posIn[INDEX(GID.x, GID.y + 2)];
	  }

	  if (LTD.isLeftEdge && LTD.isTopEdge)
	  {
		  tile[LID.y + 0][LID.x + 0] = (GTD2.isLeftEdge || GTD2.isTopEdge) ? NAN : d_posIn[INDEX(GID.x - 2, GID.y - 2)];
	  }
	  else if (LTD.isRightEdge && LTD.isTopEdge)
	  {
		  tile[LID.y + 0][LID.x + 4] = (GTD2.isRightEdge || GTD2.isTopEdge) ? NAN : d_posIn[INDEX(GID.x + 2, GID.y - 2)];
	  }
	  else if (LTD.isLeftEdge && LTD.isBottomEdge)
	  {
		  tile[LID.y + 4][LID.x + 0] = (GTD2.isLeftEdge || GTD2.isBottomEdge) ? NAN : d_posIn[INDEX(GID.x - 2, GID.y + 2)];
	  }
	  else if (LTD.isRightEdge && LTD.isBottomEdge)
	  {
		  tile[LID.y + 4][LID.x + 4] = (GTD2.isRightEdge || GTD2.isBottomEdge) ? NAN : d_posIn[INDEX(GID.x + 2, GID.y + 2)];
	  }

	  // sync threads
	  barrier(CLK_LOCAL_MEM_FENCE);


	  if (get_global_id(0) >= width || get_global_id(1) >= height)
	  {
		  return;
	  }

	  // This is just to keep every 8th particle of the first row attached to the bar
	  if ((INDEX(GID.x, GID.y) & (7)) != 0 || INDEX(GID.x, GID.y) > width - 1)
	  {
		  // structural constraints
		  if (!GTD1.isRightEdge)
		  {
			  tileOut[LID.y][LID.x] += SatisfyConstraint(tile[TID.y][TID.x], tile[TID.y][TID.x + 1], restDistance) * WEIGHT_DIAG;
		  }
		  if (!GTD1.isLeftEdge)
		  {
			  tileOut[LID.y][LID.x] += SatisfyConstraint(tile[TID.y][TID.x], tile[TID.y][TID.x - 1], restDistance) * WEIGHT_DIAG;
		  }
		  // structural constraints
		  if (!GTD2.isRightEdge)
		  {
			  tileOut[LID.y][LID.x] += SatisfyConstraint(tile[TID.y][TID.x], tile[TID.y][TID.x + 2], restDistance * 2) * WEIGHT_DIAG_2;
		  }
		  if (!GTD2.isLeftEdge)
		  {
			  tileOut[LID.y][LID.x] += SatisfyConstraint(tile[TID.y][TID.x], tile[TID.y][TID.x - 2], restDistance * 2) * WEIGHT_DIAG_2;
		  }

		  // shear constraints
		  if (!GTD1.isRightEdge && !GTD1.isBottomEdge)
		  {
			  tileOut[LID.y][LID.x] += SatisfyConstraint(tile[TID.y][TID.x], tile[TID.y + 1][TID.x + 1], ROOT_OF_2 * restDistance) * WEIGHT_ORTHO;
		  }
		  if (!GTD1.isLeftEdge && !GTD1.isBottomEdge)
		  {
			  tileOut[LID.y][LID.x] += SatisfyConstraint(tile[TID.y][TID.x], tile[TID.y + 1][TID.x - 1], ROOT_OF_2 * restDistance) * WEIGHT_ORTHO;
		  }
		  if (!GTD1.isRightEdge && !GTD1.isTopEdge)
		  {
			  tileOut[LID.y][LID.x] += SatisfyConstraint(tile[TID.y][TID.x], tile[TID.y - 1][TID.x + 1], ROOT_OF_2 * restDistance) * WEIGHT_ORTHO;
		  }
		  if (!GTD1.isLeftEdge && !GTD1.isTopEdge)
		  {
			  tileOut[LID.y][LID.x] += SatisfyConstraint(tile[TID.y][TID.x], tile[TID.y - 1][TID.x - 1], ROOT_OF_2 * restDistance) * WEIGHT_ORTHO;
		  }
		  // shear constraints
		  if (!GTD2.isRightEdge && !GTD2.isBottomEdge)
		  {
			  tileOut[LID.y][LID.x] += SatisfyConstraint(tile[TID.y][TID.x], tile[TID.y + 2][TID.x + 2], DOUBLE_ROOT_OF_2 * restDistance) * WEIGHT_ORTHO_2;
		  }
		  if (!GTD2.isLeftEdge && !GTD2.isBottomEdge)
		  {
			  tileOut[LID.y][LID.x] += SatisfyConstraint(tile[TID.y][TID.x], tile[TID.y + 2][TID.x - 2], DOUBLE_ROOT_OF_2 * restDistance) * WEIGHT_ORTHO_2;
		  }
		  if (!GTD2.isRightEdge && !GTD2.isTopEdge)
		  {
			  tileOut[LID.y][LID.x] += SatisfyConstraint(tile[TID.y][TID.x], tile[TID.y - 2][TID.x + 2], DOUBLE_ROOT_OF_2 * restDistance) * WEIGHT_ORTHO_2;
		  }
		  if (!GTD2.isLeftEdge && !GTD2.isTopEdge)
		  {
			  tileOut[LID.y][LID.x] += SatisfyConstraint(tile[TID.y][TID.x], tile[TID.y - 2][TID.x - 2], DOUBLE_ROOT_OF_2 * restDistance) * WEIGHT_ORTHO_2;
		  }
	  }

	  tileOut[LID.y][LID.x].w = 0.0f;
	  float lenght_ = length(tileOut[LID.y][LID.x]);

	  // clamp max change rate to d/2
	  if (lenght_ > restDistance / 2) 
	  {
		  tileOut[LID.y][LID.x] *= (restDistance / 2) / lenght_;
	  }

	  // store the updated position
	  d_posOut[INDEX(GID.x, GID.y)] = tile[TID.y][TID.x] + tileOut[LID.y][LID.x];

  }


///////////////////////////////////////////////////////////////////////////////
// Input data:
// width and height - the dimensions of the particle grid
// d_pos            - the input positions
// spherePos        - The position of the sphere (xyz)
// sphereRad        - The radius of the sphere
//
// Output data:
// d_pos            - The updated positions
///////////////////////////////////////////////////////////////////////////////
__kernel void CheckCollisions(unsigned int width,
								unsigned int height, 
								__global float4* d_pos,
								float4 spherePos,
								float sphereRad){
								

	// ADD YOUR CODE HERE!
	
	if (get_global_id(1) >= height || get_global_id(0) >= width)
	{
		return;
	}

	unsigned int particleID = get_global_id(0) + get_global_id(1) * width;
	float4 distVec = d_pos[particleID] - spherePos;
	float squaredDist = (distVec.x * distVec.x) + (distVec.y * distVec.y) + (distVec.z * distVec.z);

	//Is the particle inside the sphere?
	if (squaredDist < sphereRad * sphereRad)
	{
		//put the particle outside of the sphere
		d_pos[particleID] += (sphereRad - sqrt(squaredDist)) * normalize(distVec);
	}

}

///////////////////////////////////////////////////////////////////////////////
// There is no need to change this function!
///////////////////////////////////////////////////////////////////////////////
float4 CalcTriangleNormal( float4 p1, float4 p2, float4 p3) {
    float4 v1 = p2-p1;
    float4 v2 = p3-p1;

    return cross( v1, v2);
}

///////////////////////////////////////////////////////////////////////////////
// There is no need to change this kernel!
///////////////////////////////////////////////////////////////////////////////
__kernel void ComputeNormals(unsigned int width,
								unsigned int height, 
								__global float4* d_pos,
								__global float4* d_normal){
								
    int particleID = get_global_id(0) + get_global_id(1) * width;
    float4 normal = (float4)( 0.0f, 0.0f, 0.0f, 0.0f);
    
    int minX, maxX, minY, maxY, cntX, cntY;
    minX = max( (int)(0), (int)(get_global_id(0)-1));
    maxX = min( (int)(width-1), (int)(get_global_id(0)+1));
    minY = max( (int)(0), (int)(get_global_id(1)-1));
    maxY = min( (int)(height-1), (int)(get_global_id(1)+1));
    
    for( cntX = minX; cntX < maxX; ++cntX) {
        for( cntY = minY; cntY < maxY; ++cntY) {
            normal += normalize( CalcTriangleNormal(
                d_pos[(cntX+1)+width*(cntY)],
                d_pos[(cntX)+width*(cntY)],
                d_pos[(cntX)+width*(cntY+1)]));
            normal += normalize( CalcTriangleNormal(
                d_pos[(cntX+1)+width*(cntY+1)],
                d_pos[(cntX+1)+width*(cntY)],
                d_pos[(cntX)+width*(cntY+1)]));
        }
    }
    d_normal[particleID] = normalize( normal);
}
