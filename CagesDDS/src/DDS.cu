#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/device_ptr.h>
#include<thrust/sort.h>
#include<thrust/sequence.h>
#include<thrust/gather.h>
#include<thrust/count.h>
#include <thrust/execution_policy.h>
#include<thrust/copy.h>
#include "DDS.h"

#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <sstream>


void throw_on_cuda_error(cudaError_t code, const char* file, int line)
{
	if (code != cudaSuccess)
	{
		std::stringstream ss;
		ss << file << "(" << line << ")";
		std::string file_and_line;
		ss >> file_and_line;
		throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
	}
}

__global__ void CountFragsKernelS(int vxNum, int globalW, int globalH, float viewWidth, float* vmMat, float* pvmMat, float* vpos, float* vrad, int* xfcount, bool disks)
{
	int v = blockIdx.x * blockDim.x + threadIdx.x;

	if (v < vxNum)
	{

		//get vertex//
		float x = vpos[3 * v + 0];
		float y = vpos[3 * v + 1];
		float z = vpos[3 * v + 2];
		float w = 1.0;

		float posXpvm = pvmMat[0] * x + pvmMat[4] * y + pvmMat[8] * z + pvmMat[12] * w;
		float posYpvm = pvmMat[1] * x + pvmMat[5] * y + pvmMat[9] * z + pvmMat[13] * w;
		float posZpvm = pvmMat[2] * x + pvmMat[6] * y + pvmMat[10] * z + pvmMat[14] * w;
		float posWpvm = pvmMat[3] * x + pvmMat[7] * y + pvmMat[11] * z + pvmMat[15] * w;

		int xscreen = (int)(((posXpvm / posWpvm) / 2 + 0.5) * globalW);
		int yscreen = (int)(((posYpvm / posWpvm) / 2 + 0.5) * globalH);

		float posZvm = vmMat[2] * x + vmMat[6] * y + vmMat[10] * z + vmMat[14] * w;

		//int SplatSize = int(2.0 * (vrad[v] / viewWidth) * globalW);
		int SplatSize = int(round(2.0 * (vrad[v] / viewWidth) * globalW));
		

		//if (SplatSize < 1)
		//	SplatSize = 1;

		//min and max?

		int SplatRad = SplatSize / 2;
		float SplatRadSqr;

		if(disks)
			SplatRadSqr = (SplatRad + 0.5) * (SplatRad + 0.5);

		int xstart = xscreen - SplatRad;
		int xend = xscreen + SplatRad;
		int ystart = yscreen - SplatRad;
		int yend = yscreen + SplatRad;
		/*if (SplatRad % 2 == 0 && SplatRad != 0)
		{
			xend--;
			yend--;
		}*/

		for (int x = xstart; x <= xend; x++)
		//for (int x = xstart; x <= xstart; x++)
		{
			if (x<0 || x>globalW - 1)
				continue;
			for (int y = ystart; y <= yend; y++)
			//for (int y = ystart; y <= ystart; y++)
			{
				if (y<0 || y>globalH - 1)
					continue;

				if (disks)
				{
					float xdiff = abs(x - xscreen);
					float ydiff = abs(y - yscreen);
					if (xdiff > 0) xdiff -= 0.5; if (ydiff > 0) ydiff -= 0.5;
					if (SplatRad > 1 && xdiff * xdiff + ydiff * ydiff > SplatRadSqr)
						continue;
				}

				int pxl = x + y * globalW; //from x and y

				if (pxl >= 0 && pxl < globalW * globalH)
					atomicAdd(&xfcount[pxl], 1);
			}
		}


	}
}


void CountFragsCudaS(int vxNum, int globalW, int globalH, float viewWidth, float* vmMat, float* pvmMat, float* vpos, float* vrad, int* xfcount, bool disks)
{
	CountFragsKernelS << <vxNum / 256 + 1, 256 >> > (vxNum, globalW, globalH, viewWidth, vmMat, pvmMat, vpos, vrad, xfcount, disks);
}


void SetOffsetVectorCudaS(int pxNum, int* xfcount, int* xfoffset)
{
	thrust::device_ptr<int> o = thrust::device_pointer_cast(xfoffset);
	thrust::device_ptr<int> c = thrust::device_pointer_cast(xfcount);

	//call thrust function
	thrust::exclusive_scan(c, c + pxNum, o);
}

int GetFragsNumCudaS(int vxNum, int* xfcount)
{
	thrust::device_ptr<int> c = thrust::device_pointer_cast(xfcount);

	//get count of xfcount//
	int FragsNum = thrust::reduce(c, c + vxNum, (int)0, thrust::plus<int>());

	return FragsNum;
}

__device__
unsigned long long GenerateDepthPixelKeyS(float depth, int pixel)
{
	unsigned long long result = pixel;
	result = result << 32;

	//unsigned long long result=0;

	const float lineParameter = depth;
	//uint converted_key = *((uint *)&lineParameter);
	unsigned int converted_key = *((unsigned int*)&lineParameter);
	const unsigned int mask = ((converted_key & 0x80000000) ? 0xffffffff : 0x80000000);
	converted_key ^= mask;

	result |= (unsigned long long)(converted_key);

	return result;

}



__global__ void ProjectFragsKernelS(int vxNum, int globalW, int globalH, float viewWidth, float* vmMat, float* pvmMat, float* vpos, float* vrad, int* xfcount, int* xfoffset, float* FragDepth, float* FragRad, float* FragDist, unsigned long long* FragDepthPixel, bool disks)
{
	int v = blockIdx.x * blockDim.x + threadIdx.x;

	if (v < vxNum)
	{

		//get vertex//
		float x = vpos[3 * v + 0];
		float y = vpos[3 * v + 1];
		float z = vpos[3 * v + 2];
		float w = 1.0;

		float posXpvm = pvmMat[0] * x + pvmMat[4] * y + pvmMat[8] * z + pvmMat[12] * w;
		float posYpvm = pvmMat[1] * x + pvmMat[5] * y + pvmMat[9] * z + pvmMat[13] * w;
		float posZpvm = pvmMat[2] * x + pvmMat[6] * y + pvmMat[10] * z + pvmMat[14] * w;
		float posWpvm = pvmMat[3] * x + pvmMat[7] * y + pvmMat[11] * z + pvmMat[15] * w;

		int xscreen = (int)(((posXpvm / posWpvm) / 2 + 0.5) * globalW);
		int yscreen = (int)(((posYpvm / posWpvm) / 2 + 0.5) * globalH);

		float posZvm = vmMat[2] * x + vmMat[6] * y + vmMat[10] * z + vmMat[14] * w;

		//int SplatSize = int(2.0 * (vrad[v] / viewWidth) * globalW);
		int SplatSize = int(round(2.0 * (vrad[v] / viewWidth) * globalW));


		//if (SplatRad < 1)
		//	SplatRad = 1;

		//min and max?

		int SplatRad = SplatSize / 2;
		float SplatRadSqr;

		if (disks)
			SplatRadSqr = (SplatRad + 0.5) * (SplatRad + 0.5);

		//float rad = vrad[v];
		float rad = vrad[v] * 1.25;
		//float depth = -posZvm;
		//float depth = -posZvm - rad;
		float depth = posZvm - rad;

		int xstart = xscreen - SplatRad;
		int xend = xscreen + SplatRad;
		int ystart = yscreen - SplatRad;
		int yend = yscreen + SplatRad;
		/*if (SplatRad % 2 == 0 && SplatRad != 0)
		{
			xend--;
			yend--;
		}*/

		for (int x = xstart; x <= xend; x++)
		{
			if (x<0 || x>globalW - 1)
				continue;
			for (int y = ystart; y <= yend; y++)
			{
				if (y<0 || y>globalH - 1)
					continue;

				if (disks)
				{
					float xdiff = abs(x - xscreen);
					float ydiff = abs(y - yscreen);
					if (xdiff > 0) xdiff -= 0.5; if (ydiff > 0) ydiff -= 0.5;
					if (SplatRad > 1 && xdiff * xdiff + ydiff * ydiff > SplatRadSqr)
						continue;
				}

				int pxl = x + y * globalW; //from x and y

				int index, offset;
				if (pxl >= 0 && pxl < globalW * globalH)
				{
					offset = xfoffset[pxl];
					index = atomicAdd(&xfcount[pxl], 1) + offset;

					FragDepth[index] = depth;
					FragRad[index] = rad;
					//FragDist[index] = abs(x - xscreen) + abs(y - yscreen);
					FragDist[index] = sqrtf((x - xscreen) * (x - xscreen) + (y - yscreen) * (y - yscreen));
					FragDepthPixel[index] = GenerateDepthPixelKeyS(depth, pxl);

				}
			}
		}


	}

}

void ProjectFragsCudaS(int vxNum, int globalW, int globalH, float viewWidth, float* vmMat, float* pvmMat, float* vpos, float* vrad, int* xfcount, int* xfoffset, float* FragDepth, float* FragRad, float* FragDist, unsigned long long* FragDepthPixel, bool disks)
{
	ProjectFragsKernelS << <vxNum / 256 + 1, 256 >> > (vxNum, globalW, globalH, viewWidth, vmMat, pvmMat, vpos, vrad, xfcount, xfoffset, FragDepth, FragRad, FragDist, FragDepthPixel, disks);
}

//works fine as long as #frags is ok. when not, need reformulating, so that not all buffers are allocated at same time//
void SortFragsCudaS(int FragsNum, float* FragDepth, float* FragRad, float* FragDist, unsigned long long* FragDepthPixel)
{
	//device pointers//
	thrust::device_ptr<float> fd = thrust::device_pointer_cast(FragDepth);
	thrust::device_ptr<float> fr = thrust::device_pointer_cast(FragRad);
	thrust::device_ptr<float> fs = thrust::device_pointer_cast(FragDist);
	thrust::device_ptr<unsigned long long> fdp = thrust::device_pointer_cast(FragDepthPixel);

	//tmp buffers for thrust::gather//
	float* FragDepthTmp;
	float* FragRadTmp;
	float* FragDistTmp;
	cudaMalloc((void**)&FragDepthTmp, FragsNum * sizeof(float));
	cudaMalloc((void**)&FragRadTmp, FragsNum * sizeof(float));
	cudaMalloc((void**)&FragDistTmp, FragsNum * sizeof(float));
	thrust::device_ptr<float> fdt = thrust::device_pointer_cast(FragDepthTmp);
	thrust::device_ptr<float> frt = thrust::device_pointer_cast(FragRadTmp);
	thrust::device_ptr<float> fst = thrust::device_pointer_cast(FragDistTmp);
	
	//init an index buffer//
	unsigned int* FragIndex;
	cudaMalloc((void**)&FragIndex, FragsNum * sizeof(unsigned int));
	thrust::device_ptr<unsigned int> fi = thrust::device_pointer_cast(FragIndex);
	thrust::sequence(fi, fi + FragsNum, 0);


	//sort depth and index//
	thrust::sort_by_key(fdp, fdp + FragsNum, fi);


	//change all other arrays based on the sorted index//
	thrust::gather(fi, fi + FragsNum, fd, fdt);
	thrust::gather(fi, fi + FragsNum, fr, frt);
	thrust::gather(fi, fi + FragsNum, fs, fst);
	cudaMemcpy(FragDepth, FragDepthTmp, FragsNum * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(FragRad, FragRadTmp, FragsNum * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(FragDist, FragDistTmp, FragsNum * sizeof(float), cudaMemcpyDeviceToDevice);

}


__global__
void PileUnionKernelS(int pxnum, int* xfcount, int* xfoffset, float* FragDepth, float* FragRad, float* FragDist, float* pstartBig, float* pendBig, int* ppixelBig, int* xpcount)
{
	int pxl = blockIdx.x * blockDim.x + threadIdx.x;

	if (pxl < pxnum)
	{
		int count = xfcount[pxl];//get count//
		int offset = xfoffset[pxl];//get offset//

		if (count != 0)
		{
			int pileIndex = offset;
			float currstart, currend; int currdist;
			for (int index = offset; index < offset + count; index++)
			{
				//read start and end
				float start = FragDepth[index];
				float end = start + 2 * FragRad[index];
				float dist = FragDist[index];
				//get pixel too?

				if (index == offset) //if newpile, set currstart and currend
				{
					currstart = start;
					currend = end;
					currdist = dist;
				}
				else
				{
					if (start < currend) //if start less than currend, then same pile
					//if(false)
					{
						if (end > currend)//if end bigger than currend, update currend
							currend = end;

						//vertex thing
						if (dist < currdist)
							currdist = dist;
					}
					else //if start bigger than currend, then save currpile, and make a new pile (with new currstart and currend)
					{
						//save current pile at pileIndex
						pstartBig[pileIndex] = currstart;
						pendBig[pileIndex] = currend;
						ppixelBig[pileIndex] = pxl;
						pileIndex++;

						currstart = start;
						currend = end;
						currdist = dist;
					}
				}

			}

			//save last pile at pileIndex
			pstartBig[pileIndex] = currstart;
			pendBig[pileIndex] = currend;
			ppixelBig[pileIndex] = pxl;

			pileIndex++;
			atomicAdd(&xpcount[pxl], pileIndex - offset);
		}
	}

}

__global__
void PileBlendKernelS(int pxnum, int* xfcount, int* xfoffset, float* FragDepth, float* FragRad, float* FragDist, float* pstartBig, float* pendBig, int* ppixelBig, int* xpcount)
{
	int pxl = blockIdx.x * blockDim.x + threadIdx.x;

	if (pxl < pxnum)
	{
		int count = xfcount[pxl];//get count//
		int offset = xfoffset[pxl];//get offset//

		if (count != 0)
		{
			int pileIndex = offset;
			float currstart, currend, currdist;
			float SumStartDepth, SumEndDepth;	float SumStart, SumEnd;

			int fFirstIndex, fLastIndex, fFurthestEndIndex; //fFirstIndex: first frag in pile, and has nearest start. fLastIndex: last frag in pile. FurthestEndIndex: has furthest end.//

			for (int index = offset; index < offset + count; index++)
			{
				float depth = FragDepth[index];
				float rad = FragRad[index];

				//read start and end
				float start = depth;
				float end = start + 2 * rad;
				float dist = FragDist[index];

				if (index == offset) //if newpile, set currstart and currend//
				{
					currstart = start;
					currend = end;
					currdist = dist;

					fFirstIndex = index;
					fFurthestEndIndex = index;
					
				}
				else
				{
					if (start < currend) //if start less than currend, then same pile//
					{
						if (end > currend)//if end bigger than currend, update currend//
						{
							currend = end;
							fFurthestEndIndex = index;
						}

						//vertex thing
						if (dist < currdist)
							currdist = dist;

					}
					else //if start bigger than currend, then save currpile, and make a new pile (with new currstart and currend)//
					{
						//save pile//
						{
							ppixelBig[pileIndex] = pxl;

							fLastIndex = index - 1;

							float e;
							SumStartDepth = 0.0; SumEndDepth = 0.0; SumStart = 0.0; SumEnd = 0.0;
							float FirstEnd = FragDepth[fFirstIndex] + 2 * FragRad[fFirstIndex];
							float FurthestEndStart = FragDepth[fFurthestEndIndex];
							for (int j = fFirstIndex; j <= fLastIndex; j++)
							{
								float jStart = FragDepth[j];
								float jEnd = FragDepth[j] + 2 * FragRad[j];
								if (jStart < FirstEnd || jEnd > FurthestEndStart)
								{
									e = expf(-FragDist[j] * FragDist[j]);
									if (e < 0.000001) e = 0.000001;
									//e = expf(-FragDist[j] * FragDist[j])/(5*5);
									//if (e < 0.000001) e = 0.000001;
									//if (e < 0.000001) e = 0.1;
								}

								if (jStart < FirstEnd)
								{
									SumStartDepth += (jStart * e); SumStart += e; 
								}
								if (jEnd > FurthestEndStart)
								{
									SumEndDepth += (jEnd * e); SumEnd += e; 
								}
							}
							pstartBig[pileIndex] = (SumStartDepth / SumStart);
							pendBig[pileIndex] = (SumEndDepth / SumEnd);


							pileIndex++;
						}

						//start a new pile//
						currstart = start;
						currend = end;
						currdist = dist;

						fFirstIndex = index;
						fFurthestEndIndex = index;
					}
				}

			} //end of loop//

			
			//save pile//
			{
				ppixelBig[pileIndex] = pxl;

				fLastIndex = offset + count - 1;

				float e;
				SumStartDepth = 0.0; SumEndDepth = 0.0; SumStart = 0.0; SumEnd = 0.0;//0s in sums
				float FirstEnd = FragDepth[fFirstIndex] + 2 * FragRad[fFirstIndex];
				float FurthestEndStart = FragDepth[fFurthestEndIndex];
				for (int j = fFirstIndex; j <= fLastIndex; j++)
				{
					float jStart = FragDepth[j];
					float jEnd = FragDepth[j] + 2 * FragRad[j];
					if (jStart < FirstEnd || jEnd > FurthestEndStart)
					{
						e = expf(-FragDist[j] * FragDist[j]);
						if (e < 0.000001) e = 0.000001;
					}

					
					if (jStart < FirstEnd)
					{
						SumStartDepth += (FragDepth[j] * e); SumStart += e; 
					}
					if (jEnd > FurthestEndStart)
					{
						SumEndDepth += (FragDepth[j] + 2 * FragRad[j]) * e; SumEnd += e; 
					}
				}
				pstartBig[pileIndex] = (SumStartDepth / SumStart);
				pendBig[pileIndex] = (SumEndDepth / SumEnd);
				
				pileIndex++;
			}
			


			atomicAdd(&xpcount[pxl], pileIndex - offset);
		}
	}

}

__global__
void PileKernelS(int pxnum, int* xfcount, int* xfoffset, float* FragDepth, float* FragRad, float* FragDist, float* pstartBig, float* pendBig, int* ppixelBig, float* pdepthBig, int* xpcount)
{
	int pxl = blockIdx.x * blockDim.x + threadIdx.x;

	if (pxl < pxnum)
	{
		int count = xfcount[pxl];//get count//
		int offset = xfoffset[pxl];//get offset//

		if (count != 0)
		{
			int pileIndex = offset;
			float currstart, currend; int currdist;
			float wdepthSum, wSum; float e;
			for (int index = offset; index < offset + count; index++)
			{
				//read start and end
				float start = FragDepth[index];
				float rad = FragRad[index];
				float end = start + 2 * rad;
				float dist = FragDist[index];
				
				//get pixel too?

				e = expf(-dist * dist);
				if (e < 0.000001) e = 0.000001;

				if (index == offset) //if newpile, set currstart and currend
				{
					currstart = start;
					currend = end;
					currdist = dist;
					
					wdepthSum = e * (start + rad);
					wSum = e;
				}
				else
				{
					if (start < currend) //if start less than currend, then same pile
					//if(false)
					{
						if (end > currend)//if end bigger than currend, update currend
							currend = end;

						//vertex thing
						if (dist < currdist)
							currdist = dist;

						wdepthSum += e * (start + rad);
						wSum += e;
					}
					else //if start bigger than currend, then save currpile, and make a new pile (with new currstart and currend)
					{
						//save current pile at pileIndex
						pstartBig[pileIndex] = currstart;
						pendBig[pileIndex] = currend;
						ppixelBig[pileIndex] = pxl;
						pdepthBig[pileIndex] = wdepthSum / wSum;
						pileIndex++;

						currstart = start;
						currend = end;
						currdist = dist;
						wdepthSum = e * (start + rad);
						wSum = e;
					}
				}

			}

			//save last pile at pileIndex
			pstartBig[pileIndex] = currstart;
			pendBig[pileIndex] = currend;
			ppixelBig[pileIndex] = pxl;
			pdepthBig[pileIndex] = wdepthSum / wSum;

			pileIndex++;
			atomicAdd(&xpcount[pxl], pileIndex - offset);
		}
	}

}


__global__
void PileResampleKernelS(int pxnum, int* xfcount, int* xfoffset, float* FragDepth, float* FragRad, float* FragDist, float* pstartBig, float* pendBig, int* ppixelBig, float* pdepthBig, int* xpcount)
{
	int pxl = blockIdx.x * blockDim.x + threadIdx.x;

	if (pxl < pxnum)
	{
		int count = xfcount[pxl];//get count//
		int offset = xfoffset[pxl];//get offset//

		if (count != 0)
		{
			int pileIndex = offset;
			float currstart, currend; int currdist;
			float wdepthSum, wSum; float e;
			for (int index = offset; index < offset + count; index++)
			{
				//read start and end
				float start = FragDepth[index];
				float rad = FragRad[index];
				float end = start + 2 * rad;
				float dist = FragDist[index];

				//get pixel too?

				e = expf(-dist * dist);
				if (e < 0.000001) e = 0.000001;

				if (index == offset) //if newpile, set currstart and currend
				{
					currstart = start;
					currend = end;
					currdist = dist;

					wdepthSum = e * (start + rad);
					wSum = e;
				}
				else
				{
					if (start < currend) //if start less than currend, then same pile
					//if(false)
					{
						if (end > currend)//if end bigger than currend, update currend
							currend = end;

						//vertex thing
						if (dist < currdist)
							currdist = dist;

						wdepthSum += e * (start + rad);
						wSum += e;
					}
					else //if start bigger than currend, then save currpile, and make a new pile (with new currstart and currend)
					{
						//save current pile at pileIndex
						ppixelBig[pileIndex] = pxl;
						pdepthBig[pileIndex] = wdepthSum / wSum;
						pileIndex++;

						currstart = start;
						currend = end;
						currdist = dist;
						wdepthSum = e * (start + rad);
						wSum = e;
					}
				}

			}

			//save last pile at pileIndex
			ppixelBig[pileIndex] = pxl;
			pdepthBig[pileIndex] = wdepthSum / wSum;

			pileIndex++;
			atomicAdd(&xpcount[pxl], pileIndex - offset);
		}
	}

}


void PileCudaS(int pxNum, int* xfcount, int* xfoffset, float* FragDepth, float* FragRad, float* FragDist,   float* pstartBig, float* pendBig, int* ppixelBig, float* pdepthBig, int* xpcount, pileMode pmode)
{
	switch (pmode)
	{
	case blnd:
		PileBlendKernelS << <pxNum / 256 + 1, 256 >> > (pxNum, xfcount, xfoffset, FragDepth, FragRad, FragDist, pstartBig, pendBig, ppixelBig, xpcount);
		break;
	case unn:
		PileUnionKernelS << <pxNum / 256 + 1, 256 >> > (pxNum, xfcount, xfoffset, FragDepth, FragRad, FragDist, pstartBig, pendBig, ppixelBig, xpcount);
		break;
	case both:
		PileKernelS << <pxNum / 256 + 1, 256 >> > (pxNum, xfcount, xfoffset, FragDepth, FragRad, FragDist, pstartBig, pendBig, ppixelBig, pdepthBig, xpcount);
		break;
	case resample:
		PileResampleKernelS << <pxNum / 256 + 1, 256 >> > (pxNum, xfcount, xfoffset, FragDepth, FragRad, FragDist, pstartBig, pendBig, ppixelBig, pdepthBig, xpcount);
	}

}

struct is_nonMinusOnef
{
	__host__ __device__
		bool operator()(const float x)
	{
		return (x != -1);
	}
};

struct is_nonMinusOnei
{
	__host__ __device__
		bool operator()(const int x)
	{
		return (x != -1);
	}
};

struct is_nonmaxull
{
	__host__ __device__
		bool operator()(const unsigned long long x)
	{
		return (x != 0xFFFFFFFFFFFFFFFF);
	}
};

int CountPilesCudaS(int FragsNum, int* ppixelBig)
{
	thrust::device_ptr<int> pp = thrust::device_pointer_cast(ppixelBig);
	int PilesNum = thrust::count_if(pp, pp + FragsNum, is_nonMinusOnei());

	return PilesNum;
}

void FinalizePilesCudaS(int FragsNum, float* pstartBig, float* pendBig, int* ppixelBig, float* pdepthBig, float* pstart, float* pend, int* ppixel, float* pdepth, pileMode pmode)
{
	if (pmode != resample)
	{
		thrust::device_ptr<float> psB = thrust::device_pointer_cast(pstartBig);
		thrust::device_ptr<float> ps = thrust::device_pointer_cast(pstart);
		thrust::copy_if(psB, psB + FragsNum, ps, is_nonMinusOnef());

		thrust::device_ptr<float> peB = thrust::device_pointer_cast(pendBig);
		thrust::device_ptr<float> pe = thrust::device_pointer_cast(pend);
		thrust::copy_if(peB, peB + FragsNum, pe, is_nonMinusOnef());
	}

	thrust::device_ptr<int> ppB = thrust::device_pointer_cast(ppixelBig);
	thrust::device_ptr<int> pp = thrust::device_pointer_cast(ppixel);
	thrust::copy_if(ppB, ppB + FragsNum, pp, is_nonMinusOnef());

	if (pmode == both || pmode == resample)
	{
		thrust::device_ptr<float> pdB = thrust::device_pointer_cast(pdepthBig);
		thrust::device_ptr<float> pd = thrust::device_pointer_cast(pdepth);
		thrust::copy_if(pdB, pdB + FragsNum, pd, is_nonMinusOnef());
	}
}

//use a template?
void FillAllWithValue(int* arr, int sz, int val)
{

	//thrust::device_ptr<int> d = thrust::device_pointer_cast(arr);
	//thrust::fill(d, d + sz, val);

	try
	{
		// do something crazy
		thrust::device_ptr<int> d = thrust::device_pointer_cast(arr);
		thrust::fill(d, d + sz, val);

		throw_on_cuda_error(cudaSetDevice(-1), __FILE__, __LINE__);
	}
	catch (thrust::system_error& e)
	{
		std::cerr << "CUDA error after cudaSetDevice: " << e.what() << std::endl;

		// oops, recover
		cudaSetDevice(0);
	}

}

void FillAllWithValue(float* arr, int sz, float val)
{

	thrust::device_ptr<float> d = thrust::device_pointer_cast(arr);
	thrust::fill(d, d + sz, val);

}

void FillAllWithValue(unsigned long long* arr, int sz, unsigned long long val)
{
	thrust::device_ptr<unsigned long long> d = thrust::device_pointer_cast(arr);
	thrust::fill(d, d + sz, val);
}

