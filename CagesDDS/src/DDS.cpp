#include "DDS.h"
#include <chrono>
#include <iostream>
//#include "OpenGlClasses.h"
#include "glm/gtc/type_ptr.hpp"
#include<thrust/device_ptr.h>
#include<thrust/sort.h>
#include <thrust/execution_policy.h>
//#include "ClassesDefinitions.h"
using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

DDS::DDS() {}

DDS::DDS(int w, int h, float viewWidthI, vector<PointCoords>* Pos, vector<float>* Rad, glm::mat4 vmMatI, glm::mat4 pvmMatI)
{
	globalW = w;
	globalH = h;
	viewWidth = viewWidthI;

	vxPos = Pos;
	vxRad = Rad;

	vmMat = vmMatI;
	pvmMat = pvmMatI;

	//glViewport(0, 0, w, h);

	//create vxIndex//
	vxIndex.reserve(vxPos->size());
	for (int i = 0; i < vxPos->size(); i++)
		vxIndex.push_back(i);

	//SetLighting(); //no colors for now

}

void DDS::PrepareInput()
{
	float milliseconds;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	//allocate
	/*cudaMalloc((void**)&matrixVM, 16 * sizeof(float));
	cudaMalloc((void**)&matrixPVM, 16 * sizeof(float));
	cudaMalloc((void**)&vpos, vxPos->size() * 3 * sizeof(float));
	cudaMalloc((void**)&vrad, vxPos->size() * sizeof(float));
	cudaMalloc((void**)&xfcount, globalW * globalH * sizeof(int));*/

	gpuErrchk(cudaMalloc((void**)&matrixVM, 16 * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&matrixPVM, 16 * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&vpos, vxPos->size() * 3 * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&vrad, vxPos->size() * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&xfcount, globalW * globalH * sizeof(int)));
	

	//copy
	/*cudaMemcpy(matrixVM, (float*)glm::value_ptr(vmMat), 16 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(matrixPVM, (float*)glm::value_ptr(pvmMat), 16 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(vpos, vxPos->data(), vxPos->size() * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(vrad, vxRad->data(), vxRad->size() * sizeof(float), cudaMemcpyHostToDevice);*/
	gpuErrchk(cudaMemcpy(matrixVM, (float*)glm::value_ptr(vmMat), 16 * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(matrixPVM, (float*)glm::value_ptr(pvmMat), 16 * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(vpos, vxPos->data(), vxPos->size() * 3 * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(vrad, vxRad->data(), vxRad->size() * sizeof(float), cudaMemcpyHostToDevice));

	FillAllWithValue(xfcount, globalW * globalH, 0);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	if(debug_details)
		cout << "***Preparation time: " << milliseconds << '\n';


}

void DDS::CountFrags(bool disks)
{
	float milliseconds;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	CountFragsCudaS(vxPos->size(), globalW, globalH, viewWidth, matrixVM, matrixPVM, vpos, vrad, xfcount, disks);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	if (debug_details)
		cout << "***CountFrags time: " << milliseconds << '\n';


}

void DDS::TestCountFrags()
{
	int MoreThanOne = 0;

	//get vector back
	int* xfcountHost = new int[globalW * globalH];
	cudaMemcpy(xfcountHost, xfcount, globalW * globalH * sizeof(int), cudaMemcpyDeviceToHost);

	//make sure all either 1s or 0s
	int ctr = 0;
	int occPixels = 0;
	for (int i = 0; i < globalW * globalH; i++)
	{
		/*if (xfcountHost[i] == 1)
			ctr++;
		else if (xfcountHost[i] == 0)
		{

		}
		else
			cout << "incorrect value " << xfcountHost[i] << " at " << i << '\n';*/
		if (xfcountHost[i] > 0)
			ctr += xfcountHost[i];

		if (xfcountHost[i] > 1)
			MoreThanOne++;
		//if (xfcountHost[i] != 0)
		//	cout << "value in xfcountHost is: " << xfcountHost[i] << '\n';

		if (xfcountHost[i] > 0)
			occPixels++;
	}

	cout << MoreThanOne << " pixels have more than one frag (testing count)\n";

	//make sure 1s equal to vxs num
	cout << "sizes: " << vxPos->size() << ' ' << ctr << '\n'; //set splatRad to 1. then ctr should be 3 times the vxpos size//
	cout << "occupied pixels: " << occPixels << " \n";


}

void DDS::CreateOffsetAndFragsVectors()
{
	float milliseconds;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);


	//allocate
	cudaMalloc((void**)&xfoffset, globalW * globalH * sizeof(int));

	//set values
	SetOffsetVectorCudaS(globalW * globalH, xfcount, xfoffset);

	FragsNum = GetFragsNumCudaS(globalW * globalH, xfcount);
	if (debug_details)
		cout << "fragsnum: " << FragsNum << '\n';

	cudaMalloc((void**)&FragDepth, FragsNum * sizeof(float));
	cudaMalloc((void**)&FragRad, FragsNum * sizeof(float));
	cudaMalloc((void**)&FragDist, FragsNum * sizeof(float));
	cudaMalloc((void**)&FragDepthPixel, FragsNum * sizeof(unsigned long long));

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	if (debug_details)
		cout << "***Create Offset and Frags vecs time: " << milliseconds << '\n';

}

void DDS::TestCreateOffset()
{
	int* xfcountHost = new int[globalW * globalH];
	cudaMemcpy(xfcountHost, xfcount, globalW * globalH * sizeof(int), cudaMemcpyDeviceToHost);
	int* xfoffsetHost = new int[globalW * globalH];
	cudaMemcpy(xfoffsetHost, xfoffset, globalW * globalH * sizeof(int), cudaMemcpyDeviceToHost);

	//int ctr = xfcountHost[0];
	int ctr = 0;
	for (int i = 1; i < globalW * globalH; i++)
	{
		ctr += xfcountHost[i - 1];
		if (xfoffsetHost[i] != ctr)
			cout << "problem at " << i << '\n';
	}

	//to make one frag, make loops from start to start, not from start to end//
	cout << "offset test: " << vxPos->size() << ' ' << ctr << '\n'; //equal when one frag per vertex//


}

void DDS::ProjectFrags(bool disks)
{
	float milliseconds;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	//zero count again
	FillAllWithValue(xfcount, globalW * globalH, 0);

	//project using vbo data, count and offset
	ProjectFragsCudaS(vxPos->size(), globalW, globalH, viewWidth, matrixVM, matrixPVM, vpos, vrad, xfcount, xfoffset, FragDepth, FragRad, FragDist, FragDepthPixel, disks);

	cudaFree(vpos);
	cudaFree(vrad);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	if (debug_details)
		cout << "***ProjectFrags time: " << milliseconds << '\n';

}

void DDS::TestProjectFrags() //may be not the best test
{
	int newSum = GetFragsNumCudaS(globalW * globalH, xfcount);
	cout << "fragsnum: " << newSum << ' ' << FragsNum << '\n';
}

void DDS::SortFrags()
{

	float milliseconds;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	SortFragsCudaS(FragsNum, FragDepth, FragRad, FragDist, FragDepthPixel);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	if (debug_details)
		cout << "***SortFrags time: " << milliseconds << '\n';

}

void DDS::TestSortFrags()
{

	/////////////
	//check (patch,pixel) are ordered. check depths are ordered//
	/////////////

	float* fragdepthHost = new float[FragsNum];
	cudaMemcpy(fragdepthHost, FragDepth, FragsNum * sizeof(float), cudaMemcpyDeviceToHost);
	unsigned long long* fragdepthpixelHost = new unsigned long long[FragsNum];
	cudaMemcpy(fragdepthpixelHost, FragDepthPixel, FragsNum * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 50; i++)
	{
		cout << fragdepthHost[i] << " " << fragdepthpixelHost[i] << '\n';
	}


	bool pxlproblem = false;
	bool dstproblem = false;
	float prevf = fragdepthHost[0];
	unsigned long long prevull = fragdepthpixelHost[0];
	for (int i = 1; i < FragsNum; i++)
	{
		float f = fragdepthHost[i];
		unsigned long long ull = fragdepthpixelHost[i];
		if (ull != prevull)
		{
			if (ull < prevull)
			{
				cout << "problem in the pixel order at " << i - 1 << " and " << i << '\n';
				pxlproblem = true;
			}
		}
		else
		{
			if (f < prevf)
			{
				cout << "problem in the depth order at " << i - 1 << " and " << i << '\n';
				cout << "values are " << prevull << ' ' << ull << ' ' << prevf << ' ' << f << '\n';
				dstproblem = true;
			}
		}

		prevull = ull;
		prevf = f;
	}
	if (!pxlproblem)
		cout << "no problems found in the sorted pixel list\n";
	if (!dstproblem)
		cout << "no problems found in the sorted depth list\n";

	
	/////////////
	//check all pixels with frags, exist in FragPatchPixel after sort//
	/////////////

	//get old count//
	int* xfcountHost = new int[globalW * globalH];
	cudaMemcpy(xfcountHost, xfcount, globalW * globalH * sizeof(int), cudaMemcpyDeviceToHost);



	vector<bool> pixelFilled(globalW * globalH, false);
	for (int i = 0; i < globalW * globalH; i++)
	{
		if (xfcountHost[i] > 0)
			pixelFilled[i] = true;
	}

	vector<bool> pixelFilled2(globalW * globalH, false);
	for (int i = 0; i < FragsNum; i++)
	{
		unsigned long long patchpixel = fragdepthpixelHost[i];
		patchpixel = patchpixel >> 32;
		int pxl = patchpixel; //get it
		pixelFilled2[pxl] = true;
	}

	bool pbmFound = false;
	for (int i = 0; i < globalW * globalH; i++)
	{
		if ((pixelFilled[i] && !pixelFilled2[i]) || (!pixelFilled[i] && pixelFilled2[i]))
			cout << "problem at pixel " << i << '\n';
		if ((pixelFilled[i] && !pixelFilled2[i]) || (!pixelFilled[i] && pixelFilled2[i]))
			pbmFound = true;
	}
	if (!pbmFound)
		cout << "pixels with frags are same frags saved in depthpixel\n";

	int ccc1 = 0, ccc2 = 0;
	for (int i = 0; i < globalW * globalH; i++)
	{
		if (pixelFilled[i])
			ccc1++;
		if (pixelFilled2[i])
			ccc2++;
	}
	cout << "pixels filled: " << ccc1 << ' ' << ccc2 << '\n';

	/////////////
	//check all pixels with in fragdepthpixel, are equivalent to pixel of subarray//
	/////////////

	int* xfoffsetHost = new int[globalW * globalH];
	cudaMemcpy(xfoffsetHost, xfoffset, globalW * globalH * sizeof(int), cudaMemcpyDeviceToHost);

	pbmFound = false;
	for (int i = 0; i < globalW * globalH; i++)
	{
		bool pxlPbm = false;
		for (int j = xfoffsetHost[i]; j < xfoffsetHost[i] + xfcountHost[i]; j++)
		{
			unsigned long long patchpixel = fragdepthpixelHost[j];
			patchpixel = patchpixel >> 32;
			int pxl = patchpixel; //get it
			if (pxl != i)
				pxlPbm = true;
				
		}
		if (pxlPbm)
		{
			pbmFound = true;
			cout << "problem at pixel " << i << ", pixels: ";
			for (int j = xfoffsetHost[i]; j < xfoffsetHost[i] + xfcountHost[i]; j++)
			{
				unsigned long long patchpixel = fragdepthpixelHost[j];
				patchpixel = patchpixel >> 32;
				int pxl = patchpixel; //get it
				cout << pxl << ' ';
			}
			cout << '\n';
		}

	}
	if (!pbmFound)
		cout << "All pixels saved in fragdepthpixelHost are equivalent to pixel of the subarray\n";


}


void DDS::Pile()
{
	float milliseconds;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	if (PileMode != resample) cudaMalloc((void**)&pstartBig, FragsNum * sizeof(float));
	if (PileMode != resample) cudaMalloc((void**)&pendBig, FragsNum * sizeof(float));
	cudaMalloc((void**)&ppixelBig, FragsNum * sizeof(int));
	if (PileMode == both || PileMode == resample) cudaMalloc((void**)&pdepthBig, FragsNum * sizeof(float));

	if (PileMode != resample) FillAllWithValue(pstartBig, FragsNum, -1.0);
	if (PileMode != resample) FillAllWithValue(pendBig, FragsNum, -1.0);
	FillAllWithValue(ppixelBig, FragsNum, -1);
	if (PileMode == both || PileMode == resample) FillAllWithValue(pdepthBig, FragsNum, -1);

	cudaMalloc((void**)&xpcount, globalW * globalH * sizeof(int));
	FillAllWithValue(xpcount, globalW*globalH, 0);

	PileCudaS(globalW * globalH, xfcount, xfoffset, FragDepth, FragRad, FragDist, pstartBig, pendBig, ppixelBig, pdepthBig, xpcount, PileMode);

	cudaFree(FragDepth);
	cudaFree(FragRad);
	cudaFree(FragDist);
	cudaFree(FragDepthPixel);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	if (debug_details)
		cout << "***Pile time: " << milliseconds << '\n';


}

void DDS::TestPile()
{
	////////
	//check output in ppixel is correct (all pixels with frags, exist in ppixel)//
	////////

	//get old count//
	int* xfcountHost = new int[globalW * globalH];
	cudaMemcpy(xfcountHost, xfcount, globalW * globalH * sizeof(int), cudaMemcpyDeviceToHost);
	int* xfoffsetHost = new int[globalW * globalH];
	cudaMemcpy(xfoffsetHost, xfoffset, globalW * globalH * sizeof(int), cudaMemcpyDeviceToHost);
	int* ppixelHost = new int[FragsNum];
	cudaMemcpy(ppixelHost, ppixelBig, FragsNum * sizeof(int), cudaMemcpyDeviceToHost);

	vector<bool> pixelFilled(globalW * globalH, false);
	for (int i = 0; i < globalW * globalH; i++)
	{
		if (xfcountHost[i] > 0)
			pixelFilled[i] = true;
	}

	vector<bool> pixelFilled2(globalW * globalH, false);
	for (int i = 0; i < FragsNum; i++)
	{
		int pxl = ppixelHost[i];
		//if (pxl<0 || pxl>globalW * globalH - 1)
		//	cout << "Test Pile: we have a problem. a pixel of " << pxl <<'\n'; //incorrect test. many memory locations will contain -1s

		if (pxl >= 0 && pxl < globalW * globalH)
			pixelFilled2[pxl] = true;
	}

	bool pbmFound = false;
	for (int i = 0; i < globalW * globalH; i++)
	{
		if ((pixelFilled[i] && !pixelFilled2[i]) || (!pixelFilled[i] && pixelFilled2[i]))
		{
			pbmFound = true;

			cout << "problem at pixel " << i << '\n';

			//use next block instead of previous cout, to find out more about pixels of piles at each pixel//
			/*
			cout << "problem at pixel " << i << ", pixels: ";
			for (int j = xfoffsetHost[i]; j < xfoffsetHost[i] + xfcountHost[i]; j++)
			{
				int pxl = ppixelHost[j];
				cout << pxl << ' ';
			}
			cout << '\n';
			*/
			
		}
	}
	if (!pbmFound)
		cout << "all pixels with frags have piles\n";

	int ccc1 = 0, ccc2 = 0;
	for (int i = 0; i < globalW * globalH; i++)
	{
		if (pixelFilled[i])
			ccc1++;
		if (pixelFilled2[i])
			ccc2++;
	}
	cout << "pixels filled (with frags, with piles): " << ccc1 << ' ' << ccc2 << '\n';

	if (PileMode != resample)
	{


		////////
		//check start is always less than end//
		////////

		float* pstartHost = new float[FragsNum];
		cudaMemcpy(pstartHost, pstartBig, FragsNum * sizeof(float), cudaMemcpyDeviceToHost);
		float* pendHost = new float[FragsNum];
		cudaMemcpy(pendHost, pendBig, FragsNum * sizeof(float), cudaMemcpyDeviceToHost);

		pbmFound = false;
		for (int i = 0; i < FragsNum; i++)
		{
			float start = pstartHost[i];
			float end = pendHost[i];

			if (start == -1.0 && end == -1.0)
				continue;
			else if (start > end)
			{
				cout << "start is bigger than end at index " << i << '\n';
				pbmFound = true;
			}
		}
		if (!pbmFound)
			cout << "starts are less than ends in all piles\n";

		////////
		//check no start comes after its successor//
		////////

		pbmFound = false;
		for (int i = 0; i < globalW * globalH; i++)
		{
			if (xfcountHost[i] < 2)
				continue;

			float prevstart = pstartHost[xfoffsetHost[i]];
			for (int j = xfoffsetHost[i] + 1; j < xfoffsetHost[i] + xfcountHost[i]; j++)
			{
				float start = pstartHost[j];
				if (start == -1.0)
					continue;
				if (start < prevstart)
				{
					pbmFound = true;
					cout << "problem at pixel " << i << ", a pile starts after its successor\n";
				}
				prevstart = start;
			}

		}
		if (!pbmFound)
			cout << "starts are sorted in all pixels\n";

		////////
		//check depth values lie between start and end values//
		////////
		if (PileMode == both)
		{
			float* pdepthHost = new float[FragsNum];
			cudaMemcpy(pdepthHost, pdepthBig, FragsNum * sizeof(float), cudaMemcpyDeviceToHost);

			for (int i = 0; i < FragsNum; i++)
			{
				float start = pstartHost[i];
				float end = pendHost[i];
				float depth = pdepthHost[i];

				if (start == -1.0 && end == -1.0)
					continue;
				else if (depth < start)
				{
					cout << "start is smaller than start at index " << i << '\n';
					pbmFound = true;
				}
				else if (depth > end)
				{
					cout << "depth is bigger than end at index " << i << '\n';
					pbmFound = true;
				}
			}
			if (!pbmFound)
				cout << "depths are between starts and ends in all piles\n";

		}
	}
}

void DDS::CountPiles()
{
	float milliseconds;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);


	PilesNum = CountPilesCudaS(FragsNum, ppixelBig);

	if (debug_details)
		cout << "Piles number: " << PilesNum << '\n';


	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	if (debug_details)
		cout << "***CountPiles time: " << milliseconds << '\n';


}

void DDS::TestCountPiles()
{
	if (PileMode == resample)
		return;

	float* pstartHost = new float[FragsNum];
	cudaMemcpy(pstartHost, pstartBig, FragsNum * sizeof(float), cudaMemcpyDeviceToHost);
	float* pendHost = new float[FragsNum];
	cudaMemcpy(pendHost, pendBig, FragsNum * sizeof(float), cudaMemcpyDeviceToHost);
	int* ppixelHost = new int[FragsNum];
	cudaMemcpy(ppixelHost, ppixelBig, FragsNum * sizeof(int), cudaMemcpyDeviceToHost);


	bool pbmFound = false;
	int index = 0;
	for (int i = 0; i < FragsNum; i++)
	{
		if (pstartHost[i] == -1.0)
		{
			if (pendHost[i] != -1.0)
			{
				cout << "pend should be -1.0 at frag " << i << '\n';
				pbmFound = true;
			}

			if (ppixelHost[i] != -1)
			{
				cout << "ppixel should be -1 at frag " << i << '\n';
				pbmFound = true;
			}
		}
		else
		{
			if (pendHost[i] == -1.0)
			{
				cout << "pend is incorrectly -1.0 at frag " << i << '\n';
				pbmFound = true;
			}

			if (ppixelHost[i] == -1)
			{
				cout << "ppixel is incorrectly -1 at frag " << i << '\n';
				pbmFound = true;
			}
			index++;
		}
	}

	if (!pbmFound)
		cout << "no problems in piles arrays\n";
	cout << "piles num (in test fnc): " << PilesNum << ' ' << index << '\n';


}

void DDS::FinalizePiles()
{
	float milliseconds;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);


	//count should be ready//

	//create offset//
	cudaMalloc((void**)&xpoffset, globalW * globalH * sizeof(int));
	FillAllWithValue(xpoffset, globalW * globalH, 0);
	SetOffsetVectorCudaS(globalW * globalH, xpcount, xpoffset);

	
	//remove -1s

	if (PileMode != resample) cudaMalloc((void**)&pstart, PilesNum * sizeof(float));
	if (PileMode != resample) cudaMalloc((void**)&pend, PilesNum * sizeof(float));
	cudaMalloc((void**)&ppixel, PilesNum * sizeof(int));
	if (PileMode == both|| PileMode == resample) cudaMalloc((void**)&pdepth, PilesNum * sizeof(float));

	FinalizePilesCudaS(FragsNum, pstartBig, pendBig, ppixelBig, pdepthBig ,pstart, pend, ppixel, pdepth, PileMode);

	if (PileMode != resample) cudaFree(pstartBig);
	if (PileMode != resample) cudaFree(pendBig);
	cudaFree(ppixelBig);
	if (PileMode == both || PileMode == resample) cudaFree(pdepthBig);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	if (debug_details)
		cout << "***finalize time: " << milliseconds << '\n';

}

void DDS::CopyPilesToCPU()
{

	float milliseconds;
	cudaEvent_t starte, stop;
	cudaEventCreate(&starte);
	cudaEventCreate(&stop);

	cudaEventRecord(starte);


	PilesCount.resize(globalW * globalH, 0);
	PilesOffset.resize(globalW * globalH);
	Piles.resize(PilesNum);
	
	int* ppixelHost = new int[PilesNum];
	cudaMemcpy(ppixelHost, ppixel, PilesNum * sizeof(int), cudaMemcpyDeviceToHost);
	int pixel;
	for (int i = 0; i < PilesNum; i++)
	{
		pixel = ppixelHost[i];

		PilesCount[pixel]++;
		Piles[i].pixel = pixel;
	}
	delete[] ppixelHost;

	
	if (PileMode != resample)
	{
		float* pstartHost = new float[PilesNum];
		cudaMemcpy(pstartHost, pstart, PilesNum * sizeof(float), cudaMemcpyDeviceToHost);
		float* pendHost = new float[PilesNum];
		cudaMemcpy(pendHost, pend, PilesNum * sizeof(float), cudaMemcpyDeviceToHost);

		for (int i = 0; i < PilesNum; i++)
		{
			Piles[i].start = pstartHost[i];
			Piles[i].end = pendHost[i];
		}

		delete[] pstartHost;
		delete[] pendHost;

	}

	if (PileMode == both || PileMode == resample)
	{
		float* pdepthHost = new float[PilesNum];
		cudaMemcpy(pdepthHost, pdepth, PilesNum * sizeof(float), cudaMemcpyDeviceToHost);

		for (int i = 0; i < PilesNum; i++)
			Piles[i].depth = pdepthHost[i];
		delete[] pdepthHost;
	}

	PilesOffset[0] = 0;
	for (int i = 1; i < globalW * globalH; i++)
		PilesOffset[i] = PilesCount[i - 1] + PilesOffset[i - 1];


	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, starte, stop);
	if (debug_details)
		cout << "***FinalizePiles: " << milliseconds << '\n';


}

void DDS::TestCopyPilesToCPU()
{
	
	int* xpcountHost = new int[globalW * globalH];
	cudaMemcpy(xpcountHost, xpcount, globalW * globalH * sizeof(int), cudaMemcpyDeviceToHost);
	int* xpoffsetHost = new int[globalW * globalH];
	cudaMemcpy(xpoffsetHost, xpoffset, globalW * globalH * sizeof(int), cudaMemcpyDeviceToHost);

	int problemCount = 0;
	bool problemFound = false;
	for (int i = 0; i < globalW * globalH; i++)
	{
		if (PilesCount[i] != xpcountHost[i])
			problemFound = true;
		if (PilesCount[i] != xpcountHost[i])
			problemCount++;
	}
	if (!problemFound)
		cout << "no problem found in the xpcount\n";
	else
		cout << "problems in xpcount " << problemCount << '\n';

	problemCount = 0;
	problemFound = false;
	for (int i = 1; i < globalW * globalH; i++)
	{
		if (PilesCount[i] != 0)
		{
			if (PilesOffset[i] != xpoffsetHost[i])
				problemFound = true;
			if (PilesOffset[i] != xpoffsetHost[i])
				problemCount++;
		}
	}
	if (!problemFound)
		cout << "no problem found in the xpoffset\n";
	else
		cout << "problems in xpoffset " << problemCount << '\n';


}

void DDS::FreeMemory()
{
	float milliseconds;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	cudaFree(matrixVM);
	cudaFree(matrixPVM);

	cudaFree(xfcount);
	cudaFree(xfoffset);

	if (PileMode != resample) cudaFree(pstart);
	if (PileMode != resample) cudaFree(pend);
	cudaFree(ppixel);
	if (PileMode == both || PileMode == resample) cudaFree(pdepth);


	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	if (debug_details)
		cout << "***FreeMemory time: " << milliseconds << '\n';

}

float DDS::BuildDDS(bool disks, pileMode pmode, bool dd)
{
	debug_details = dd;
	PileMode = pmode;

	float milliseconds;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	PrepareInput();
	CountFrags(disks);
	//TestCountFrags();
	CreateOffsetAndFragsVectors();
	//TestCreateOffset();
	ProjectFrags(disks);
	//TestProjectFrags();
	SortFrags();
	//TestSortFrags();
	Pile();
	//TestPile();
	CountPiles();
	//TestCountPiles();
	FinalizePiles();
	
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << "***total DDS time: " << milliseconds << '\n';

	cout << "---------\n";

	CopyPilesToCPU();
	//TestCopyPilesToCPU();
	FreeMemory();

	return milliseconds;
}


