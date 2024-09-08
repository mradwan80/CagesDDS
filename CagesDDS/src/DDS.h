#pragma once

#include "glew/include/glew.h"
#include "glm/glm.hpp"
#include <thrust/host_vector.h>

#include <vector>
using namespace std;

//blnd: blend depth intervals into [start,end]
//unn: take union of depth intervals into [start,end]
//both: take union and resample a depth value
//resample: sample depth only (no bounding interval)
enum pileMode {blnd, unn, both, resample};


struct PointCoords
{
	float x, y, z;

	PointCoords() {}
	PointCoords(float x, float y, float z): x(x), y(y), z(z) {}
};

class PileStruct
{
public:
	float depth;
	int pixel;
	float start, end;
};

class DDS
{

private:

	bool debug_details;
	pileMode PileMode;

	vector<PointCoords>* vxPos;
	vector<float>* vxRad;
	vector<int>vxIndex;

	float viewWidth;
	glm::mat4 vmMat, pvmMat;
	
	int FragsNum;

	float* matrixPVM;
	float* matrixVM;
	float* vpos;
	float* vrad;
	int* xfcount;
	int* xfoffset;
	int* xpcount;
	int* xpoffset;

	float* FragDepth; float* FragRad; float* FragDist; unsigned long long* FragDepthPixel;
	float* pstartBig;	float* pendBig; float* pdepthBig; int* ppixelBig;
	float* pstart;	float* pend; float* pdepth; int* ppixel;


	void PrepareInput();
	void CountFrags(bool disks);
	void CreateOffsetAndFragsVectors();
	void ProjectFrags(bool disks);
	void SortFrags();
	void Pile();
	void CountPiles();
	void FinalizePiles();
	void CopyPilesToCPU();
	void FreeMemory();

	void TestCountFrags();
	void TestCreateOffset();
	void TestProjectFrags();
	void TestSortFrags();
	void TestPile();
	void TestCountPiles();
	void TestCopyPilesToCPU();
	
public:

	int globalW, globalH;

	int PilesNum;
	vector<int> PilesCount;
	vector<int> PilesOffset;
	vector<PileStruct> Piles;


	DDS();
	DDS(int w, int h, float viewWidthI, vector<PointCoords>* Pos, vector<float>* Rad, glm::mat4 vmMatI, glm::mat4 pvmMatI);

	float BuildDDS(bool disks = true, pileMode pmode = both, bool debug_details = false);

	void DataForTesting();

};

void FillAllWithValue(int* arr, int sz, int val);
void FillAllWithValue(float* arr, int sz, float val);
void FillAllWithValue(unsigned long long* arr, int sz, unsigned long long val);
void CountFragsCudaS(int vxNum, int globalW, int globalH, float viewWidth, float* vmMat, float* pvmMat, float* vpos, float* vrad, int* xfcount, bool disks);
void SetOffsetVectorCudaS(int pxNum, int* xfcount, int* xfoffset);
int GetFragsNumCudaS(int vxNum, int* xfcount);
void ProjectFragsCudaS(int vxNum, int globalW, int globalH, float viewWidth, float* vmMat, float* pvmMat, float* vpos, float* vrad, int* xfcount, int* xfoffset, float* FragDepth, float* FragRad, float* FragDist, unsigned long long* FragDepthPixel, bool disks);
void SortFragsCudaS(int FragsNum, float* FragDepth, float* FragRad, float* FragDist, unsigned long long* FragDepthPixel);
void PileCudaS(int pxNum, int* xfcount, int* xfoffset, float* FragDepth, float* FragRad, float* FragDist, float* pstart, float* pend, int* ppixel, float* pdepth, int* xpcount, pileMode pmode);
int CountPilesCudaS(int FragsNum, int* ppixel);
void FinalizePilesCudaS(int FragsNum, float* pstartBig, float* pendBig, int* ppixelBig, float* pdepthBig, float* pstart, float* pend, int* ppixel, float* pdepth, pileMode pmode);

//use a template?
void FillAllWithValue(int* arr, int sz, int val);
void FillAllWithValue(float* arr, int sz, float val);
void FillAllWithValue(unsigned long long* arr, int sz, unsigned long long val);

#pragma once
