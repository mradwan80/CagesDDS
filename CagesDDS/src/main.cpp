#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <stack>
#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"
#include "glm/gtc/type_ptr.hpp"
//#include "AugDDS.h"
#include <set>
#include "DDS.h"
using namespace std;

struct Pixel
{
	int x, y;
	int PilesOffset;
	int PilesCount;
	
	Pixel() {} //why necessary?
	Pixel(int x, int y):x(x),y(y){}

	void SetCoordinates(int xx, int yy)
	{
		x = xx;
		y = yy;
	}

	int ComputeIndex(int w, int h)
	{
		return y * w + x;
	}

	bool ContainsPiles()
	{
		return PilesCount > 0;
	}

	void GetNbsWithBiggerIndex(vector<Pixel>& pixels, int w, int h, vector<int>& nbpixels, bool OnlyValid = true)
	{
		vector<Pixel> nbs{ Pixel(x + 1, y), Pixel(x - 1,y + 1),Pixel(x,y + 1),Pixel(x + 1,y + 1) };

		for (auto nb : nbs)
		{
			int nbIndex = nb.ComputeIndex(w, h);
			if (Pixel::CoordsWithinRange(nb.x, nb.y, w, h) && pixels[nbIndex].ContainsPiles())
				nbpixels.push_back(nbIndex);
			else if (!OnlyValid)
				nbpixels.push_back(-1);

		}
	}

	void GetDirectNbs(vector<Pixel>& pixels, int w, int h, vector<int>& nbpixels, bool OnlyValid = true)
	{
		vector<Pixel> nbs{ Pixel(x , y - 1), Pixel(x + 1,y),Pixel(x,y + 1),Pixel(x - 1,y) };

		for (auto nb : nbs)
		{
			int nbIndex = nb.ComputeIndex(w, h);
			if (Pixel::CoordsWithinRange(nb.x, nb.y, w, h) && pixels[nbIndex].ContainsPiles())
				nbpixels.push_back(nbIndex);
			else if(!OnlyValid)
				nbpixels.push_back(nbIndex);
				
		}

	}

	void GetCrossNbs(vector<Pixel>& pixels, int w, int h, vector<int>& nbpixels, bool OnlyValid = true)
	{
		vector<Pixel> nbs{ Pixel(x - 1 , y - 1), Pixel(x + 1,y - 1),Pixel(x + 1,y + 1),Pixel(x - 1,y + 1) };

		for (auto nb : nbs)
		{
			int nbIndex = nb.ComputeIndex(w, h);
			if (Pixel::CoordsWithinRange(nb.x, nb.y, w, h) && pixels[nbIndex].ContainsPiles())
				nbpixels.push_back(nbIndex);
			else if (!OnlyValid)
				nbpixels.push_back(nbIndex);
		}


	}

	//the edge of pixel1 crossing pixel2
	static int GetCrossEdge(const Pixel& pixel1, const Pixel& pixel2)
	{
		if ((pixel2.x - pixel1.x == -1) && (pixel2.y - pixel1.y == -1))
			return 0;
		else if ((pixel2.x - pixel1.x == 1) && (pixel2.y - pixel1.y == -1))
			return 1;
		else if ((pixel2.x - pixel1.x == 1) && (pixel2.y - pixel1.y == 1))
			return 2;
		else if ((pixel2.x - pixel1.x == -1) && (pixel2.y - pixel1.y == 1))
			return 3;
		else
			return -1;

	}

	//the starting vertex of the edge of pixel 2 crossing pixel1
	static int GetCrossVertex(const Pixel& pixel1, const Pixel& pixel2)
	{
		if ((pixel2.x - pixel1.x == -1) && (pixel2.y - pixel1.y == -1))
			return 2;
		else if ((pixel2.x - pixel1.x == 1) && (pixel2.y - pixel1.y == -1))
			return 3;
		else if ((pixel2.x - pixel1.x == 1) && (pixel2.y - pixel1.y == 1))
			return 0;
		else if ((pixel2.x - pixel1.x == -1) && (pixel2.y - pixel1.y == 1))
			return 1;
		else
			return -1;
	}

	//the two vertices of the face of pixel 2 that faces pixel1
	static void GetFacingVertices(const Pixel& pixel1, const Pixel& pixel2, int& nbv0, int& nbv1)
	{
		if ((pixel2.x - pixel1.x == 0) && (pixel2.y - pixel1.y == -1))
		{
			nbv0 = 3; nbv1 = 2;
		}
		else if ((pixel2.x - pixel1.x == 1) && (pixel2.y - pixel1.y == 0))
		{
			nbv0 = 0; nbv1 = 3;
		}
		else if ((pixel2.x - pixel1.x == 0) && (pixel2.y - pixel1.y == 1))
		{
			nbv0 = 1; nbv1 = 0;
		}
		else if ((pixel2.x - pixel1.x == -1) && (pixel2.y - pixel1.y == 0))
		{
			nbv0 = 2; nbv1 = 1;
		}
		else
		{
			nbv0 = -1; nbv1 = -1;
		}
	}

	//get the index of the face of pixel1, that faces pixel2
	static int GetFacingFace(const Pixel& pixel1, const Pixel& pixel2)
	{

		if ((pixel2.x - pixel1.x == 0) && (pixel2.y - pixel1.y == -1))
			return 0;
		else if ((pixel2.x - pixel1.x == 1) && (pixel2.y - pixel1.y == 0))
			return 1;
		else if ((pixel2.x - pixel1.x == 0) && (pixel2.y - pixel1.y == 1))
			return 2;
		else if ((pixel2.x - pixel1.x == -1) && (pixel2.y - pixel1.y == 0))
			return 3;
		else
			return -1;
	}

	static bool CoordsWithinRange(int x, int y, int w, int h)
	{
		return (x >= 0 && x < w&& y >= 0 && y < h);
	}

};

struct PointCoordsExt : PointCoords
{
	PointCoordsExt(float x, float y, float z) { PointCoords(x, y, z); }

	int CheckAgainstPilePoints(vector<PointCoordsExt>::iterator begin)
	{
		vector<PointCoordsExt>::iterator end = begin + 8;
		for(vector<PointCoordsExt>::iterator it=begin; it!=end; it++)
		{
			if (x == it->x && y == it->y && z == it->z)
				return it-begin;
		}
		return -1;
	}

};

struct Edge
{
	vector<int>epoints;
	void RemoveDuplicates()
	{
		//remove duplicates
		auto end0 = epoints.end();
		for (auto it = epoints.begin(); it != end0; ++it) {
			end0 = std::remove(it + 1, end0, *it);
		}
	}

	void SortByZ(const vector<PointCoordsExt>& points)
	{
		for (int i = 0; i < epoints.size(); i++)
		{
			int minind = -1;
			float min = points[epoints[i]].z;
			for (int j = i + 1; j < epoints.size(); j++)
			{
				if (points[epoints[j]].z > min)
				{
					min = points[epoints[j]].z;
					minind = j;
				}
			}

			if (minind != -1)
			{
				int tmp = epoints[minind];
				epoints[minind] = epoints[i];
				epoints[i] = tmp;
			}

		}

	}
};

struct PileEdges
{
	vector<set<int>> edges;
	PileEdges() 
	{
		edges.resize(4);
	}
};

struct Face
{
	int v0, v1, v2, v3;
	Edge e0, e1;

	void FillVerticesFromEdgesToEdges(set<int>& E0, set<int>& E1, vector<PointCoordsExt>& points)
	{
		//loop over vertices of E0. assign them to e0
		for (int v : E0)
		{
			if (points[v0].z <= points[v].z && points[v].z <= points[v2].z)
				e0.epoints.push_back(v);

		}
		//loop over vertices of e1. assign them to e1 of faces//
		for (int v : E1)
		{
			if (points[v0].z <= points[v].z && points[v].z <= points[v2].z)
				e1.epoints.push_back(v);
		}
	}
};

struct Triangle
{
	int v0, v1, v2;
	
	static void Triangulate(const Face& fc, vector<Triangle>& triangles)
	{

		//start with base0 and base1, then advance them. could happen that one reaches end before the other, which is handled too.

		int e0size = fc.e0.epoints.size();
		int e1size = fc.e1.epoints.size();

		bool BothEnded = false;
		int base0 = 0;
		int base1 = 0;
		int ptr0, ptr1;
		while (!BothEnded)
		{

			if (base0 < e0size - 1)
				ptr0 = base0 + 1;
			if (base1 < e1size - 1)
				ptr1 = base1 + 1;

			//triangulate with base0,base1,ptr0,ptr1//
			if (base1 != ptr1)
			{
				Triangle t{ fc.e0.epoints[base0] , fc.e1.epoints[ptr1], fc.e1.epoints[base1] };
				triangles.push_back(t);
			}
			if (base0 != ptr0)
			{
				Triangle t{ fc.e0.epoints[base0], fc.e0.epoints[ptr0], fc.e1.epoints[ptr1] };
				triangles.push_back(t);
			}

			base0 = ptr0;
			base1 = ptr1;

			if (base0 == e0size - 1 && base1 == e1size - 1)
				BothEnded = true;
		}

	}
};

struct TrigEdge
{
	int v0, v1, v2;

	bool operator < (const TrigEdge& str) const
	{
		return ((v0 < str.v0) || (v0 == str.v0 && v1 < str.v1));
	}
};

//specific for ortho projection//
void MeshPiles(string modelpath, string modelname, vector<PileStruct>& Piles, vector<Pixel>& pixels, const int w, const int h, float left, float right, float bottom, float top, float near, float far);
void VisualizePiles(string modelpath, string modelname, vector<PileStruct>& Piles, vector<int>& PilesCount, vector<int>& PilesOffset, const int w, const int h, float left, float right, float bottom, float top, float near, float far);


void main()
{
	//int GlobalW = 1024;
	//int GlobalW = 512;
	//int GlobalW = 256;
	int GlobalW = 128;
	//int GlobalW = 64;
	//int GlobalW = 32;
	//int GlobalW = 16;
	//int GlobalW = 8;
	//int GlobalW = 4;
	//int GlobalW = 2;
	int GlobalH = GlobalW;

	//int GlobalW = 1600;
	//int GlobalH = 1200;
	//int GlobalW = 800;
	//int GlobalH = 600;


	int pnum;
	vector<PointCoords>coords;
	vector<float>rads;

	/////////////
	//read file//
	/////////////

	float model_rad;

	string model_path = "D://work//data//OFF//";
	string model_name = "bunny2"; model_rad = 0.002;
	//string model_name = "armadillo"; model_rad = 0.5;
	//string model_name = "ant"; model_rad = 0.015;
	//string model_name = "bird"; model_rad = 0.013;
	//string model_name = "human"; model_rad = 0.013;
	//string model_name = "elephant"; model_rad = 0.01;
	//string model_name = "dragon3"; model_rad = 0.002;
	//string model_name = "glass"; model_rad = 0.013;
	//string model_name = "cup"; model_rad = 0.015;
	ifstream input_file(model_path + model_name + ".OFF");

	
	int tnum;
	int dummyi;
	char ch;
	input_file >> ch >> ch >> ch;
	input_file >> pnum >> tnum >> dummyi;
	coords.resize(pnum); rads.resize(pnum);
	vector<int>connectivity(3 * tnum);
	for (int i = 0; i < pnum; i++)
	{
		input_file >> coords[i].x >> coords[i].y >> coords[i].z;
		rads[i] = model_rad;
	}
	for (int i = 0; i < tnum; i++)
	{
		input_file >> dummyi >> connectivity[3*i] >> connectivity[3 * i + 1] >> connectivity[3 * i + 2];
	}
	input_file.close();

	///////////////
	//set matrices//
	///////////////

	float minx = std::numeric_limits<float>::max();
	float miny = std::numeric_limits<float>::max();
	float minz = std::numeric_limits<float>::max();
	float maxx = std::numeric_limits<float>::lowest();
	float maxy = std::numeric_limits<float>::lowest();
	float maxz = std::numeric_limits<float>::lowest();
	for (int i = 0; i < pnum; i++)
	{
		if (minx > coords[i].x)
			minx = coords[i].x;
		if (miny > coords[i].y)
			miny = coords[i].y;
		if (minz > coords[i].z)
			minz = coords[i].z;

		if (maxx < coords[i].x)
			maxx = coords[i].x;
		if (maxy < coords[i].y)
			maxy = coords[i].y;
		if (maxz < coords[i].z)
			maxz = coords[i].z;
	}
	float midx = (minx + maxx) / 2;
	float midy = (miny + maxy) / 2;
	float midz = (minz + maxz) / 2;

	float Near, Far;
	float fov = 20;
	Near = 0.0;
	Far = +3 * (maxz - minz);
	float ViewWidth = 1.02 * (maxx - minx);

	glm::mat4 ProjectionMat = glm::perspective(fov, float(GlobalW) / float(GlobalH), Near, Far);
	glm::mat4 Tcenter = glm::translate(glm::vec3(-midx, -midy, -midz));
	glm::mat4 R = glm::translate(glm::vec3(0, 0, 0));
	glm::mat4 Tmodel = glm::translate(glm::vec3(0.0, 0.0, -2.0 * (maxz - minz) / 1));
	glm::mat4 ModelMat = Tmodel * R * Tcenter;
	glm::mat4 ViewMat = glm::translate(glm::vec3(0, 0, 0));
	glm::mat4 vmMat = ViewMat * ModelMat;
	glm::mat4 pvmMat = ProjectionMat * ViewMat * ModelMat;


	//special settings, for building the DDS ON the mesh !!!//
	ViewMat = glm::translate(glm::vec3(0, 0, 0));
	vmMat = ViewMat; //no model matrix//
	float left, right, bottom, top;
	/*left = -0.51 * (maxx - minx);
	right = +0.51 * (maxx - minx);
	bottom = -0.51 * (maxy - miny);
	top = +0.51 * (maxy - miny);*/
	/*if ((maxx - minx) > (maxy - miny))
	{
		left = minx - 0.01 * (maxx - minx);
		right = maxx + 0.01 * (maxx - minx);
		bottom = miny - 0.01 * (maxx - minx);
		top = maxy + 0.01 * (maxx - minx);
	}
	else
	{
		left = minx - 0.01 * (maxy - miny);
		right = maxx + 0.01 * (maxy - miny);
		bottom = miny - 0.01 * (maxy - miny);
		top = maxy + 0.01 * (maxy - miny);
	}*/
	if ((maxx - minx) > (maxy - miny))
	{
		left = midx - 0.51 * (maxx - minx);
		right = midx + 0.51 * (maxx - minx);
		bottom = midy - 0.51 * (maxx - minx);
		top = midy + 0.51 * (maxx - minx);
	}
	else
	{
		left = midx - 0.51 * (maxy - miny);
		right = midx + 0.51 * (maxy - miny);
		bottom = midy - 0.51 * (maxy - miny);
		top = midy + 0.51 * (maxy - miny);
	}
	Near = minz;
	Far = maxz;
	glm::mat4 ProjectionOrthoMat = glm::ortho(left, right, bottom, top, Near, Far);
	glm::mat4 pvmOrthoMat = ProjectionOrthoMat ;

	///////////////
	//compute dds//
	///////////////
	
	
	//AugDDS* augdds = new AugDDS(AVG, true, GlobalW, GlobalH, ViewWidth, &coords, &rads, &oid, ProjectionMat, vmMat, pvmMat, pvmMat);
	//augdds->BuildAugDDS();

	DDS* dds = new DDS(GlobalW, GlobalH, ViewWidth, &coords, &rads, vmMat, pvmOrthoMat);
	//dds->BuildDDS();
	dds->BuildDDS(true, blnd);
	cout << "dds finished\n";
	//dds->DataForTesting();

	vector<Pixel> pixels(GlobalW * GlobalH);
	for (int px = 0; px < GlobalW; px++)
	{
		for (int py = 0; py < GlobalH; py++)
		{
			int pixelIndex = py * GlobalW + px;
			pixels[pixelIndex].SetCoordinates(px, py);
			pixels[pixelIndex].PilesCount = dds->PilesCount[pixelIndex];
			pixels[pixelIndex].PilesOffset = dds->PilesOffset[pixelIndex];
		}
	}
	for (int pxlIndex = 0; pxlIndex < pixels.size(); pxlIndex++)
	{
		
	}



	///////////////////
	//create a mesh//
	///////////////////
	MeshPiles(model_path, model_name, dds->Piles, pixels, GlobalW, GlobalH, left, right, bottom, top, Near, Far);
	cout << "meshing finished\n";

	///////////////////
	//visualize piles//
	///////////////////
	VisualizePiles(model_path, model_name, dds->Piles, dds->PilesCount, dds->PilesOffset, GlobalW, GlobalH, left, right, bottom, top, Near, Far);


	//meshing logic. 
	//for each pile, make 4 trigs (2 faces)
	//for upper face, check the 4 neighbors
	//if a (first) pile intersects, but our upper is above, then connect with 2 trigs
	//if a (first) pile intersects, but our upper is covered, ignore this face
	//stop when a first pile is found
	//if pile does not interect anything, connect with lower !
	//for lower face, check the 4 neighbors (from down to up!)
	//if a (first) pile intersects, but our lower is below, then connect with 2 trigs
	//if a (first) pile intersects, but our lower is covered, ignore this face
	//check the pile below and the 4 neighbors. 
	//if a pile covers lower/upper, then connect

	//decimate (how?)

}

/*void FindNonManifoldEdges(vector<Triangle>& triangles, int pnum)
{
	vector<TrigEdge>edges; edges.reserve(3 * triangles.size());
	for (int t = 0; t < triangles.size(); t++)
	{
		TrigEdge te;

		te.v2 = triangles[t].v2;
		if (triangles[t].v0 < triangles[t].v1)
		{
			te.v0 = triangles[t].v0;
			te.v1 = triangles[t].v1;
		}
		else
		{
			te.v0 = triangles[t].v1;
			te.v1 = triangles[t].v0;
		}
		edges.push_back(te);

		te.v2 = triangles[t].v0;
		if (triangles[t].v1 < triangles[t].v2)
		{
			te.v0 = triangles[t].v1;
			te.v1 = triangles[t].v2;
		}
		else
		{
			te.v0 = triangles[t].v2;
			te.v1 = triangles[t].v1;
		}
		edges.push_back(te);

		te.v2 = triangles[t].v1;
		if (triangles[t].v0 < triangles[t].v2)
		{
			te.v0 = triangles[t].v0;
			te.v1 = triangles[t].v2;
		}
		else
		{
			te.v0 = triangles[t].v2;
			te.v1 = triangles[t].v0;
		}
		edges.push_back(te);
	}

	std::sort(edges.begin(), edges.end());


	vector<bool>NonManifoldVx(pnum, false);
	vector<int>NonManifoldVxRep(pnum, 0);
	int NonManifoldEdgesNum = 0;
	int EdgeRep = 0;
	int oldv0 = -1, oldv1 = -1;
	int EdgeOnMoreThanFour = 0;
	int EdgeOnOdd = 0;
	for (int i = 0; i < edges.size(); i++)
	{
		if (edges[i].v0 != oldv0 || edges[i].v1 != oldv1)
		{
			if (EdgeRep > 2)
			{
				if (EdgeRep > 4)
					EdgeOnMoreThanFour++;
				if (EdgeRep > 1 && EdgeRep % 2 != 0)
					EdgeOnOdd++;

				NonManifoldEdgesNum++;
				NonManifoldVx[oldv0] = true;
				NonManifoldVx[oldv1] = true;
				NonManifoldVxRep[oldv0]++;
				NonManifoldVxRep[oldv1]++;
			}
			oldv0 = edges[i].v0;
			oldv1 = edges[i].v1;

			EdgeRep = 1;
		}
		else
			EdgeRep++;
	}
	
	int NonManifoldVxNum = 0;
	int NonManifoldVxOnMoreThanOne = 0;
	for (int i = 0; i < pnum; i++)
	{
		if (NonManifoldVx[i])
			NonManifoldVxNum++;
		if (NonManifoldVxRep[i] > 1)
			NonManifoldVxOnMoreThanOne++;

	}

	cout << "number of non manifold vertices " << NonManifoldVxNum << '\n';
	cout << "number of non manifold edges " << NonManifoldEdgesNum << '\n';
	cout << "number of non manifold vertices incident to more than one non manifold edges " << NonManifoldVxOnMoreThanOne << '\n';
	cout << "number of non manifold edges between more than 4 triangles " << EdgeOnMoreThanFour << '\n';
	cout << "number of non manifold edges between odd triangles " << EdgeOnOdd << '\n';
		
	int u = 3;
}*/

bool TrigHasTwoVxsNotThird(Triangle trig, int v0, int v1, int v2, int& third)
{
	if (trig.v0 == v0 && trig.v1 == v1 && trig.v2 != v2)
	{
		third = trig.v2;
		return  true;
	}
	if (trig.v0 == v0 && trig.v2 == v1 && trig.v1 != v2)
	{
		third = trig.v1;
		return  true;
	}
	if (trig.v1 == v0 && trig.v0 == v1 && trig.v2 != v2)
	{
		third = trig.v2;
		return  true;
	}
	if (trig.v1 == v0 && trig.v2 == v1 && trig.v0 != v2)
	{
		third = trig.v0;
		return  true;
	}
	if (trig.v2 == v0 && trig.v1 == v1 && trig.v0 != v2)
	{
		third = trig.v0;
		return  true;
	}
	if (trig.v2 == v0 && trig.v0 == v1 && trig.v1 != v2)
	{
		third = trig.v1;
		return  true;
	}

	return false;
}

bool TrigHasTwoVxs(Triangle trig, int v0, int v1, int& third)
{
	if (trig.v0 == v0 && trig.v1 == v1)
	{
		third = trig.v2;
		return  true;
	}
	if (trig.v0 == v0 && trig.v2 == v1)
	{
		third = trig.v1;
		return  true;
	}
	if (trig.v1 == v0 && trig.v0 == v1)
	{
		third = trig.v2;
		return  true;
	}
	if (trig.v1 == v0 && trig.v2 == v1)
	{
		third = trig.v0;
		return  true;
	}
	if (trig.v2 == v0 && trig.v1 == v1)
	{
		third = trig.v0;
		return  true;
	}
	if (trig.v2 == v0 && trig.v0 == v1)
	{
		third = trig.v1;
		return  true;
	}

	return false;
}

void FixNonManifoldEdges(vector<PointCoordsExt>& points, vector<Triangle>& triangles)
{
	vector<vector<int>>vxToTrigs(points.size());
	for (int t = 0; t < triangles.size(); t++)
	{
		vxToTrigs[triangles[t].v0].push_back(t);
		vxToTrigs[triangles[t].v1].push_back(t);
		vxToTrigs[triangles[t].v2].push_back(t);
	}
	
	//for testing purposes
	/*for (int t = 0; t < triangles.size(); t++)
	{
		if (triangles[t].v0 == 22 || triangles[t].v1 == 22 || triangles[t].v2 == 22 || triangles[t].v0 == 101 || triangles[t].v1 == 101 || triangles[t].v2 == 101)
			t = t;
	}*/

	vector<TrigEdge>edges; edges.reserve(3 * triangles.size());
	for (int t = 0; t < triangles.size(); t++)
	{
		TrigEdge te;

		te.v2 = triangles[t].v2;
		if (triangles[t].v0 < triangles[t].v1)
		{
			te.v0 = triangles[t].v0;
			te.v1 = triangles[t].v1;
		}
		else
		{
			te.v0 = triangles[t].v1;
			te.v1 = triangles[t].v0;
		}
		edges.push_back(te);

		te.v2 = triangles[t].v0;
		if (triangles[t].v1 < triangles[t].v2)
		{
			te.v0 = triangles[t].v1;
			te.v1 = triangles[t].v2;
		}
		else
		{
			te.v0 = triangles[t].v2;
			te.v1 = triangles[t].v1;
		}
		edges.push_back(te);

		te.v2 = triangles[t].v1;
		if (triangles[t].v0 < triangles[t].v2)
		{
			te.v0 = triangles[t].v0;
			te.v1 = triangles[t].v2;
		}
		else
		{
			te.v0 = triangles[t].v2;
			te.v1 = triangles[t].v0;
		}
		edges.push_back(te);
	}

	std::sort(edges.begin(), edges.end());


	int NonManifoldEdgesNum = 0;
	int EdgeRep = 0;
	int oldv0 = -1, oldv1 = -1;
	int EdgeOnMoreThanFour = 0;
	int EdgeOnOdd = 0;
	for (int i = 0; i < edges.size(); i++)
	{
		if (edges[i].v0 != oldv0 || edges[i].v1 != oldv1)
		{
			if (EdgeRep > 2)
			{
				int v0 = oldv0;
				int v1 = oldv1;

				vector<int>TrigsToChange;

				int fixed = v0; int othervx = v1;

				int tindex;	int movable;
				bool stop = false;
				for (int vt = 0; vt < vxToTrigs[fixed].size() && !stop; vt++)
				{
					tindex = vxToTrigs[fixed][vt];
					stop = TrigHasTwoVxs(triangles[tindex], fixed, othervx, movable);
				}
				TrigsToChange.push_back(tindex);
				int firstmovable = movable;
				
				int avoided = v1;
				bool OtherTrigFound = false;
				while (!OtherTrigFound)
				{
					int third;
					bool found = false;
					int tindex;
					for (int vt = 0; vt < vxToTrigs[fixed].size() && !found; vt++)
					{
						//get using fixed, movable, avoided
						tindex = vxToTrigs[fixed][vt];
						found = TrigHasTwoVxsNotThird(triangles[tindex], fixed, movable, avoided, third);
					}
					TrigsToChange.push_back(tindex);

					avoided = movable;
					movable = third;

					if (third == othervx)
						OtherTrigFound = true;
				}


				int newv0 = points.size();
				points.push_back(points[v0]);

				for (int j = 0; j < TrigsToChange.size(); j++)
				{
					if (triangles[TrigsToChange[j]].v0 == v0)
						triangles[TrigsToChange[j]].v0 = newv0;
					if (triangles[TrigsToChange[j]].v1 == v0)
						triangles[TrigsToChange[j]].v1 = newv0;
					if (triangles[TrigsToChange[j]].v2 == v0)
						triangles[TrigsToChange[j]].v2 = newv0;
				}
 
			}
			oldv0 = edges[i].v0;
			oldv1 = edges[i].v1;

			EdgeRep = 1;
		}
		else
			EdgeRep++;
	}



}

void CheckConnectedComponents(int pnum, vector<Triangle>&triangles)
{
	vector<bool> taken(pnum, false);

	vector<set<int>>nbs(pnum); //of vertices

	for (int t = 0; t < triangles.size(); t++)
	{
		nbs[triangles[t].v0].insert(triangles[t].v1);
		nbs[triangles[t].v0].insert(triangles[t].v2);
		nbs[triangles[t].v1].insert(triangles[t].v0);
		nbs[triangles[t].v1].insert(triangles[t].v2);
		nbs[triangles[t].v2].insert(triangles[t].v0);
		nbs[triangles[t].v2].insert(triangles[t].v1);
	}

	int cc = 0;
	for (int p = 0; p < pnum; p++)
	{
		if (!taken[p])
		{
			taken[p] = true;
			stack<int>st;
			st.push(p);
			while (!st.empty())
			{
				int pnt = st.top();
				st.pop();

				for (int nb : nbs[pnt])
				{
					if (!taken[nb])
					{
						taken[nb] = true;
						st.push(nb);
					}
				}
			}
			cout << "one connected component finished ***\n";
			cc++;
		}
	}
	cout << "connected components number: " << cc << '\n';

	/*
	int takenNum = 0;
	bool AllTaken = false;
	while (takenNum < pnum)
	{
		for (int t = 0; t < triangles.size(); t++)
		{
			if(triangles[t].v0)
		}
	}
	*/

}

//each pile has 8 points. will be added to a vector
void fillPoints(vector<PileStruct>& Piles, vector<Pixel>& pixels, const int w, const int h, const int left, const int right, float bottom, float top, vector<PointCoordsExt>& points)
{
	for (int pixelIndex = 0; pixelIndex < w * h; pixelIndex++)
	{
		if (pixels[pixelIndex].PilesCount == 0)
			continue;

		float x = pixelIndex % w;
		float y = pixelIndex / w;
		float inc = 1;

		for (int pilePixelIndex = 0; pilePixelIndex < pixels[pixelIndex].PilesCount; pilePixelIndex++)	//index within the pixel piles
		{

			int pileIndex = pixels[pixelIndex].PilesOffset + pilePixelIndex;	//index within whole set of piles
			float zmin = Piles[pileIndex].start;
			float zmax = Piles[pileIndex].end;

			vector<PointCoordsExt>corners{ PointCoordsExt{ x + 0, y + 0, zmin } ,
				PointCoordsExt{ x + inc, y + 0, zmin },
				PointCoordsExt{ x + inc, y + inc, zmin },
				PointCoordsExt{ x + 0, y + inc, zmin },
				PointCoordsExt{ x + 0, y + 0, zmax } ,
				PointCoordsExt{ x + inc, y + 0, zmax },
				PointCoordsExt{ x + inc, y + inc, zmax },
				PointCoordsExt{ x + 0, y + inc, zmax }
			};

			for (int i = 0; i < corners.size(); i++)
			{
				PointCoordsExt pc = corners[i];
				pc.x = ((((pc.x / w) * 2 - 1) * (right - left)) + (right + left)) / 2;
				pc.y = ((((pc.y / h) * 2 - 1) * (top - bottom)) + (top + bottom)) / 2;
				points.push_back(pc);


			}
		}
	}
}

//idea: moave to another vector, along with their indices. define how to compare (x,y,z) then index. sort. search for duplicates, all refer to first.
void findDuplicates(vector<PointCoordsExt>& points, vector<PileStruct>& Piles, vector<Pixel>& pixels, const int w, const int h, vector<int>& firstOccurance)
{
	for (int p = 0; p < points.size(); p++)
	{
		firstOccurance[p] = p;
	}

	for (int pixelIndex = 0; pixelIndex < w * h; pixelIndex++)
	{

		if(!pixels[pixelIndex].ContainsPiles())
			continue;

		int pixelPilesOffset = pixels[pixelIndex].PilesOffset;

		//make 4 direct nbs//
		vector<int>nbpixels;				//the 4 nbs whose indeces are bigger than the index of the pixel
		pixels[pixelIndex].GetNbsWithBiggerIndex(pixels, w, h, nbpixels);

		for (int pilePixelIndex = 0; pilePixelIndex < pixels[pixelIndex].PilesCount; pilePixelIndex++) //go through each pile of the pixel
		{

			int pileIndex =  pixelPilesOffset + pilePixelIndex;
			int vxoffset = 8 * pileIndex;

			for (int nb = 0; nb < nbpixels.size(); nb++) //go through each neighbor of the pixel (we are still inside the pixel)
			{
				int nbpixelIndex = nbpixels[nb];

				bool intersection = false; 
				for (int nbpilePixelIndex = 0; nbpilePixelIndex < pixels[nbpixelIndex].PilesCount && !intersection; nbpilePixelIndex++) //go through each pile of the nb pixel
				{

					int nbpileIndex = pixels[nbpixelIndex].PilesOffset + nbpilePixelIndex;
					int nbvxoffset = 8 * nbpileIndex;

					int ret;

					//check each point of the 8 of the nb pile, against the points of the pile
					for (int j = nbvxoffset; j < nbvxoffset + 8; j++)
					{
						ret = points[j].CheckAgainstPilePoints(points.begin() + vxoffset);
						if (ret != -1) firstOccurance[j] = firstOccurance[ret + vxoffset];
					}

				}
			}

		}
	}
}

//for each pile, fill 4 trigs: 2 for upper face, 2 for lower. edges are not needed for these
void fillUpperLowerFaces(vector<Pixel>& pixels, vector<int>& firstOccurance, const int w, const int h, vector<Triangle>& triangles)
{
	for (int pixelIndex = 0; pixelIndex < w * h; pixelIndex++)
	{

		if (!pixels[pixelIndex].ContainsPiles())
			continue;

		//piles of current pixel//
		for (int pilePixelIndex = 0; pilePixelIndex < pixels[pixelIndex].PilesCount; pilePixelIndex++)
		{

			int pileIndex = pixels[pixelIndex].PilesOffset + pilePixelIndex;
			int pntoffset = 8 * pileIndex;

			//make 4 trigs for upper and lower faces//
			triangles.push_back(Triangle{ firstOccurance[pntoffset], firstOccurance[pntoffset + 2], firstOccurance[pntoffset + 1] });
			triangles.push_back(Triangle{ firstOccurance[pntoffset], firstOccurance[pntoffset + 3], firstOccurance[pntoffset + 2] });
			triangles.push_back(Triangle{ firstOccurance[pntoffset + 4], firstOccurance[pntoffset + 5], firstOccurance[pntoffset + 6] });
			triangles.push_back(Triangle{ firstOccurance[pntoffset + 4], firstOccurance[pntoffset + 6], firstOccurance[pntoffset + 7] });

		}
	}

}

//in short, if an edge intersects with edges of other piles, add the vertices lying inside the edge to the edge vxs
//each pile has 4 edges. each point falling inside the edge of current pile, is added to the points of this edge.
//piles of 4 direct nb pixels intersect the piles of current pixel in faces. an intersection adds 2 points, each on an edge
//piles of 4 cross nb pixels intersect the piles of current pixel in edges. an intersection adds 1 point on an edge
void InsertPointsOnEdges(vector<PileStruct>& Piles, vector<Pixel>& pixels, vector<int>& firstOccurance, const int w, const int h, vector<PileEdges>& pileEdges)
{
	for (int pixelIndex = 0; pixelIndex < w * h; pixelIndex++)
	{

		if (!pixels[pixelIndex].ContainsPiles())
			continue;

		//piles of current pixel//
		for (int pilePixelIndex = 0; pilePixelIndex < pixels[pixelIndex].PilesCount; pilePixelIndex++)
		{

			int pileIndex = pixels[pixelIndex].PilesOffset + pilePixelIndex;
			int vxoffset = 8 * pileIndex;

			float zmin = Piles[pileIndex].start;
			float zmax = Piles[pileIndex].end;

			//each pile has exactly 4 edges. each edge has a set of points
			//We already know two points on each edge of the 4. These are independent on the neighboring pixels//
			pileEdges[pileIndex].edges[0].insert(firstOccurance[vxoffset]); pileEdges[pileIndex].edges[0].insert(firstOccurance[vxoffset + 4]);
			pileEdges[pileIndex].edges[1].insert(firstOccurance[vxoffset + 1]); pileEdges[pileIndex].edges[1].insert(firstOccurance[vxoffset + 5]);
			pileEdges[pileIndex].edges[2].insert(firstOccurance[vxoffset + 2]); pileEdges[pileIndex].edges[2].insert(firstOccurance[vxoffset + 6]);
			pileEdges[pileIndex].edges[3].insert(firstOccurance[vxoffset + 3]); pileEdges[pileIndex].edges[3].insert(firstOccurance[vxoffset + 7]);
		}
	}

	for (int pixelIndex = 0; pixelIndex < w * h; pixelIndex++)
	{

		if (!pixels[pixelIndex].ContainsPiles())
			continue;


		/////////////////////
		//check 4 direct nbs//
		/////////////////////

		vector<int>nbpixels;
		pixels[pixelIndex].GetDirectNbs(pixels, w, h, nbpixels); //pixels that share a face


		//piles of current pixel//
		for (int pilePixelIndex = 0; pilePixelIndex < pixels[pixelIndex].PilesCount; pilePixelIndex++)
		{

			int pileIndex = pixels[pixelIndex].PilesOffset + pilePixelIndex;

			float zmin = Piles[pileIndex].start;
			float zmax = Piles[pileIndex].end;


			for (int nb = 0; nb < nbpixels.size(); nb++) //go through the 4 direct nb pixels (this is for a single pile of the current pixel). compare against the current pile of the current pixel)
			{
				//if a point on an edge of an nb pile intersects an edge of current pile, add the points to the current edge

				int nbpixel = nbpixels[nb];
				bool intersection = false;
				for (int nbpileIndex = 0; nbpileIndex < pixels[nbpixel].PilesCount && !intersection; nbpileIndex++) //go through all piles of each nb pixel
				{

					int nbpile = pixels[nbpixel].PilesOffset + nbpileIndex;
					int nbvxoffset = 8 * nbpile;

					//get zmin and zmax of pile//
					float nbzmin = Piles[nbpile].start;
					float nbzmax = Piles[nbpile].end;

					int facingFace = Pixel::GetFacingFace(pixels[pixelIndex], pixels[nbpixel]);
					int edge0 = facingFace;
					int edge1 = (facingFace + 1) % 4;

					int nbv00, nbv10, nbv01, nbv11;
					Pixel::GetFacingVertices(pixels[pixelIndex], pixels[nbpixel], nbv00, nbv10);
					nbv01 = nbv00 + 4;
					nbv11 = nbv10 + 4;
					
					//add upper two vxs
					if (nbzmin > zmin && nbzmin < zmax)
					{
						pileEdges[pileIndex].edges[edge0].insert(firstOccurance[nbvxoffset + nbv00]);
						pileEdges[pileIndex].edges[edge1].insert(firstOccurance[nbvxoffset + nbv10]);
					}

					//add lower two vxs
					if (nbzmax > zmin && nbzmax < zmax)
					{
						pileEdges[pileIndex].edges[edge0].insert(firstOccurance[nbvxoffset + nbv01]);
						pileEdges[pileIndex].edges[edge1].insert(firstOccurance[nbvxoffset + nbv11]);
					}
				}
			}

		}
		

		//////////////////////////
		//check 4 cross nb pixels//
		//////////////////////////

		nbpixels.clear();
		pixels[pixelIndex].GetCrossNbs(pixels, w, h, nbpixels); //pixels that share an edge
		
		//vector<int>edgesInds;	//which edge of the current pile intersect with the piles of the nb pixel
		//vector<int>nbvs; nbvs.reserve(4);		//which point of the piles of the nb pixel (should be 2 points, the other is the saved + 4)

		
		for (int pilePixelIndex = 0; pilePixelIndex < pixels[pixelIndex].PilesCount; pilePixelIndex++)
		{

			int pileIndex = pixels[pixelIndex].PilesOffset + pilePixelIndex;

			float zmin = Piles[pileIndex].start;
			float zmax = Piles[pileIndex].end;

			for (int nb = 0; nb < nbpixels.size(); nb++)
			{
				int nbpixel = nbpixels[nb];
				for (int nbpileIndex = 0; nbpileIndex < pixels[nbpixels[nb]].PilesCount; nbpileIndex++)
				{

					int nbpile = pixels[nbpixel].PilesOffset + nbpileIndex;
					int nbvxoffset = 8 * nbpile;

					int facingedge = Pixel::GetCrossEdge(pixels[pixelIndex], pixels[nbpixels[nb]]);

					int nbv0 = Pixel::GetCrossVertex(pixels[pixelIndex], pixels[nbpixels[nb]]);
					int nbv1 = nbv0 + 4;

					//get zmin and zmax
					float nbzmin = Piles[nbpile].start;
					float nbzmax = Piles[nbpile].end;

					if (nbzmin > zmin && nbzmin < zmax)
						pileEdges[pileIndex].edges[facingedge].insert(firstOccurance[nbvxoffset + nbv0]);
					if (nbzmax > zmin && nbzmax < zmax)
						pileEdges[pileIndex].edges[facingedge].insert(firstOccurance[nbvxoffset + nbv1]);
				}
			}

		}

	}
}


//each pile has initial 4 faces
//the function just splits each face, depending on the neighboring faces
//each face will have 4 vertices. more vertices are added later in the function
void CreateFaces(vector<PointCoordsExt>& points, vector<PileStruct>& Piles, vector<Pixel>& pixels, vector<int>& firstOccurance, const int w, const int h, vector<PileEdges>& pileEdges, vector<Face>& PilesFaces)
{
	for (int pixelIndex = 0; pixelIndex < w * h; pixelIndex++) //go through all pixels
	{
		if(!pixels[pixelIndex].ContainsPiles())
			continue;


		/////////////////////
		//check 4 direct nbs//
		/////////////////////

		//Note: faces that have no nb pixels should go through the face extraction too ! and we assume each face has its corresponding nb at same index. so, even if no nbs, keep a -1
		vector<int>nbpixels;
		pixels[pixelIndex].GetDirectNbs(pixels, w, h, nbpixels, false); //pixels that share a face
		

		//the two vertices (of 8) of the piles of each neighboring pixels that can intersect with the piles of the current pixel
		//actually, should be 4 vertices. the saved two are incremented by 4 to get the other two.
		vector<int>nbv0s; nbv0s.reserve(4);
		vector<int>nbv1s; nbv1s.reserve(4);

		//what happens if face has no corresponding nb pixel? fixed//

		for (int pilePixelIndex = 0; pilePixelIndex < pixels[pixelIndex].PilesCount; pilePixelIndex++) //go through the piles of current pixel
		{

			int pileIndex = pixels[pixelIndex].PilesOffset + pilePixelIndex;
			int vxoffset = 8 * pileIndex;


			for (int nb = 0; nb < nbpixels.size(); nb++)	//go through the 4 direct nbs
			{
				//the two vertices (of 8) of the piles of the current pixel that can intersect with the piles of each neighboring pixels
				//actually, should be 4 vertices. the saved two are incremented by 4 to get the other two.
				int v0 = nb;
				int v1 = (nb + 1) % nbpixels.size();

				
				//very first 4 vertices of a pile
				Face fc; 
				fc.v0 = firstOccurance[vxoffset + v0]; 
				fc.v1 = firstOccurance[vxoffset + v1]; 
				fc.v2 = firstOccurance[vxoffset + v0 + 4]; 
				fc.v3 = firstOccurance[vxoffset + v1 + 4];

				vector<Face>FaceSubFaces;

				//can also have them from piles. all upper vxs have same z, and also lower.
				//are these updated after cutting? check!
				//pile has different faces. z values have to be found here, not before loop.//
				float fzmin = points[fc.v0].z;
				float fzmax = points[fc.v2].z;

				//probably cut face in case the faces intersect
				bool TakeFace = true;
				bool stop = false; //gets true when we pass our pile anyway along z
				if (nbpixels[nb] != -1)	//we have an nb pixel. otherwise, will just add the face as it is
				{
					int nbv0, nbv1;
					Pixel::GetFacingVertices(pixels[pixelIndex], pixels[nbpixels[nb]], nbv0, nbv1);


					int nbpixel = nbpixels[nb];
					for (int nbpileIndex = 0; nbpileIndex < pixels[nbpixel].PilesCount && !stop; nbpileIndex++) //in case no pile, or no intersection, the face is moved as is//
					{

						int nbpile = pixels[nbpixel].PilesOffset + nbpileIndex;
						int nbvxoffset = 8 * nbpile;

						//get zmin and zmax
						float nbzmin = Piles[nbpile].start;
						float nbzmax = Piles[nbpile].end;

						//make a face from the current pile here
						Face nbfc; nbfc.v0 = firstOccurance[nbvxoffset + nbv0]; nbfc.v1 = firstOccurance[nbvxoffset + nbv1]; nbfc.v2 = firstOccurance[nbvxoffset + nbv0 + 4]; nbfc.v3 = firstOccurance[nbvxoffset + nbv1 + 4];

						if (fzmin < nbzmin && nbzmin < fzmax && fzmin < nbzmax && nbzmax < fzmax) //splits the face, and lose a part in the middle
						{
							//add a face to addt//
							Face newface;
							newface.v0 = fc.v0;
							newface.v1 = fc.v1;
							newface.v2 = nbfc.v0;
							newface.v3 = nbfc.v1;
							FaceSubFaces.push_back(newface);

							//update face//
							fc.v0 = nbfc.v2; fc.v1 = nbfc.v3; //take lower vxs of neighbor//
							fzmin = nbzmax;


						}
						else if (fzmin < nbzmin && nbzmin < fzmax) //nbzmax could be equal to fzmax//
						{
							//update face//
							fc.v2 = nbfc.v0; fc.v3 = nbfc.v1;
							//no need to update z value. we stop anyway//

							stop = true;

						}
						else if (fzmin < nbzmax && nbzmax < fzmax)
						{
							//update face//
							fc.v0 = nbfc.v2; fc.v1 = nbfc.v3;
							fzmin = nbzmax;
						}
						else if (nbzmin <= fzmin && fzmax <= nbzmax) //nbzmin could be equal to fzmin// //pile is totally covered//
						{
							TakeFace = false;
							stop = true;
						}
						else if (nbzmin >= fzmax) //passed the pile
						{
							stop = true;
						}

					}

				}

				if (TakeFace)
					FaceSubFaces.push_back(fc);

				//distribute vxs over edges of faces//
				//PilesEdges are used to add vxs to the 2 edges of the found faces !
				for (auto f : FaceSubFaces)
				{
					f.FillVerticesFromEdgesToEdges(pileEdges[pileIndex].edges[v0], pileEdges[pileIndex].edges[v1], points);
					PilesFaces.push_back(f);
				}

			}

		}
	}
}

void TriangulateFaces(vector<Face>& pilesFaces, vector<PointCoordsExt>& points, vector<Triangle>& triangles)
{
	for (Face fc : pilesFaces)
	{

		//remove duplicates//
		fc.e0.RemoveDuplicates();
		fc.e1.RemoveDuplicates();

		
		//sort//
		fc.e0.SortByZ(points);
		fc.e1.SortByZ(points);

		//after sorting//
		//triangulate by moving two pointers along the two edges, and advance
		Triangle::Triangulate(fc, triangles);
	}
}

void MeshPiles(string modelpath, string modelname, vector<PileStruct>& Piles, vector<Pixel>& pixels, const int w, const int h, float left, float right, float bottom, float top, float near, float far)
{
	//start file
	ofstream oplyfile(modelpath + modelname + "_mesh.ply", std::ios_base::out);

	//compute vxs and trigs from piles size
	//int pnum = Piles.size()/100;
	int pnum = Piles.size();
	int vnum = pnum * 8;
	int tnum = 0;


	vector<PointCoordsExt>points;
	points.reserve(pnum);
	fillPoints(Piles, pixels, w, h, left, right, bottom, top, points);
	cout << "points coords calculations ended\n";


	//find duplicates//
	vector<int>firstOccurance(points.size());
	findDuplicates(points, Piles, pixels, w, h, firstOccurance);
	cout << "finding duplicates ended\n";

	/////////////////
	//now the faces//
	/////////////////

	vector<Triangle>triangles; //each triangls has 3 indeces of points


	//upper and lower faces//
	fillUpperLowerFaces(pixels, firstOccurance, w, h, triangles);
	cout << "upper and lower faces ended\n";


	//points on edges//
	//fill EdgesPoints. each pile has 4 edges, and each edge has a set of vertices//
	vector<PileEdges>pileEdges(Piles.size());
	InsertPointsOnEdges(Piles, pixels, firstOccurance, w, h, pileEdges);
	cout << "inserting points on edges ended\n";

	
	//divide into faces, and distribute points of edges
	//loop over pixels
	//loop over piles
	//for each, loop over piles of 4 direct nbs
	//consider current pile a face
	//each intersecting pile go through the faces, divides one of them into one or two faces
	//after finish, fill the points of the 2 edges of the face
	
	vector<Face>pilesFaces; pilesFaces.reserve(Piles.size());//could be larger or smaller//

	CreateFaces(points, Piles, pixels, firstOccurance, w, h, pileEdges, pilesFaces);
	cout << "creating faces ended\n";

	//triangulate faces, using vxs of edges
	//for each face, sort vxs of each edge by z. then 
	TriangulateFaces(pilesFaces, points, triangles);
	cout << "triangulating faces ended\n";
	
	FixNonManifoldEdges(points, triangles);


	//write!//

	//write header
	oplyfile << "ply\n";
	oplyfile << "format ascii 1.0\n";
	oplyfile << "comment generated by mradwan\n";
	oplyfile << "element vertex " << points.size() << "\n";
	oplyfile << "property float x\n";
	oplyfile << "property float y\n";
	oplyfile << "property float z\n";
	/*oplyfile << "property uchar red\n";
	oplyfile << "property uchar green\n";
	oplyfile << "property uchar blue\n";*/
	oplyfile << "element face " << triangles.size() << '\n';
	oplyfile << "property list uchar int vertex_indices\n";
	oplyfile << "end_header\n";

	for (int p = 0; p < points.size(); p++)
	{
		//oplyfile << points[p].x << ' ' << points[p].y << ' ' << points[p].z << " 0 255 0\n";
		oplyfile << points[p].x << ' ' << points[p].y << ' ' << points[p].z << " \n";
	}
	for (int t = 0; t < triangles.size(); t++)
	{
		oplyfile << "3 " << triangles[t].v0 << ' ' << triangles[t].v1 << ' ' << triangles[t].v2 << '\n';
	}

	cout << "wirting visual piles finished\n";
	oplyfile.flush();
	oplyfile.close();


	CheckConnectedComponents(points.size(), triangles);


	//for testing purposes
	/*vector<triangle>triangles2(5);
	triangles2[0].v0 = 0; triangles2[0].v1 = 1; triangles2[0].v2 = 2;
	triangles2[1].v0 = 3; triangles2[1].v1 = 4; triangles2[1].v2 = 5;
	triangles2[2].v0 = 1; triangles2[2].v1 = 2; triangles2[2].v2 = 4;
	triangles2[3].v0 = 4; triangles2[3].v1 = 5; triangles2[3].v2 = 6;
	triangles2[4].v0 = 7; triangles2[4].v1 = 8; triangles2[4].v2 = 9;
	CheckConnectedComponents(10, triangles2);*/

	//FindNonManifoldEdges(triangles, points.size()); //use this to find non manifold edges//
	
}

void VisualizePiles(string modelpath, string modelname, vector<PileStruct>& Piles, vector<int>& PilesCount, vector<int>& PilesOffset, const int w, const int h, float left, float right, float bottom, float top, float near, float far)
{

	//start file
	ofstream oplyfile(modelpath + modelname + "_piles.ply", std::ios_base::out);

	//compute vxs and trigs from piles size
	//int pnum = Piles.size()/100;
	int pnum = Piles.size();
	int vnum = pnum * 8;
	int tnum = pnum * 12;

	//write header
	oplyfile << "ply\n";
	oplyfile << "format ascii 1.0\n";
	oplyfile << "comment generated by mradwan\n";
	oplyfile << "element vertex " << vnum << "\n";
	oplyfile << "property float x\n";
	oplyfile << "property float y\n";
	oplyfile << "property float z\n";
	oplyfile << "element face " << tnum << '\n';
	oplyfile << "property list uchar int vertex_indices\n";
	oplyfile << "end_header\n";

	/*float minpx = numeric_limits<float>::max();
	float minpy = numeric_limits<float>::max();
	float minpz = numeric_limits<float>::max();
	float maxpx = numeric_limits<float>::lowest();
	float maxpy = numeric_limits<float>::lowest();
	float maxpz = numeric_limits<float>::lowest();*/

	int MoreThanOne = 0;//for testing purposes
	int MoreThanZero = 0;//for testing purposes
	float xs[8]; float ys[8]; float zs[8];
	int poffset = 0;
	int pcounter = 0;
	for (int pixel = 0; pixel < w * h && pcounter < pnum; pixel++)
	{
		if (PilesCount[pixel] == 0)
			continue;

		//for testing purposes
		if (PilesCount[pixel] > 1)
			MoreThanOne++;

		//for testing purposes
		if (PilesCount[pixel] > 0)
			MoreThanZero++;

		float x = pixel % w;
		float y = pixel / w;
		float inc = 1;

		for (int pileIndex = 0; pileIndex < PilesCount[pixel] && pcounter < pnum; pileIndex++)
		{

			int pile = poffset + pileIndex;
			float zmin = Piles[pile].start;
			float zmax = Piles[pile].end;


			xs[0] = x + 0, ys[0] = y + 0, zs[0] = zmin;
			xs[1] = x + inc, ys[1] = y + 0, zs[1] = zmin;
			xs[2] = x + inc, ys[2] = y + inc, zs[2] = zmin;
			xs[3] = x + 0, ys[3] = y + inc, zs[3] = zmin;
			xs[4] = x + 0, ys[4] = y + 0, zs[4] = zmax;
			xs[5] = x + inc, ys[5] = y + 0, zs[5] = zmax;
			xs[6] = x + inc, ys[6] = y + inc, zs[6] = zmax;
			xs[7] = x + 0, ys[7] = y + inc, zs[7] = zmax;

			for (int i = 0; i < 8; i++)
			{
				float px = xs[i]; float py = ys[i]; float pz = zs[i];
				px = ((((px / w) * 2 - 1) * (right - left)) + (right + left)) / 2;
				py = ((((py / h) * 2 - 1) * (top - bottom)) + (top + bottom)) / 2;
				//pz = -pz;

				oplyfile << px << ' ' << py << ' ' << pz << '\n';

				//for testing purposes
				/*if (px < minpx)
					minpx = px;
				if (px > maxpx)
					maxpx = px;
				if (py < minpy)
					minpy = py;
				if (py > maxpy)
					maxpy = py;
				if (pz < minpz)
					minpz = pz;
				if (pz > maxpz)
					maxpz = pz;*/
			}

			pcounter++;
		}
		poffset += PilesCount[pixel];
	}
	//cout << minpx << " " << maxpx << " " << minpy << " " << maxpy << " " << minpz << " " << maxpz << "\n"; //for testing purposes

	poffset = 0;
	//int toffset = 1;
	int toffset = 0;
	pcounter = 0;
	for (int pixel = 0; pixel < w * h && pcounter < pnum; pixel++)
	{
		if (PilesCount[pixel] == 0)
			continue;

		for (int pileIndex = 0; pileIndex < PilesCount[pixel] && pcounter < pnum; pileIndex++)
		{

			int pile = poffset + pileIndex;

			int t0 = toffset;
			int t1 = toffset + 1;
			int t2 = toffset + 2;
			int t3 = toffset + 3;
			int t4 = toffset + 4;
			int t5 = toffset + 5;
			int t6 = toffset + 6;
			int t7 = toffset + 7;

			//the 12 trigs
			oplyfile << "3 " << t0 << ' ' << t1 << ' ' << t2 << '\n';
			oplyfile << "3 " << t0 << ' ' << t2 << ' ' << t3 << '\n';
			oplyfile << "3 " << t4 << ' ' << t6 << ' ' << t5 << '\n';
			oplyfile << "3 " << t4 << ' ' << t7 << ' ' << t6 << '\n';
			oplyfile << "3 " << t0 << ' ' << t3 << ' ' << t7 << '\n';
			oplyfile << "3 " << t0 << ' ' << t7 << ' ' << t4 << '\n';
			oplyfile << "3 " << t1 << ' ' << t5 << ' ' << t6 << '\n';
			oplyfile << "3 " << t1 << ' ' << t6 << ' ' << t2 << '\n';
			oplyfile << "3 " << t0 << ' ' << t4 << ' ' << t5 << '\n';
			oplyfile << "3 " << t0 << ' ' << t5 << ' ' << t1 << '\n';
			oplyfile << "3 " << t3 << ' ' << t2 << ' ' << t6 << '\n';
			oplyfile << "3 " << t3 << ' ' << t6 << ' ' << t7 << '\n';


			toffset += 8;

			pcounter++;
		}
	}

	//for testing purposes
	cout << MoreThanOne << " pixles have more than one pile\n";
	cout << MoreThanZero << " pixles have piles\n";

	cout << "wirting visual piles finished\n";
	oplyfile.flush();
	oplyfile.close();
}

//incorrect//
	//same vertices in all edges of a face//
	/*for (int pileindex = 0; pileindex < Piles.size(); pileindex++)
	{
		for (int v : EdgesPoints[4 * pileindex + 0])
		{
			if (EdgesPoints[4 * pileindex + 1].find(v) == EdgesPoints[4 * pileindex + 1].end())
				EdgesPoints[4 * pileindex + 1].insert(v);
			if (EdgesPoints[4 * pileindex + 2].find(v) == EdgesPoints[4 * pileindex + 2].end())
				EdgesPoints[4 * pileindex + 2].insert(v);
			if (EdgesPoints[4 * pileindex + 3].find(v) == EdgesPoints[4 * pileindex + 3].end())
				EdgesPoints[4 * pileindex + 3].insert(v);
		}

		for (int v : EdgesPoints[4 * pileindex + 1])
		{
			if (EdgesPoints[4 * pileindex + 0].find(v) == EdgesPoints[4 * pileindex + 0].end())
				EdgesPoints[4 * pileindex + 0].insert(v);
			if (EdgesPoints[4 * pileindex + 2].find(v) == EdgesPoints[4 * pileindex + 2].end())
				EdgesPoints[4 * pileindex + 2].insert(v);
			if (EdgesPoints[4 * pileindex + 3].find(v) == EdgesPoints[4 * pileindex + 3].end())
				EdgesPoints[4 * pileindex + 3].insert(v);
		}

		for (int v : EdgesPoints[4 * pileindex + 2])
		{
			if (EdgesPoints[4 * pileindex + 1].find(v) == EdgesPoints[4 * pileindex + 1].end())
				EdgesPoints[4 * pileindex + 1].insert(v);
			if (EdgesPoints[4 * pileindex + 0].find(v) == EdgesPoints[4 * pileindex + 0].end())
				EdgesPoints[4 * pileindex + 0].insert(v);
			if (EdgesPoints[4 * pileindex + 3].find(v) == EdgesPoints[4 * pileindex + 3].end())
				EdgesPoints[4 * pileindex + 3].insert(v);
		}

		for (int v : EdgesPoints[4 * pileindex + 3])
		{
			if (EdgesPoints[4 * pileindex + 1].find(v) == EdgesPoints[4 * pileindex + 1].end())
				EdgesPoints[4 * pileindex + 1].insert(v);
			if (EdgesPoints[4 * pileindex + 2].find(v) == EdgesPoints[4 * pileindex + 2].end())
				EdgesPoints[4 * pileindex + 2].insert(v);
			if (EdgesPoints[4 * pileindex + 0].find(v) == EdgesPoints[4 * pileindex + 0].end())
				EdgesPoints[4 * pileindex + 0].insert(v);
		}
	}

	vector<set<int>>EdgesSortedPoints(Piles.size() * 4);
	*/
