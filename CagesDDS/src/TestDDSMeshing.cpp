#include "DDS.h"


void DDS::DataForTesting()
{
	PilesCount.resize(globalW * globalH);
	PilesOffset.resize(globalW * globalH);

	for (int i = 0; i < globalW * globalH; i++)
		PilesCount[i] = 0;

	/*PilesCount[9] = 1;
	PilesCount[11] = 1;

	PilesOffset[0] = 0;
	for (int i = 1; i < globalW * globalH; i++)
		PilesOffset[i] = PilesOffset[i - 1] + PilesCount[i - 1];

	PilesNum = 2;

	Piles.clear();
	Piles.resize(PilesNum);

	Piles[0].pixel = 9;
	Piles[0].start = -0.03;
	Piles[0].end = -0.04;

	Piles[1].pixel = 11;
	Piles[1].start = -0.06;
	Piles[1].end = -0.08;*/

	/*
	//success !!!
	PilesCount[41] = 1;
	PilesCount[42] = 1;

	PilesOffset[0] = 0;
	for (int i = 1; i < globalW * globalH; i++)
		PilesOffset[i] = PilesOffset[i - 1] + PilesCount[i - 1];

	PilesNum = 2;

	Piles.clear();
	Piles.resize(PilesNum);

	Piles[0].pixel = 41;
	Piles[0].start = 0.03;
	Piles[0].end = 0.04;

	Piles[1].pixel = 42;
	Piles[1].start = 0.035;
	Piles[1].end = 0.045;
	*/

	/*PilesCount[41] = 1;
	PilesCount[42] = 1;
	PilesCount[43] = 1;
	PilesCount[44] = 1;

	PilesOffset[0] = 0;
	for (int i = 1; i < globalW * globalH; i++)
		PilesOffset[i] = PilesOffset[i - 1] + PilesCount[i - 1];

	PilesNum = 4;

	Piles.clear();
	Piles.resize(PilesNum);

	Piles[0].pixel = 41;
	Piles[0].start = 0.03;
	Piles[0].end = 0.04;

	Piles[1].pixel = 42;
	Piles[1].start = 0.035;
	Piles[1].end = 0.045;

	Piles[2].pixel = 43;
	Piles[2].start = 0.0425;
	Piles[2].end = 0.0525;

	Piles[3].pixel = 44;
	Piles[3].start = 0.0425;
	Piles[3].end = 0.0525;*/

	/*PilesCount[40] = 1;
	PilesCount[41] = 1;

	PilesOffset[0] = 0;
	for (int i = 1; i < globalW * globalH; i++)
		PilesOffset[i] = PilesOffset[i - 1] + PilesCount[i - 1];

	PilesNum = 2;

	Piles.clear();
	Piles.resize(PilesNum);

	Piles[0].pixel = 40;
	Piles[0].start = 0.036;
	Piles[0].end = 0.038;

	Piles[1].pixel = 41;
	Piles[1].start = 0.03;
	Piles[1].end = 0.04;*/

	/*PilesCount[40] = 1;
	PilesCount[41] = 1;
	PilesCount[42] = 1;
	PilesCount[43] = 1;
	PilesCount[44] = 1;
	PilesCount[57] = 1;

	PilesOffset[0] = 0;
	for (int i = 1; i < globalW * globalH; i++)
		PilesOffset[i] = PilesOffset[i - 1] + PilesCount[i - 1];

	PilesNum = 6;

	Piles.clear();
	Piles.resize(PilesNum);

	Piles[0].pixel = 40;
	Piles[0].start = 0.036;
	Piles[0].end = 0.038;

	Piles[1].pixel = 41;
	Piles[1].start = 0.03;
	Piles[1].end = 0.04;

	Piles[2].pixel = 42;
	Piles[2].start = 0.035;
	Piles[2].end = 0.045;

	Piles[3].pixel = 43;
	Piles[3].start = 0.0425;
	Piles[3].end = 0.0525;

	Piles[4].pixel = 44;
	Piles[4].start = 0.0425;
	Piles[4].end = 0.0525;

	Piles[5].pixel = 41;
	Piles[5].start = 0.03;
	Piles[5].end = 0.04;*/

	/*PilesCount[40] = 1;
	PilesCount[41] = 1;
	PilesCount[42] = 1;
	PilesCount[43] = 1;
	PilesCount[44] = 1;

	PilesOffset[0] = 0;
	for (int i = 1; i < globalW * globalH; i++)
		PilesOffset[i] = PilesOffset[i - 1] + PilesCount[i - 1];

	PilesNum = 5;

	Piles.clear();
	Piles.resize(PilesNum);

	Piles[0].pixel = 40;
	Piles[0].start = 0.036;
	Piles[0].end = 0.038;

	Piles[1].pixel = 41;
	Piles[1].start = 0.03;
	Piles[1].end = 0.04;

	Piles[2].pixel = 42;
	Piles[2].start = 0.035;
	Piles[2].end = 0.045;

	Piles[3].pixel = 43;
	Piles[3].start = 0.0425;
	Piles[3].end = 0.0525;

	Piles[4].pixel = 44;
	Piles[4].start = 0.0425;
	Piles[4].end = 0.0525;*/

	
	/*PilesCount[40] = 2;
	PilesCount[41] = 4;
	PilesCount[42] = 1;
	PilesCount[43] = 2;
	
	PilesOffset[0] = 0;
	for (int i = 1; i < globalW * globalH; i++)
		PilesOffset[i] = PilesOffset[i - 1] + PilesCount[i - 1];

	PilesNum = 9;

	Piles.clear();
	Piles.resize(PilesNum);

	Piles[0].pixel = 40;
	Piles[0].start = 0.01;
	Piles[0].end = 0.05;

	Piles[1].pixel = 40;
	Piles[1].start = 0.055;
	Piles[1].end = 0.09;

	Piles[2].pixel = 41;
	Piles[2].start = 0.02;
	Piles[2].end = 0.04;

	Piles[3].pixel = 41;
	Piles[3].start = 0.06;
	Piles[3].end = 0.065;

	Piles[4].pixel = 41;
	Piles[4].start = 0.07;
	Piles[4].end = 0.075;

	Piles[5].pixel = 41;
	Piles[5].start = 0.08;
	Piles[5].end = 0.1;

	Piles[6].pixel = 42;
	Piles[6].start = 0.03;
	Piles[6].end = 0.09;

	Piles[7].pixel = 43;
	Piles[7].start = 0.01;
	Piles[7].end = 0.04;

	Piles[8].pixel = 43;
	Piles[8].start = 0.085;
	Piles[8].end = 0.11;*/

	
	/*PilesCount[40] = 2;
	PilesCount[41] = 4;
	PilesCount[42] = 1;
	PilesCount[43] = 2;
	PilesCount[55] = 1;

	PilesOffset[0] = 0;
	for (int i = 1; i < globalW * globalH; i++)
		PilesOffset[i] = PilesOffset[i - 1] + PilesCount[i - 1];

	PilesNum = 10;

	Piles.clear();
	Piles.resize(PilesNum);

	Piles[0].pixel = 40;
	Piles[0].start = 0.01;
	Piles[0].end = 0.05;

	Piles[1].pixel = 40;
	Piles[1].start = 0.055;
	Piles[1].end = 0.09;

	Piles[2].pixel = 41;
	Piles[2].start = 0.02;
	Piles[2].end = 0.04;

	Piles[3].pixel = 41;
	Piles[3].start = 0.06;
	Piles[3].end = 0.065;

	Piles[4].pixel = 41;
	Piles[4].start = 0.07;
	Piles[4].end = 0.075;

	Piles[5].pixel = 41;
	Piles[5].start = 0.08;
	Piles[5].end = 0.1;

	Piles[6].pixel = 42;
	Piles[6].start = 0.03;
	Piles[6].end = 0.09;

	Piles[7].pixel = 43;
	Piles[7].start = 0.01;
	Piles[7].end = 0.04;

	Piles[8].pixel = 43;
	Piles[8].start = 0.085;
	Piles[8].end = 0.11;

	Piles[9].pixel = 55;
	Piles[9].start = 0.0765;
	Piles[9].end = 0.0785;*/


	/*PilesCount[40] = 1;
	PilesCount[55] = 1;

	PilesOffset[0] = 0;
	for (int i = 1; i < globalW * globalH; i++)
		PilesOffset[i] = PilesOffset[i - 1] + PilesCount[i - 1];

	PilesNum = 2;

	Piles.clear();
	Piles.resize(PilesNum);

	Piles[0].pixel = 40;
	Piles[0].start = 0.055;
	Piles[0].end = 0.09;

	
	Piles[1].pixel = 55;
	Piles[1].start = 0.0765;
	Piles[1].end = 0.0785;*/


	/*PilesCount[40] = 2;
	PilesCount[41] = 4;
	PilesCount[42] = 1;
	PilesCount[43] = 2;
	PilesCount[56] = 2;
	PilesCount[57] = 4;
	PilesCount[58] = 1;
	PilesCount[59] = 2;

	PilesOffset[0] = 0;
	for (int i = 1; i < globalW * globalH; i++)
		PilesOffset[i] = PilesOffset[i - 1] + PilesCount[i - 1];

	PilesNum = 18;

	Piles.clear();
	Piles.resize(PilesNum);

	Piles[0].pixel = 40;
	Piles[0].start = 0.01;
	Piles[0].end = 0.05;

	Piles[1].pixel = 40;
	Piles[1].start = 0.055;
	Piles[1].end = 0.09;

	Piles[2].pixel = 41;
	Piles[2].start = 0.02;
	Piles[2].end = 0.04;

	Piles[3].pixel = 41;
	Piles[3].start = 0.06;
	Piles[3].end = 0.065;

	Piles[4].pixel = 41;
	Piles[4].start = 0.07;
	Piles[4].end = 0.075;

	Piles[5].pixel = 41;
	Piles[5].start = 0.08;
	Piles[5].end = 0.1;

	Piles[6].pixel = 42;
	Piles[6].start = 0.03;
	Piles[6].end = 0.09;

	Piles[7].pixel = 43;
	Piles[7].start = 0.01;
	Piles[7].end = 0.04;

	Piles[8].pixel = 43;
	Piles[8].start = 0.085;
	Piles[8].end = 0.11;

	Piles[9].pixel = 40;
	Piles[9].start = 0.01;
	Piles[9].end = 0.05;

	Piles[10].pixel = 40;
	Piles[10].start = 0.055;
	Piles[10].end = 0.09;

	Piles[11].pixel = 41;
	Piles[11].start = 0.02;
	Piles[11].end = 0.04;

	Piles[12].pixel = 41;
	Piles[12].start = 0.06;
	Piles[12].end = 0.065;

	Piles[13].pixel = 41;
	Piles[13].start = 0.07;
	Piles[13].end = 0.075;

	Piles[14].pixel = 41;
	Piles[14].start = 0.08;
	Piles[14].end = 0.1;

	Piles[15].pixel = 42;
	Piles[15].start = 0.03;
	Piles[15].end = 0.09;

	Piles[16].pixel = 43;
	Piles[16].start = 0.01;
	Piles[16].end = 0.04;

	Piles[17].pixel = 43;
	Piles[17].start = 0.085;
	Piles[17].end = 0.11;*/


	/*PilesCount[40] = 2;
	PilesCount[41] = 4;
	PilesCount[42] = 1;
	PilesCount[43] = 2;
	PilesCount[57] = 2;
	PilesCount[58] = 4;
	PilesCount[59] = 1;
	PilesCount[60] = 2;

	PilesOffset[0] = 0;
	for (int i = 1; i < globalW * globalH; i++)
		PilesOffset[i] = PilesOffset[i - 1] + PilesCount[i - 1];

	PilesNum = 18;

	Piles.clear();
	Piles.resize(PilesNum);

	Piles[0].pixel = 40;
	Piles[0].start = 0.01;
	Piles[0].end = 0.05;

	Piles[1].pixel = 40;
	Piles[1].start = 0.055;
	Piles[1].end = 0.09;

	Piles[2].pixel = 41;
	Piles[2].start = 0.02;
	Piles[2].end = 0.04;

	Piles[3].pixel = 41;
	Piles[3].start = 0.06;
	Piles[3].end = 0.065;

	Piles[4].pixel = 41;
	Piles[4].start = 0.07;
	Piles[4].end = 0.075;

	Piles[5].pixel = 41;
	Piles[5].start = 0.08;
	Piles[5].end = 0.1;

	Piles[6].pixel = 42;
	Piles[6].start = 0.03;
	Piles[6].end = 0.09;

	Piles[7].pixel = 43;
	Piles[7].start = 0.01;
	Piles[7].end = 0.04;

	Piles[8].pixel = 43;
	Piles[8].start = 0.085;
	Piles[8].end = 0.11;

	Piles[9].pixel = 40;
	Piles[9].start = 0.01;
	Piles[9].end = 0.05;

	Piles[10].pixel = 40;
	Piles[10].start = 0.055;
	Piles[10].end = 0.09;

	Piles[11].pixel = 41;
	Piles[11].start = 0.02;
	Piles[11].end = 0.04;

	Piles[12].pixel = 41;
	Piles[12].start = 0.06;
	Piles[12].end = 0.065;

	Piles[13].pixel = 41;
	Piles[13].start = 0.07;
	Piles[13].end = 0.075;

	Piles[14].pixel = 41;
	Piles[14].start = 0.08;
	Piles[14].end = 0.1;

	Piles[15].pixel = 42;
	Piles[15].start = 0.03;
	Piles[15].end = 0.09;

	Piles[16].pixel = 43;
	Piles[16].start = 0.01;
	Piles[16].end = 0.04;

	Piles[17].pixel = 43;
	Piles[17].start = 0.085;
	Piles[17].end = 0.11;*/

	
	PilesCount[40] = 2;
	PilesCount[41] = 4;
	PilesCount[42] = 1;
	PilesCount[43] = 2;
	PilesCount[44] = 1;
	PilesCount[47] = 1;
	PilesCount[48] = 1;
	PilesCount[255] = 1;

	PilesOffset[0] = 0;
	for (int i = 1; i < globalW * globalH; i++)
		PilesOffset[i] = PilesOffset[i - 1] + PilesCount[i - 1];

	PilesNum = 13;

	Piles.clear();
	Piles.resize(PilesNum);

	Piles[0].pixel = 40;
	Piles[0].start = 0.01;
	Piles[0].end = 0.05;

	Piles[1].pixel = 40;
	Piles[1].start = 0.055;
	Piles[1].end = 0.09;

	Piles[2].pixel = 41;
	Piles[2].start = 0.02;
	Piles[2].end = 0.04;

	Piles[3].pixel = 41;
	Piles[3].start = 0.06;
	Piles[3].end = 0.065;

	Piles[4].pixel = 41;
	Piles[4].start = 0.07;
	Piles[4].end = 0.075;

	Piles[5].pixel = 41;
	Piles[5].start = 0.08;
	Piles[5].end = 0.1;

	Piles[6].pixel = 42;
	Piles[6].start = 0.03;
	Piles[6].end = 0.09;

	Piles[7].pixel = 43;
	Piles[7].start = 0.01;
	Piles[7].end = 0.04;

	Piles[8].pixel = 43;
	Piles[8].start = 0.085;
	Piles[8].end = 0.11;

	Piles[9].pixel = 44;
	Piles[9].start = 0.06;
	Piles[9].end = 0.07;

	Piles[10].pixel = 47;
	Piles[10].start = 0.06;
	Piles[10].end = 0.07;

	Piles[11].pixel = 48;
	Piles[11].start = 0.06;
	Piles[11].end = 0.07;

	Piles[12].pixel = 255;
	Piles[12].start = 0.06;
	Piles[12].end = 0.07;


}