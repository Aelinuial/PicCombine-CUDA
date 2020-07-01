#if defined(WIN32)
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "stdafx.h"

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <queue>
#include <math.h>  

#include "common/vec3f.h"
#include "common/box.h"
#include "common/xyz-rgb.h"
#include "common/timer.h"
#include "common/bvh.h"
#include "cudaerror.h"
using namespace std;


void build_bvh();
void destroy_bvh();
bool findNearest2(xyz2rgb &org, xyz2rgb &ret);
int maxPts();

t_bvhTree *gTree;
t_linearBVH *lTree;
std::vector<xyz2rgb> gHashTab;

void build_bvh()
{
	gTree = new t_bvhTree(gHashTab);
	lTree = new t_linearBVH(gTree);
}

void destroy_bvh()
{
	delete gTree;
	delete lTree;
}

bool findNearest2(xyz2rgb &org, xyz2rgb &ret)
{
	return lTree->query(org, ret);
}

int maxPts() { return gTree->maxPts(); }



bool getColor(vec3f &xyz, float *nearest, vec3f &cr, int &idx, int &pidx)
{
	xyz2rgb input(xyz, vec3f(), idx, pidx), ret;

	//给input找xyz最近点
	bool find = findNearest2(input, ret);
	if (!find)
		return false;

	vec3f now = ret.xyz();
	double dist = vdistance(xyz, now);
	if (dist < *nearest) {
		//对于这个像素来说，把最近距离替换成最近点的距离
		*nearest = dist;
		cr = ret.rgb();
		idx = ret.index();
		pidx = ret.pos();

		return true;
	}
	else
		return false;
}

class SceneData {
public:
	int _cx, _cy;
	float *_xyzs;	//深度图数据
	float *_rgbs;	//读进来的RGB数据

	float *_rgbsNew;	//重建的新图的RGB数据
	BOX _bound;

public:
	void loadRGB(char *fname)
	{
		FILE *fp = fopen(fname, "rb");
		//buffer首个对象的指针，每个对象的大小，对象个数，输入流
		fread(&_cx, sizeof(int), 1, fp);
		fread(&_cy, sizeof(int), 1, fp);

		int sz = _cx * _cy * 3;	//图像的长乘宽乘三通道
		_rgbs = new float[sz];
		fread(_rgbs, sizeof(float), sz, fp);	//读入所有RGB数值
		fclose(fp);
	}

	void loadXYZ(char *fname)
	{
		FILE *fp = fopen(fname, "rb");
		fread(&_cx, sizeof(int), 1, fp);
		fread(&_cy, sizeof(int), 1, fp);

		int sz = _cx * _cy * 3;
		_xyzs = new float[sz];
		fread(_xyzs, sizeof(float), sz, fp);	//读入所有深度数值
		fclose(fp);
	}

	int width() const { return _cx; }	//图像的长和宽
	int height() const { return _cy; }

	virtual void saveAsBmp(float *ptr, char *fn) {
		int sz = _cx * _cy * 3;

		BYTE *idx = new BYTE[sz];
		for (int i = 0; i < sz; ) {
			idx[i] = ptr[i + 2] * 255;
			idx[i + 1] = ptr[i + 1] * 255;
			idx[i + 2] = ptr[i] * 255;

			i += 3;
		}

		if (!idx)
			return;

		int colorTablesize = 0;

		int biBitCount = 24;

		if (biBitCount == 8)
			colorTablesize = 1024;

		//待存储图像数据每行字节数为4的倍数
		int lineByte = (_cx * biBitCount / 8 + 3) / 4 * 4;

		//以二进制写的方式打开文件
		FILE *fp = fopen(fn, "wb");
		if (fp == 0)
			return;

		//申请位图文件头结构变量，填写文件头信息
		BITMAPFILEHEADER fileHead;
		fileHead.bfType = 0x4D42;//bmp类型
								 //bfSize是图像文件4个组成部分之和
		fileHead.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + colorTablesize + lineByte * _cy;
		fileHead.bfReserved1 = 0;
		fileHead.bfReserved2 = 0;

		//bfOffBits是图像文件前3个部分所需空间之和
		fileHead.bfOffBits = 54 + colorTablesize;

		//写文件头进文件
		fwrite(&fileHead, sizeof(BITMAPFILEHEADER), 1, fp);

		//申请位图信息头结构变量，填写信息头信息
		BITMAPINFOHEADER head;
		head.biBitCount = biBitCount;
		head.biClrImportant = 0;
		head.biClrUsed = 0;
		head.biCompression = 0;
		head.biHeight = _cy;
		head.biPlanes = 1;
		head.biSize = 40;
		head.biSizeImage = lineByte * _cy;
		head.biWidth = _cx;
		head.biXPelsPerMeter = 0;
		head.biYPelsPerMeter = 0;

		//写位图信息头进内存
		fwrite(&head, sizeof(BITMAPINFOHEADER), 1, fp);

		//写位图数据进文件
		fwrite(idx, _cy*lineByte, 1, fp);

		//关闭文件
		fclose(fp);

		delete[] idx;
	}

	void resetRGBnew() {	//暂时不懂
		if (_rgbsNew) delete[] _rgbsNew;
		_rgbsNew = new float[_cx*_cy * 3];

		for (int i = 0; i < _cx; i++) {
			for (int j = 0; j < _cy; j++) {
				float *p2 = _rgbsNew + (i*_cy + j) * 3;
				p2[0] = 1;
				p2[1] = 0;
				p2[2] = 0;
			}
		}
	}

public:
	SceneData() {
		_cx = _cy = 0;
		_xyzs = _rgbs = NULL;
		_rgbsNew = NULL;
	}

	~SceneData() {
		if (_xyzs) delete[] _xyzs;
		if (_rgbs) delete[] _rgbs;
		if (_rgbsNew) delete[] _rgbsNew;
	}

	float *rgbs() { return _rgbs; }

	void saveNewRGB(char *fn) {
		saveAsBmp(_rgbsNew, fn);
	}

	void load(char *fname, int numofImg, bool updateHash = true) {
		char rgbFile[512], xyzFile[512];
		sprintf(rgbFile, "%s%s", fname, ".rgb");
		sprintf(xyzFile, "%s%s", fname, ".xyz");

		if (updateHash)
			loadRGB(rgbFile);	//加载RGB文件

		loadXYZ(xyzFile);	//加载深度文件

		resetRGBnew();

		float *p1 = _xyzs;
		float *p2 = _rgbs;
		int num = _cx * _cy;	//总像素值

		for (int i = 0; i < num; i++) {
			if (updateHash)
				gHashTab.push_back(xyz2rgb(vec3f(p1), vec3f(p2), numofImg, i));	//深度和RGB的对应

			_bound += vec3f(p1);
			p1 += 3;	//可以把这个当下标看
			p2 += 3;
		}
	}
};


class ProgScene : public SceneData {
public:
	float *_nearest;

	void resetNearest()
	{
		int sz = _cx * _cy;
		_nearest = new float[sz];

		for (int i = 0; i < sz; i++)
			_nearest[i] = 1000000;	//初始化每个点的最近距离
	}

public:
	ProgScene() {
		_nearest = NULL;
	}

	~ProgScene() {
		if (_nearest)
			delete[] _nearest;
	}


	void load(char *fname) {
		SceneData::load(fname, -1, false);
		resetNearest();
	}

	void save(char *fname) {
		saveNewRGB(fname);
	}

	void update() {

#pragma omp parallel for schedule(dynamic, 5)
		for (int i = 0; i < _cx; i++) {
			printf("%d of %d done...\n", i, _cx);

			for (int j = 0; j < _cy; j++) {
				float *p1 = _xyzs + (i*_cy + j) * 3;	//_xyzs里面是orin每个点的深度，每个点有xyz三个值
				float *p2 = _rgbsNew + (i*_cy + j) * 3;	//_rgbsNew里面是result里每个点的rgb值，也是一个点三个值
				float *p3 = _nearest + (i*_cy + j);		//_nearest是每个点和离它最近的点的距离
				float *p4 = _rgbs + (i*_cy + j) * 3;	//_rgbs里面是orin每个点的rgb值，也是一个点三个值

				int idx, pidx;	//分别是一张图上的位置索引和该张图的图片索引

				vec3f cr;
				bool ret = getColor(vec3f(p1), p3, cr, idx, pidx);
				if (ret) {
					p2[0] = cr.x;
					p2[1] = cr.y;
					p2[2] = cr.z;
				}
			}
		}

	}
};

SceneData scene[18];
ProgScene target;

__device__ bool findNearest2(xyz2rgb &org, xyz2rgb &ret, t_linearBVH *dev_lTree)
{
	return dev_lTree->query(org, ret);
}

//version 2.0
//******************************************
/*
__device__ vec3f traverseRecursive(t_linearBVH *bvh,xyz2rgb input){

	t_linearBvhNode* stack[32];
	t_linearBvhNode** stackPtr = stack;
	*stackPtr++ = NULL; // push
	vec3f input_pos = input._xyz;
	
	t_linearBvhNode* root = bvh->_root;


	//t_linearBvhNode* bvh_start = bvh - numObject;
	kBOX point(input._xyz);
	point.dilate(1.5f);
	
	// Traverse nodes starting from the root.
	t_linearBvhNode* node = root;
	float Nearest = 2000;
	int Nearest_id = -1;
	vec3f rgb = vec3f(1, 0, 0);

	do
	{

		int childL = node->_left;
		int childR = node->_right;
		t_linearBvhNode left = root[childL];
		t_linearBvhNode right = root[childR];
		bool overlapL = (left._box.overlaps(point));
		bool overlapR = (right._box.overlaps(point));

		if (overlapL && left.isLeaf())
		{
			if (point.inside(left._item.xyz())) {
				float dist = vdistance(left._item.xyz(), input_pos);
				
				if (dist < Nearest) {
					Nearest = dist;
					Nearest_id = childL;
				}
			}
		}

		if (overlapR && right.isLeaf())
		{
			if (point.inside(right._item.xyz())) {
				float dist = vdistance(right._item.xyz(), input_pos);
				if (dist < Nearest) {
					Nearest = dist;
					Nearest_id = childR;
				}
			}
		}

		bool traverseL = (overlapL && !left.isLeaf());
		bool traverseR = (overlapR && !right.isLeaf());

		if (!traverseL && !traverseR)
			node = *--stackPtr; // pop
		else
		{
			if (childL == 0 || childR == 0) {
				printf("error\n");
			}
			int id= (traverseL) ? childL : childR;
			node = root + id;;
			if (traverseL && traverseR)
				*stackPtr++ = root+childR; // push
		}
	} while (node != NULL);
	if (Nearest_id > 0) {
		rgb = root[Nearest_id]._item.rgb();
	}
	return rgb;
}

__device__ vec3f getColor(vec3f xyz, float *nearest, vec3f cr, int idx, int pidx, t_linearBVH *dev_lTree) {
	
	xyz2rgb input(xyz, vec3f(), idx, pidx), ret;

	return	traverseRecursive(dev_lTree, input);
}

__global__ void CudaGetColor(float *_xyzs, float *_rgbsNew, float *_nearest, t_linearBVH *dev_lTree) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	float *p1 = _xyzs + tid * 3;
	float *p2 = _rgbsNew + tid * 3;
	float *p3 = _nearest + tid;

	int idx, pidx;
	vec3f cr;

	cr = getColor(vec3f(p1), p3, cr, idx, pidx, dev_lTree);
	p2[0] = cr.x;
	p2[1] = cr.y;
	p2[2] = cr.z;
}

*/
//******************************************

//version 1.0
//******************************************
__device__ bool getColor(vec3f &xyz, float *nearest, vec3f &cr, int &idx, int &pidx, t_linearBVH *dev_lTree) {

	xyz2rgb input(xyz, vec3f(), idx, pidx), ret;

	//给input找xyz最近点
	bool find = findNearest2(input, ret, dev_lTree);
	if (!find)
		return false;

	vec3f now = ret.xyz();
	double dist = vdistance(xyz, now);
	if (dist < *nearest) {
		*nearest = dist;
		cr = ret.rgb();
		idx = ret.index();
		pidx = ret.pos();

		return true;
	}
	else
		return false;
}

__global__ void CudaGetColor(float *_xyzs, float *_rgbsNew, float *_nearest, t_linearBVH *dev_lTree) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	float *p1 = _xyzs + tid * 3;
	float *p2 = _rgbsNew + tid * 3;
	float *p3 = _nearest + tid;
	
	int idx, pidx;
	vec3f cr;
	
	bool ret = getColor(vec3f(p1), p3, cr, idx, pidx, dev_lTree);

	if (ret) {
		p2[0] = cr.x;
		p2[1] = cr.y;
		p2[2] = cr.z;
	}
}
//******************************************


int main() {
	//计时用
	cudaEvent_t start, stop;
	float elapsedTime;

	TIMING_BEGIN("Start loading...")
		target.load("all.bmp");	//target里是没有加载rgb进来的，不要搞错了
	scene[0].load("0-55.bmp", 0);//这个才是加载了xyz和rgb的
	TIMING_END("Loading done...")

	TIMING_BEGIN("Start build_bvh...")
		build_bvh();
	TIMING_END("build_bvh done...")

	//把线性BVH树复制给GPU
	t_linearBvhNode* _root_gpu;
	checkCudaErrors(cudaMalloc((void**)&_root_gpu, sizeof(t_linearBvhNode) * (1024 * 1024 * 2 + 1)));
	checkCudaErrors(cudaMemcpy(_root_gpu, lTree->_root, sizeof(t_linearBvhNode) * (1024 * 1024 * 2 + 1), cudaMemcpyHostToDevice));
	lTree->_root = _root_gpu;

	t_linearBVH* dev_lTree = 0;
	checkCudaErrors(cudaMalloc((void**)&dev_lTree, sizeof(t_linearBVH)));
	checkCudaErrors(cudaMemcpy(dev_lTree, lTree, sizeof(t_linearBVH), cudaMemcpyHostToDevice));

	//把颜色和深度信息复制给GPU
	int N = target._cx * target._cy * 3;
	float* dev_xyzs = 0;
	float* dev_rgbsNew = 0;
	float* dev_nearest = 0;

	checkCudaErrors(cudaMalloc((void**)&dev_xyzs, N * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dev_rgbsNew, N * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dev_nearest, N / 3 * sizeof(float)));

	checkCudaErrors(cudaMemcpy(dev_xyzs, target._xyzs, N * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_rgbsNew, target._rgbsNew, N * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_nearest, target._nearest, N / 3 * sizeof(float), cudaMemcpyHostToDevice));

	CUDA_TIMING_BEGIN("Start querying...")
		CudaGetColor << < 2048, 512 >> > (dev_xyzs, dev_rgbsNew, dev_nearest, dev_lTree);
	checkCudaErrors(cudaDeviceSynchronize());
	CUDA_TIMING_END("Querying done...")

	float *new_rgbs= new float[1024 * 1024 * 3];
	checkCudaErrors(cudaMemcpy(new_rgbs, dev_rgbsNew, N * sizeof(float), cudaMemcpyDeviceToHost));
	target._rgbsNew = new_rgbs;

	printf("end");

	destroy_bvh();
	target.save("output2.bmp");

	return 0;
}