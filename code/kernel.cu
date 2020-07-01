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

	//��input��xyz�����
	bool find = findNearest2(input, ret);
	if (!find)
		return false;

	vec3f now = ret.xyz();
	double dist = vdistance(xyz, now);
	if (dist < *nearest) {
		//�������������˵������������滻�������ľ���
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
	float *_xyzs;	//���ͼ����
	float *_rgbs;	//��������RGB����

	float *_rgbsNew;	//�ؽ�����ͼ��RGB����
	BOX _bound;

public:
	void loadRGB(char *fname)
	{
		FILE *fp = fopen(fname, "rb");
		//buffer�׸������ָ�룬ÿ������Ĵ�С�����������������
		fread(&_cx, sizeof(int), 1, fp);
		fread(&_cy, sizeof(int), 1, fp);

		int sz = _cx * _cy * 3;	//ͼ��ĳ��˿����ͨ��
		_rgbs = new float[sz];
		fread(_rgbs, sizeof(float), sz, fp);	//��������RGB��ֵ
		fclose(fp);
	}

	void loadXYZ(char *fname)
	{
		FILE *fp = fopen(fname, "rb");
		fread(&_cx, sizeof(int), 1, fp);
		fread(&_cy, sizeof(int), 1, fp);

		int sz = _cx * _cy * 3;
		_xyzs = new float[sz];
		fread(_xyzs, sizeof(float), sz, fp);	//�������������ֵ
		fclose(fp);
	}

	int width() const { return _cx; }	//ͼ��ĳ��Ϳ�
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

		//���洢ͼ������ÿ���ֽ���Ϊ4�ı���
		int lineByte = (_cx * biBitCount / 8 + 3) / 4 * 4;

		//�Զ�����д�ķ�ʽ���ļ�
		FILE *fp = fopen(fn, "wb");
		if (fp == 0)
			return;

		//����λͼ�ļ�ͷ�ṹ��������д�ļ�ͷ��Ϣ
		BITMAPFILEHEADER fileHead;
		fileHead.bfType = 0x4D42;//bmp����
								 //bfSize��ͼ���ļ�4����ɲ���֮��
		fileHead.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + colorTablesize + lineByte * _cy;
		fileHead.bfReserved1 = 0;
		fileHead.bfReserved2 = 0;

		//bfOffBits��ͼ���ļ�ǰ3����������ռ�֮��
		fileHead.bfOffBits = 54 + colorTablesize;

		//д�ļ�ͷ���ļ�
		fwrite(&fileHead, sizeof(BITMAPFILEHEADER), 1, fp);

		//����λͼ��Ϣͷ�ṹ��������д��Ϣͷ��Ϣ
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

		//дλͼ��Ϣͷ���ڴ�
		fwrite(&head, sizeof(BITMAPINFOHEADER), 1, fp);

		//дλͼ���ݽ��ļ�
		fwrite(idx, _cy*lineByte, 1, fp);

		//�ر��ļ�
		fclose(fp);

		delete[] idx;
	}

	void resetRGBnew() {	//��ʱ����
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
			loadRGB(rgbFile);	//����RGB�ļ�

		loadXYZ(xyzFile);	//��������ļ�

		resetRGBnew();

		float *p1 = _xyzs;
		float *p2 = _rgbs;
		int num = _cx * _cy;	//������ֵ

		for (int i = 0; i < num; i++) {
			if (updateHash)
				gHashTab.push_back(xyz2rgb(vec3f(p1), vec3f(p2), numofImg, i));	//��Ⱥ�RGB�Ķ�Ӧ

			_bound += vec3f(p1);
			p1 += 3;	//���԰�������±꿴
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
			_nearest[i] = 1000000;	//��ʼ��ÿ������������
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
				float *p1 = _xyzs + (i*_cy + j) * 3;	//_xyzs������orinÿ�������ȣ�ÿ������xyz����ֵ
				float *p2 = _rgbsNew + (i*_cy + j) * 3;	//_rgbsNew������result��ÿ�����rgbֵ��Ҳ��һ��������ֵ
				float *p3 = _nearest + (i*_cy + j);		//_nearest��ÿ�������������ĵ�ľ���
				float *p4 = _rgbs + (i*_cy + j) * 3;	//_rgbs������orinÿ�����rgbֵ��Ҳ��һ��������ֵ

				int idx, pidx;	//�ֱ���һ��ͼ�ϵ�λ�������͸���ͼ��ͼƬ����

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

	//��input��xyz�����
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
	//��ʱ��
	cudaEvent_t start, stop;
	float elapsedTime;

	TIMING_BEGIN("Start loading...")
		target.load("all.bmp");	//target����û�м���rgb�����ģ���Ҫ�����
	scene[0].load("0-55.bmp", 0);//������Ǽ�����xyz��rgb��
	TIMING_END("Loading done...")

	TIMING_BEGIN("Start build_bvh...")
		build_bvh();
	TIMING_END("build_bvh done...")

	//������BVH�����Ƹ�GPU
	t_linearBvhNode* _root_gpu;
	checkCudaErrors(cudaMalloc((void**)&_root_gpu, sizeof(t_linearBvhNode) * (1024 * 1024 * 2 + 1)));
	checkCudaErrors(cudaMemcpy(_root_gpu, lTree->_root, sizeof(t_linearBvhNode) * (1024 * 1024 * 2 + 1), cudaMemcpyHostToDevice));
	lTree->_root = _root_gpu;

	t_linearBVH* dev_lTree = 0;
	checkCudaErrors(cudaMalloc((void**)&dev_lTree, sizeof(t_linearBVH)));
	checkCudaErrors(cudaMemcpy(dev_lTree, lTree, sizeof(t_linearBVH), cudaMemcpyHostToDevice));

	//����ɫ�������Ϣ���Ƹ�GPU
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