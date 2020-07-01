#pragma once

#include <vector>
using namespace std;

#include "vec3f.h"
#include "xyz-rgb.h"
#include <stdlib.h>

#define MAX(a,b)	((a) > (b) ? (a) : (b))
#define MIN(a,b)	((a) < (b) ? (a) : (b))

class kDOP18 {
public:
	__host__ __device__ FORCEINLINE static void getDistances(const vec3f& p,
		double &d3, double &d4, double &d5, double &d6, double &d7, double &d8)
	{
		d3 = p[0] + p[1];
		d4 = p[0] + p[2];
		d5 = p[1] + p[2];
		d6 = p[0] - p[1];
		d7 = p[0] - p[2];
		d8 = p[1] - p[2];
	}

	__host__ __device__ FORCEINLINE static void getDistances(const vec3f& p, double d[])
	{
		d[0] = p[0] + p[1];
		d[1] = p[0] + p[2];
		d[2] = p[1] + p[2];
		d[3] = p[0] - p[1];
		d[4] = p[0] - p[2];
		d[5] = p[1] - p[2];
	}

	FORCEINLINE static double getDistances(const vec3f &p, int i)
	{
		if (i == 0) return p[0] + p[1];
		if (i == 1) return p[0] + p[2];
		if (i == 2) return p[1] + p[2];
		if (i == 3) return p[0] - p[1];
		if (i == 4) return p[0] - p[2];
		if (i == 5) return p[1] - p[2];
		return 0;
	}

public:
	double _dist[18];
	__host__ __device__
	FORCEINLINE kDOP18() {
		empty();
	}

	__host__ __device__ FORCEINLINE kDOP18(const vec3f &v) {
		_dist[0] = _dist[9] = v[0];
		_dist[1] = _dist[10] = v[1];
		_dist[2] = _dist[11] = v[2];

		double d3, d4, d5, d6, d7, d8;
		getDistances(v, d3, d4, d5, d6, d7, d8);
		_dist[3] = _dist[12] = d3;
		_dist[4] = _dist[13] = d4;
		_dist[5] = _dist[14] = d5;
		_dist[6] = _dist[15] = d6;
		_dist[7] = _dist[16] = d7;
		_dist[8] = _dist[17] = d8;
	}

	FORCEINLINE kDOP18(const vec3f &a, const vec3f &b) {
		_dist[0] = MIN(a[0], b[0]);
		_dist[9] = MAX(a[0], b[0]);
		_dist[1] = MIN(a[1], b[1]);
		_dist[10] = MAX(a[1], b[1]);
		_dist[2] = MIN(a[2], b[2]);
		_dist[11] = MAX(a[2], b[2]);

		double ad3, ad4, ad5, ad6, ad7, ad8;
		getDistances(a, ad3, ad4, ad5, ad6, ad7, ad8);
		double bd3, bd4, bd5, bd6, bd7, bd8;
		getDistances(b, bd3, bd4, bd5, bd6, bd7, bd8);
		_dist[3] = MIN(ad3, bd3);
		_dist[12] = MAX(ad3, bd3);
		_dist[4] = MIN(ad4, bd4);
		_dist[13] = MAX(ad4, bd4);
		_dist[5] = MIN(ad5, bd5);
		_dist[14] = MAX(ad5, bd5);
		_dist[6] = MIN(ad6, bd6);
		_dist[15] = MAX(ad6, bd6);
		_dist[7] = MIN(ad7, bd7);
		_dist[16] = MAX(ad7, bd7);
		_dist[8] = MIN(ad8, bd8);
		_dist[17] = MAX(ad8, bd8);
	}

	__host__ __device__ FORCEINLINE bool overlaps(const kDOP18 b) const
	{
		for (int i = 0; i<9; i++) {
			if (_dist[i] > b._dist[i + 9]) return false;
			if (_dist[i + 9] < b._dist[i]) return false;
		}

		return true;
	}

	FORCEINLINE bool overlaps(const kDOP18 &b, kDOP18 &ret) const
	{
		if (!overlaps(b))
			return false;

		for (int i = 0; i<9; i++) {
			ret._dist[i] = MAX(_dist[i], b._dist[i]);
			ret._dist[i + 9] = MIN(_dist[i + 9], b._dist[i + 9]);
		}
		return true;
	}

	__host__ __device__ FORCEINLINE bool inside(const vec3f p) const
	{
		for (int i = 0; i<3; i++) {
			if (p[i] < _dist[i] || p[i] > _dist[i + 9])
				return false;
		}

		double d[6];
		getDistances(p, d);
		for (int i = 3; i<9; i++) {
			if (d[i - 3] < _dist[i] || d[i - 3] > _dist[i + 9])
				return false;
		}

		return true;
	}

	FORCEINLINE kDOP18 &operator += (const vec3f &p)
	{
		_dist[0] = MIN(p[0], _dist[0]);
		_dist[9] = MAX(p[0], _dist[9]);
		_dist[1] = MIN(p[1], _dist[1]);
		_dist[10] = MAX(p[1], _dist[10]);
		_dist[2] = MIN(p[2], _dist[2]);
		_dist[11] = MAX(p[2], _dist[11]);

		double d3, d4, d5, d6, d7, d8;
		getDistances(p, d3, d4, d5, d6, d7, d8);
		_dist[3] = MIN(d3, _dist[3]);
		_dist[12] = MAX(d3, _dist[12]);
		_dist[4] = MIN(d4, _dist[4]);
		_dist[13] = MAX(d4, _dist[13]);
		_dist[5] = MIN(d5, _dist[5]);
		_dist[14] = MAX(d5, _dist[14]);
		_dist[6] = MIN(d6, _dist[6]);
		_dist[15] = MAX(d6, _dist[15]);
		_dist[7] = MIN(d7, _dist[7]);
		_dist[16] = MAX(d7, _dist[16]);
		_dist[8] = MIN(d8, _dist[8]);
		_dist[17] = MAX(d8, _dist[17]);

		return *this;
	}

	FORCEINLINE kDOP18 &operator += (const kDOP18 &b)
	{
		_dist[0] = MIN(b._dist[0], _dist[0]);
		_dist[9] = MAX(b._dist[9], _dist[9]);
		_dist[1] = MIN(b._dist[1], _dist[1]);
		_dist[10] = MAX(b._dist[10], _dist[10]);
		_dist[2] = MIN(b._dist[2], _dist[2]);
		_dist[11] = MAX(b._dist[11], _dist[11]);
		_dist[3] = MIN(b._dist[3], _dist[3]);
		_dist[12] = MAX(b._dist[12], _dist[12]);
		_dist[4] = MIN(b._dist[4], _dist[4]);
		_dist[13] = MAX(b._dist[13], _dist[13]);
		_dist[5] = MIN(b._dist[5], _dist[5]);
		_dist[14] = MAX(b._dist[14], _dist[14]);
		_dist[6] = MIN(b._dist[6], _dist[6]);
		_dist[15] = MAX(b._dist[15], _dist[15]);
		_dist[7] = MIN(b._dist[7], _dist[7]);
		_dist[16] = MAX(b._dist[16], _dist[16]);
		_dist[8] = MIN(b._dist[8], _dist[8]);
		_dist[17] = MAX(b._dist[17], _dist[17]);
		return *this;
	}

	FORCEINLINE kDOP18 operator + (const kDOP18 &v) const
	{
		kDOP18 rt(*this); return rt += v;
	}

	FORCEINLINE double length(int i) const {
		return _dist[i + 9] - _dist[i];
	}

	FORCEINLINE double width()  const { return _dist[9] - _dist[0]; }
	FORCEINLINE double height() const { return _dist[10] - _dist[1]; }
	FORCEINLINE double depth()  const { return _dist[11] - _dist[2]; }
	FORCEINLINE double volume() const { return width()*height()*depth(); }

	FORCEINLINE vec3f center() const {
		return vec3f(_dist[0] + _dist[9], _dist[1] + _dist[10], _dist[2] + _dist[11])*double(0.5);
	}

	FORCEINLINE double center(int i) const {
		return (_dist[i + 9] + _dist[i])*double(0.5);
	}
	__host__ __device__
	FORCEINLINE void empty() {
		for (int i = 0; i<9; i++) {
			_dist[i] = FLT_MAX;
			_dist[i + 9] = -FLT_MAX;
		}
	}

	__host__ __device__ FORCEINLINE void dilate(double d) {
		//static double sqrt2 = sqrt(2);
		//double sqrt2 = sqrt(2);
		//Dango:不想整，先随便给个数
		double sqrt2 = 1.4142135623731;
		for (int i = 0; i < 3; i++) {
			_dist[i] -= d;
			_dist[i + 9] += d;
		}
		for (int i = 0; i < 6; i++) {
			_dist[3 + i] -= sqrt2*d;
			_dist[3 + i + 9] += sqrt2*d;
		}
	}
};

#define kBOX kDOP18


class aap {
public:
	char _xyz;
	double _p;

	FORCEINLINE aap(const kBOX &total) {
		vec3f center = total.center();
		char xyz = 2;

		if (total.width() >= total.height() && total.width() >= total.depth()) {
			xyz = 0;
		}
		else
			if (total.height() >= total.width() && total.height() >= total.depth()) {
				xyz = 1;
			}

		_xyz = xyz;
		_p = center[xyz];
	}

	FORCEINLINE bool inside(const vec3f &mid) const {
		return mid[_xyz]>_p;
	}
};

class t_bvhTree;


class t_linearBvhNode {
public:
	kBOX _box;
	xyz2rgb _item;
	int _parent;
	int _left, _right;

	__host__ __device__ t_linearBvhNode() {
		_parent = _left = _right = -1;
	}

	__host__ __device__ FORCEINLINE bool isLeaf() { return _left == -1; }
	__host__ __device__ FORCEINLINE bool isRoot() { return _parent == -1; }

	void query(t_linearBvhNode *root, kBOX &bx, vector<xyz2rgb> &rets) {
		if (!_box.overlaps(bx))
			return;

		if (isLeaf()) {
			if (bx.inside(_item.xyz()))
				rets.push_back(_item);

			return;
		}

		root[_left].query(root, bx, rets);
		root[_right].query(root, bx, rets);
	}

	__host__ __device__ void query(t_linearBvhNode *root, kBOX &bx, xyz2rgb &dst, xyz2rgb &minPt, double &minDist) {

		if (!_box.overlaps(bx)) {
			return;
		}

		if (isLeaf()) {
			if (bx.inside(_item.xyz())) {
				double dist = vdistance(_item.xyz(), dst.xyz());
				if (dist < minDist) {
					minDist = dist;
					minPt =_item;
				}
			}
			return;
		}
		root[_left].query(root, bx, dst, minPt, minDist);
		root[_right].query(root, bx, dst, minPt, minDist);
	}

	//20191011 一个预计充满了bug的很迷的东西
	//dst是进来找最近点的那个点，bx是它扩张了的包围盒，minPt是要找到的点
	__host__ __device__ void query2(t_linearBvhNode *root, kBOX &bx, xyz2rgb &dst, xyz2rgb &minPt, double &minDist) {

		t_linearBvhNode _stack[32];
		double dist;
		int num_stack = 0;

		t_linearBvhNode pCur = root[0];

		while (!(pCur.isLeaf() && (num_stack == 0))) {

			//如果在这个范围内，判断一下是不是叶子节点，是就操作一下
			if (pCur._box.overlaps(bx)) {
				if (pCur.isLeaf()) {
					if (bx.inside(pCur._item.xyz())) {
						dist = vdistance(pCur._item.xyz(), dst.xyz());
						if (dist < minDist) {
							minDist = dist;
							minPt = pCur._item;
						}
					}
					//去找找右儿子们
					if (!(num_stack == 0)) {
						pCur = _stack[num_stack - 1];
						num_stack--;
						pCur = root[pCur._right];
					}
					else return;
				}//如果不是叶子节点就要继续查找它的儿子
				else {
					_stack[num_stack] = pCur;
					num_stack++;
					pCur = root[pCur._left];
				}
			}
			else {
				if (!(num_stack == 0)) {
					pCur = _stack[num_stack - 1];
					num_stack--;
					pCur = root[pCur._right];
				}
				else return;
			}

		}
		return;
	}

	//20191022 传说用int存储会变快
	__host__ __device__ void query3(t_linearBvhNode *root, kBOX &bx, xyz2rgb &dst, xyz2rgb &minPt, double &minDist) {

		//t_linearBvhNode _stack[32];
		int _stack[32];
		int now_index = 0;

		double dist;
		int num_stack = 0;

		t_linearBvhNode pCur = root[0];

		while (!(pCur.isLeaf() && (num_stack == 0))) {

			//如果在这个范围内，判断一下是不是叶子节点，是就操作一下
			if (pCur._box.overlaps(bx)) {
				if (pCur.isLeaf()) {
					if (bx.inside(pCur._item.xyz())) {
						dist = vdistance(pCur._item.xyz(), dst.xyz());
						if (dist < minDist) {
							minDist = dist;
							minPt = pCur._item;
						}
					}
					//去找找右儿子们
					if (!(num_stack == 0)) {
						now_index = _stack[num_stack - 1];
						num_stack--;
						pCur = root[now_index];
						now_index = pCur._right;
						pCur = root[now_index];
					}
					else return;
				}//如果不是叶子节点就要继续查找它的儿子
				else {
					_stack[num_stack] = now_index;
					num_stack++;
					now_index = pCur._left;
					pCur = root[now_index];
				}
			}
			else {
				if (!(num_stack == 0)) {
					now_index = _stack[num_stack - 1];
					num_stack--;
					pCur = root[now_index];
					now_index = pCur._right;
					pCur = root[now_index];
				}
				else return;
			}

		}
		return;
	}
};

class t_bvhNode {
public:
	kBOX _box;
	xyz2rgb _item;
	t_bvhNode *_parent;
	t_bvhNode *_left;
	t_bvhNode *_right;


	void Construct(t_bvhNode *p, xyz2rgb &pt)
	{
		_parent = p;
		_left = _right = NULL;
		_item = pt;
		_box += pt.xyz();
	}

public:
	t_bvhNode() {
		_parent = _left = _right = NULL;
	}

	t_bvhNode(t_bvhNode *p, xyz2rgb &pt) {
		Construct(p, pt);
	}

	t_bvhNode(t_bvhNode *p, vector<xyz2rgb>&pts)
	{
		if (pts.size() == 1) {
			Construct(p, pts[0]);
			return;
		}

		_parent = p;
		int num = pts.size();
		for (int i = 0; i < num; i++)
			_box += pts[i].xyz();

		if (num == 2) {
			_left = new t_bvhNode(this, pts[0]);
			_right = new t_bvhNode(this, pts[1]);
			return;
		}

		aap pln(_box);
		vector<xyz2rgb> left, right;

		for (int i = 0; i < num; i++) {
			xyz2rgb &pt = pts[i];

			if (pln.inside(pt.xyz()))
				left.push_back(pt);
			else
				right.push_back(pt);
		}

		if (left.size() == 0)
		{
			left = std::vector<xyz2rgb>(
				std::make_move_iterator(right.begin() + right.size() / 2),
				std::make_move_iterator(right.end()));
			right.erase(right.begin() + right.size() / 2, right.end());
		}
		else if (right.size() == 0) {
			right = std::vector<xyz2rgb>(
				std::make_move_iterator(left.begin() + left.size() / 2),
				std::make_move_iterator(left.end()));
			left.erase(left.begin() + left.size() / 2, left.end());
		}

		_left = new t_bvhNode(this, left);
		_right = new t_bvhNode(this, right);
	}

	~t_bvhNode() {
		if (_left) delete _left;
		if (_right) delete _right;
	}

	FORCEINLINE t_bvhNode *getLeftChild() { return _left; }
	FORCEINLINE t_bvhNode *getRightChild() { return _right; }
	FORCEINLINE t_bvhNode *getParent() { return _parent; }

	FORCEINLINE bool getItem(xyz2rgb &item) {
		if (isLeaf()) {
			item = _item;
			return true;
		}
		else
			return false;
	}

	FORCEINLINE bool isLeaf() { return _left == NULL; }
	FORCEINLINE bool isRoot() { return _parent == NULL; }

	void query(kBOX &bx, vector<xyz2rgb> &rets) {
		if (!_box.overlaps(bx))
			return;

		if (isLeaf()) {
			if (bx.inside(_item.xyz()))
				rets.push_back(_item);

			return;
		}

		_left->query(bx, rets);
		_right->query(bx, rets);
	}

	int store(t_linearBvhNode *data, int &idx, int pid) {
		int current = idx;

		data[current]._box = _box;
		data[current]._item = _item;
		data[current]._parent = pid;
		idx++;

		if (_left)
			data[current]._left = _left->store(data, idx, current);

		if (_right)
			data[current]._right = _right->store(data, idx, current);

		return current;
	}

	friend class t_bvhTree;
};

class t_linearBVH;

class t_bvhTree {
	t_bvhNode	*_root;
	int _numItems;
	int _maxPts;

	void Construct(vector<xyz2rgb> &data) {
		_numItems = data.size();

		kBOX total;
		for (int i = 0; i < data.size(); i++)
			total += data[i].xyz();

		aap pln(total);
		vector<xyz2rgb> left, right;
		for (int i = 0; i < data.size(); i++) {
			xyz2rgb &pt = data[i];

			if (pln.inside(pt.xyz()))
				left.push_back(pt);
			else
				right.push_back(pt);
		}

		_root = new t_bvhNode;
		_root->_box = total;

		if (data.size() == 1) {
			_root->_item = data[0];
			_root->_parent = NULL;
			_root->_left = _root->_right = NULL;
		}
		else {
			if (left.size() == 0)
			{
				left = std::vector<xyz2rgb>(
					std::make_move_iterator(right.begin() + right.size() / 2),
					std::make_move_iterator(right.end()));
				right.erase(right.begin() + right.size() / 2, right.end());
			}
			else if (right.size() == 0) {
				right = std::vector<xyz2rgb>(
					std::make_move_iterator(left.begin() + left.size() / 2),
					std::make_move_iterator(left.end()));
				left.erase(left.begin() + left.size() / 2, left.end());
			}

			_root->_left = new t_bvhNode(_root, left);
			_root->_right = new t_bvhNode(_root, right);
		}
	}

public:
	t_bvhTree(vector<xyz2rgb> &data) {
		_maxPts = 0;

		if (data.size())
			Construct(data);
		else
			_root = NULL;
	}

	~t_bvhTree() {
		if (!_root)
			return;

		delete _root;
	}

	int maxPts() const { return _maxPts; }

	bool query(xyz2rgb &pt, xyz2rgb &ret)
	{
		kBOX bound(pt.xyz());
		bound.dilate(1.5);

		vector<xyz2rgb> rets;
		query(bound, rets);
		//printf("find %d proximities\n", rets.size());
		if (rets.size() > _maxPts)
			_maxPts = rets.size();

		double minDist = 10000000;
		for (int i = 0; i < rets.size(); i++) {
			double dist = vdistance(rets[i].xyz(), pt.xyz());
			if (dist < minDist) {
				minDist = dist;
				ret = rets[i];
			}
		}

		return rets.size() > 0;
	}

	void query(kBOX &bx, vector<xyz2rgb> &rets)
	{
		_root->query(bx, rets);
	}

	kBOX box() {
		if (_root)
			return getRoot()->_box;
		else
			return kBOX();
	}

	FORCEINLINE t_bvhNode *getRoot() { return _root; }
	
	friend class t_bvhNode;
	friend class t_linearBVH;
};


class t_linearBVH {
	
	int _num;	//2097153(1024×1024×2+1）
	int _current;

public:
	t_linearBvhNode *_root;
	__host__ __device__ int num() const { return _num; }
	void *data() { return _root; }

public:
	t_linearBVH(t_bvhTree *tree) {
		_num = tree->_numItems * 2 + 1;
		_root = new t_linearBvhNode[_num];
		_current = 0; //put from here

		tree->_root->store(_root, _current, -1);
	}

	void query(kBOX &bx, vector<xyz2rgb> &rets)
	{
		_root->query(_root, bx, rets);
	}

	__host__ __device__ void query(kBOX &bx, xyz2rgb &dst, xyz2rgb &minPt, double &minDist)
	{
		//20191011 试试看呗
		//原始版本
		//_root->query(_root, bx, dst, minPt, minDist);

		//stack存Node版本
		//_root->query2(_root, bx, dst, minPt, minDist);

		//stack存int版本
		_root->query3(_root, bx, dst, minPt, minDist);
	}

	__host__ __device__ bool query(xyz2rgb &pt, xyz2rgb &ret)
	{
		kBOX bound(pt.xyz());
		bound.dilate(1.5);
		
		double minDist = 1000000;
		query(bound, pt, ret, minDist);
		
		return minDist < 1000000;
	}
};
