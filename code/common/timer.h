//**************************************************************************************
//  Copyright (C) 2019 - 2022, Min Tang
//  All rights reserved.
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//     1. Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//     2. Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in the
//        documentation and/or other materials provided with the distribution.
//     3. The names of its contributors may not be used to endorse or promote
//        products derived from this software without specific prior written
//        permission.
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//**************************************************************************************

#pragma once

#include <omp.h>

struct Timer {
	double last, total;

public:
	void tick() {
		last = omp_get_wtime();
	}
	
	void tock() {
		double now = omp_get_wtime();
		total = now - last;
	}
	
	void output(char *msg) {
		tm_printf(L"%hs: %3.5f s\n", msg, total);
	}
};


# define	TIMING_BEGIN(message) \
{tm_printf(L"%hs\n", message); Timer _c; _c.tick();

# define	TIMING_END(message) \
{_c.tock(); _c.output(message);}}

# define CUDA_TIMING_BEGIN(message) \
{printf("%s\n",message);\
cudaEventCreate(&start); \
cudaEventCreate(&stop);  \
cudaEventRecord(start,0);}

#define CUDA_TIMING_END(message) \
{cudaEventRecord(stop,0); \
cudaEventSynchronize(stop); \
cudaEventElapsedTime(&elapsedTime,start,stop); \
printf("%s elapsed time: %f ms\n",message,elapsedTime);\
cudaEventDestroy(start);\
cudaEventDestroy(stop);}

