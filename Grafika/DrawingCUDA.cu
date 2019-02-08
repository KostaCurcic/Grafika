#include "Drawing.h"
#include "Ray.h"

#ifdef CUDA

#include <math.h>
#include <Windows.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define SPHC 2
#define LIGHTS 1

#define THRCOUNT 8

//Point camera = Point(0, 0, -2.0f);
Sphere spheres[SPHC];
Point lights[LIGHTS];
float angle = 0;
char *imgptr, *devImgPtr;
Sphere *devSpheres;
Point *devLights;

void InitFrame()
{
	spheres[0] = Sphere(Point(sinf(angle) * 3, 0, 10 + cosf(angle) * 3), 1);
	angle += 0.01;
	spheres[1] = Sphere(Point(0, -1000, 10), 995);
	lights[0] = Point(2, 2, 10);
	//lights[1] = Point(1000, 0, 0);
}


__global__ void drawPixelCUDA(char* ptr, Point *lights, Sphere *spheres) {
	int xi = blockIdx.x * THRCOUNT + threadIdx.x;
	int yi = blockIdx.y * THRCOUNT + threadIdx.y;

	if (xi > XRES || yi > YRES) return;

	float x = xi * 2.0f / YRES - XRES / (float)YRES;
	float y = yi * 2.0 / YRES - 1.0;

	char *pix = ptr + (yi * XRES + xi) * 3;

	Point pixelPoint(x, y, 0);

	Point camera = Point(0, 0, -2.0f);
	Vector normal;

	Ray ray = Ray(camera, pixelPoint);
	Ray shadowRay;
	bool collided = false, lit = false, sCollided = false;

	float t1, t2;
	Point colPoint;

	for (int i = 0; i < SPHC; i++) {
		if (ray.intersects(spheres[i], &t1, &t2)) {
			if (t1 > t2 && t2 >= 0) {
				colPoint = ray.getPointFromT(t2);
			}
			else {
				colPoint = ray.getPointFromT(t1);
			}
			lit = false;
			for (int j = 0; j < LIGHTS; j++) {
				shadowRay = Ray(colPoint, lights[j]);
				sCollided = false;
				normal = spheres[i].Normal(colPoint);
				if (spheres[i].Normal(colPoint) * shadowRay.d > 0) {
					for (int s = 0; s < SPHC; s++) {
						if (s == i) continue;
						if (shadowRay.intersects(spheres[s], nullptr, nullptr)) {
							sCollided = true;
							break;
						}
					}


					if (!sCollided) {
						pix[0] = 50 * (normal * shadowRay.d);
						pix[1] = 200 * (normal * shadowRay.d);
						pix[2] = 100 * (normal * shadowRay.d);
						lit = true;
						break;
					}
				}
			}
			if (!lit) {
				pix[0] = 0;
				pix[1] = 0;
				pix[2] = 0;
			}
			collided = true;
			break;
		}
	}

	if (!collided) {
		pix[0] = 0;
		pix[1] = 0;
		pix[2] = 0;

	}

}

void InitDrawing(char * ptr)
{
	imgptr = ptr;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&devImgPtr, XRES * YRES * 3 * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		return;
	}

	cudaStatus = cudaMalloc((void**)&devSpheres, SPHC * sizeof(Sphere));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		return;
	}

	cudaStatus = cudaMalloc((void**)&devLights, LIGHTS * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		return;
	}
}

void DrawFrame()
{
	InitFrame();
	
	cudaError_t cudaStatus = cudaMemcpy(devSpheres, spheres, SPHC * sizeof(Sphere), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
		return;
	}

	cudaStatus = cudaMemcpy(devLights, lights, LIGHTS * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
		return;
	}

	dim3 thrds(THRCOUNT, THRCOUNT);
	dim3 blocks(XRES / THRCOUNT, YRES / THRCOUNT);

	drawPixelCUDA << <blocks, thrds >> > (devImgPtr, devLights, devSpheres);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(imgptr, devImgPtr, XRES * YRES * 3 * sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return;
	}
}
#endif
