#include "Drawing.h"
#include "Ray.h"

#ifdef CUDA

#include <math.h>
#include <Windows.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define SPHC 1
#define TRIS 2
#define LIGHTS 1

#define THRCOUNT 8

//Point camera = Point(0, 0, -2.0f);
Sphere spheres[SPHC];
Point lights[LIGHTS];
Triangle triangles[TRIS];
float angle = 0;
char *imgptr, *devImgPtr;
Sphere *devSpheres;
Point *devLights;
Triangle *devTriangles;

void InitFrame()
{
	spheres[0] = Sphere(Point(sinf(angle) * 3, -2, 10 + cosf(angle) * 3), 1);
	angle += 0.01;
	//spheres[1] = Sphere(Point(0, -1000, 10), 995);
	lights[0] = Point(2, 2, 10);
	triangles[0] = Triangle(Point(10, -2, 0), Point(-10, -2, 0), Point(10, -2, 20));
	triangles[1] = Triangle(Point(-10, -2, 0), Point(-10, -2, 20), Point(10, -2, 20));
	//lights[1] = Point(1000, 0, 0);
}

__device__ float pointLit(Point &p, Vector n, void* self, Point *lights, Sphere *spheres, Triangle *triangles) {
	Ray ray;
	float lit = 0, t;
	bool col;
	for (int i = 0; i < LIGHTS; i++) {
		ray = Ray(p, lights[i]);
		if (n * ray.d > 0) {
			col = false;
			for (int j = 0; j < SPHC; j++) {
				if (spheres + j != self && ray.intersects(spheres[j], &t) && t > 0.001) {
					col = true;
					break;
				}
			}
			if (!col) {
				for (int j = 0; j < TRIS; j++) {
					if (triangles + j != self && ray.intersects(triangles[j], &t) && t > 0.001) {
						col = true;
						break;
					}
				}
			}
			if (!col) {
				lit += n * ray.d;
			}
		}
	}
	return lit;
}


__global__ void drawPixelCUDA(char* ptr, Point *lights, Sphere *spheres, Triangle *triangles) {
	int xi = blockIdx.x * THRCOUNT + threadIdx.x;
	int yi = blockIdx.y * THRCOUNT + threadIdx.y;

	if (xi > XRES || yi > YRES) return;

	float x = xi * 2.0f / YRES - XRES / (float)YRES;
	float y = yi * 2.0 / YRES - 1.0;

	char *pix = ptr + (yi * XRES + xi) * 3;

	Point pixelPoint(x, y, 0);

	Point camera = Point(0, 0, -2.0f);
	Vector normal;
	void *obj;

	Ray ray = Ray(camera, pixelPoint);

	float t1, t2, light, nearest = INFINITY;
	Point colPoint;

	for (int i = 0; i < SPHC; i++) {
		if (ray.intersects(spheres[i], &t1, nullptr)) {
			if (t1 < nearest) {
				nearest = t1;
				colPoint = ray.getPointFromT(t1);
				normal = spheres[i].Normal(colPoint);
				obj = spheres + i;
			}
		}
	}

	for (int i = 0; i < TRIS; i++) {
		if (ray.intersects(triangles[i], &t1)) {
			if (t1 < nearest) {
				nearest = t1;
				colPoint = ray.getPointFromT(t1);
				normal = triangles[i].n;
				obj = triangles + i;
			}
		}
	}

	if (nearest < INFINITY) {
		light = pointLit(colPoint, normal, obj, lights, spheres, triangles);
		pix[0] = 50 * light;
		pix[1] = 200 * light;
		pix[2] = 100 * light;
	}
	else{
		pix[0] = 40;
		pix[1] = 120;
		pix[2] = 240;

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

	cudaStatus = cudaMalloc((void**)&devTriangles, TRIS * sizeof(Triangle));
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

	cudaStatus = cudaMemcpy(devTriangles, triangles, TRIS * sizeof(Triangle), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
		return;
	}

	dim3 thrds(THRCOUNT, THRCOUNT);
	dim3 blocks(XRES / THRCOUNT, YRES / THRCOUNT);

	drawPixelCUDA << <blocks, thrds >> > (devImgPtr, devLights, devSpheres, devTriangles);

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
