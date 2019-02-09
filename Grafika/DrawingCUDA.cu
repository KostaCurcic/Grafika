#include "Drawing.h"
#include "Ray.h"

#ifdef CUDA

#include <math.h>
#include <Windows.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define SPHC 2
#define TRIS 3
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
	spheres[0] = Sphere(Point(sinf(angle) * 3, -1, 10 + cosf(angle) * 3), 1);
	spheres[0].mirror = true;

	spheres[1] = Sphere(Point(5, -1, 5), 1);
	spheres[1].color.r = 50;
	spheres[1].color.g = 200;
	spheres[1].color.b = 100;

	lights[0] = Point(2, 2, 10);
	triangles[0] = Triangle(Point(10, -2, 0), Point(-10, -2, 0), Point(10, -2, 20));
	triangles[1] = Triangle(Point(-10, -2, 0), Point(-10, -2, 20), Point(10, -2, 20));

	triangles[2] = Triangle(Point(-6, 2, 6), Point(-5, -2, 8), Point(-5, -5, 4));
	triangles[2].color.r = 240;

	angle += 0.01;
}

__device__ float pointLit(Point &p, Vector n, GraphicsObject* self, Point *lights, Sphere *spheres, Triangle *triangles) {
	Ray ray;
	float lit = 0, t;
	bool col;
	for (int i = 0; i < LIGHTS; i++) {
		ray = Ray(p, lights[i]);
		if (n * ray.d > 0) {
			col = false;
			for (int j = 0; j < SPHC; j++) {
				if (spheres + j != self && ray.intersects(spheres[j], &t) && t > 0.0001) {
					col = true;
					break;
				}
			}
			if (!col) {
				for (int j = 0; j < TRIS; j++) {
					if (triangles + j != self && ray.intersects(triangles[j], &t) && t > 0.0001) {
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

__device__ bool findColPoint(Ray ray, Point *colPoint, Vector *colNormal, GraphicsObject **colObj, Sphere *spheres, Triangle *triangles) {

	float t1, nearest = INFINITY;
	bool mirror = false;

	for (int i = 0; i < SPHC; i++) {
		if (ray.intersects(spheres[i], &t1, nullptr)) {
			if (t1 < nearest && t1 > 0.001) {
				nearest = t1;
				*colPoint = ray.getPointFromT(t1);
				*colNormal = spheres[i].Normal(*colPoint);
				*colObj = spheres + i;
				mirror = spheres[i].mirror;
			}
		}
	}

	for (int i = 0; i < TRIS; i++) {
		if (ray.intersects(triangles[i], &t1)) {
			if (t1 < nearest && t1 > 0.001) {
				nearest = t1;
				*colPoint = ray.getPointFromT(t1);
				*colNormal = triangles[i].n;
				*colObj = triangles + i;
				mirror = triangles[i].mirror;
			}
		}
	}

	if (mirror) {
		return findColPoint(Ray(*colPoint, ray.d.Reflect(*colNormal)), colPoint, colNormal, colObj, spheres, triangles);
	}

	if (nearest < INFINITY) return true;
	return false;
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
	GraphicsObject *obj;

	Ray ray = Ray(camera, pixelPoint);

	float light;

	Point colPoint;

	if (findColPoint(ray, &colPoint, &normal, &obj, spheres, triangles)) {
		light = pointLit(colPoint, normal, obj, lights, spheres, triangles);
		pix[0] = obj->color.r * light;
		pix[1] = obj->color.g * light;
		pix[2] = obj->color.b * light;
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
