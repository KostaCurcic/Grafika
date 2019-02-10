#include "Drawing.h"
#include "Ray.h"

#ifdef CUDA

#include <math.h>
#include <Windows.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#define SPHC 2
#define TRIS 3
#define LIGHTS 1
#define RANDGENS 1000

#define THRCOUNT 8

//Point camera = Point(0, 0, -2.0f);
Sphere spheres[SPHC];
Light lights[LIGHTS];
Triangle triangles[TRIS];
float angle = 0;

char *imgptr, *devImgPtr;
float *realImg;

Sphere *devSpheres;
Light *devLights;
Triangle *devTriangles;
int iteration = 1;
bool started = false;
curandState *devState;
int fc = 0;

void InitFrame()
{
	spheres[0] = Sphere(Point(sinf(angle) * 3, -1, 8 + cosf(angle) * 3), 1);
	//spheres[0].mirror = true;

	spheres[1] = Sphere(Point(5, -1, 5), 1);
	spheres[1].color.r = 50;
	spheres[1].color.g = 200;
	spheres[1].color.b = 100;

	lights[0] = Light(Sphere(Point(-100, 100, 10), 10), .1f);
	lights[0].color.r = 239;
	lights[0].color.g = 163;
	lights[0].color.b = 56;


	//lights[1] = Sphere(Point(-7, 0, 6), 0.5);
	triangles[0] = Triangle(Point(10, -2, 0), Point(-10, -2, 0), Point(10, -2, 20));
	triangles[1] = Triangle(Point(-10, -2, 0), Point(-10, -2, 20), Point(10, -2, 20));

	triangles[2] = Triangle(Point(-4, 2, 6), Point(-5, -2, 8), Point(-5, -5, 4));
	//triangles[2].mirror = true;
	//triangles[2].color.r = 240;

	angle += 0.001f;

	cudaError_t cudaStatus = cudaMemcpy(devSpheres, spheres, SPHC * sizeof(Sphere), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
		return;
	}

	cudaStatus = cudaMemcpy(devLights, lights, LIGHTS * sizeof(Light), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
		return;
	}

	cudaStatus = cudaMemcpy(devTriangles, triangles, TRIS * sizeof(Triangle), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
		return;
	}

}

#ifdef NONRT
__device__ bool findColPointR(Ray ray, Point *colPoint, Vector *colNormal, GraphicsObject **colObj, Sphere *spheres, Triangle *triangles, Light *lights, int iterations = 1) {

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

	for (int i = 0; i < LIGHTS; i++) {
		if (ray.intersects(lights[i], &t1)) {
			if (t1 < nearest && t1 > 0.001) {
				nearest = t1;
				*colPoint = ray.getPointFromT(t1);
				*colNormal = lights[i].Normal(*colPoint);
				*colObj = lights + i;
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
		return findColPointR(Ray(*colPoint, ray.d.Reflect(*colNormal)), colPoint, colNormal, colObj, spheres, triangles, lights, iterations - 1);
	}

	if (nearest < INFINITY) return true;
	return false;
}

__global__ void setup_kernel(curandState *state) {

	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if (idx >= RANDGENS) return;
	curand_init(1234 + idx, idx, 0, &state[idx]);
}

__global__ void drawPixelCUDAR(char* ptr, float* realMap, Light *lights, Sphere *spheres, Triangle *triangles, int iter, curandState *state) {
	int xi = blockIdx.x * THRCOUNT + threadIdx.x;
	int yi = blockIdx.y * THRCOUNT + threadIdx.y;

	if (xi > XRES || yi > YRES) return;

	float x = (xi * 2.0f + curand_uniform(state + ((xi * 100 + yi + 3) % RANDGENS)) * 2.0f) / YRES - XRES / (float)YRES;
	float y = (yi * 2.0f + curand_uniform(state + ((xi * 100 + yi + 3) % RANDGENS)) * 2.0f) / YRES - 1.0;

	char *pix = ptr + (yi * XRES + xi) * 3;
	float *rm = realMap + (yi * XRES + xi) * 3;
	 
	Point pixelPoint(x, y, 0);

	Point camera = Point(0, 0, -2.0f);
	Vector normal;
	GraphicsObject *obj = nullptr;

	Ray ray = Ray(camera, pixelPoint);

	float light;
	float ra, c1, c2, c3;

	Point colPoint;

	float expMulti = 1000;
	float rMulR = 1.0f, rMulG = 1.0f, rMulB = 1.0f;

	int bounceCount = 5;

	for (bounceCount = 5; bounceCount > 0; bounceCount--) {
		if (!findColPointR(ray, &colPoint, &normal, &obj, spheres, triangles, lights)) {

			rm[0] += 18.2 * rMulR;
			rm[1] += 42.4 * rMulG;
			rm[2] += 55.2 * rMulB;
			break;
		}
		else {
			if (obj->shape == LIGHT) {

				rm[0] += ((Light*)obj)->R() * rMulR;
				rm[1] += ((Light*)obj)->G() * rMulG;
				rm[2] += ((Light*)obj)->B() * rMulB;
				break;
			}
			else {
				rMulR *= (obj->color.r * obj->color.r) / 65100.0f;
				rMulG *= (obj->color.g * obj->color.g) / 65100.0f;
				rMulB *= (obj->color.b * obj->color.b) / 65100.0f;
				ray.o = colPoint;
				do {
					ray.d.x = curand_uniform(state + ((xi * 100 + yi) % RANDGENS)) * 2 - 1.0f;
					ray.d.y = curand_uniform(state + ((xi * 100 + yi + 1) % RANDGENS)) * 2 - 1.0f;
					ray.d.z = curand_uniform(state + ((xi * 100 + yi + 2) % RANDGENS)) * 2 - 1.0f;
					ray.d.Normalize();
					if (ray.d * normal <= 0) ray.d = -ray.d;
				} while (ray.d * normal <= curand_uniform(state + ((xi * 100 + yi + 3) % RANDGENS)));
			}
		}
	}

	c1 = sqrtf(rm[0] / iter * expMulti);
	c2 = sqrtf(rm[1] / iter * expMulti);
	c3 = sqrtf(rm[2] / iter * expMulti);

	if (c1 > 255) c1 = 255;
	if (c2 > 255) c2 = 255;
	if (c3 > 255) c3 = 255;

	pix[0] = c1;
	pix[1] = c2;
	pix[2] = c3;
	return;
}

void DrawFrame() {
	dim3 thrds(THRCOUNT, THRCOUNT);
	dim3 blocks(XRES / THRCOUNT, YRES / THRCOUNT);

	cudaError_t cudaStatus;

	for (int i = 0; i < 5; i++) {
		drawPixelCUDAR << <blocks, thrds >> > (devImgPtr, realImg, devLights, devSpheres, devTriangles, iteration, devState);

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

		iteration++;

		printf("Iteration : %d\n", iteration);
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(imgptr, devImgPtr, XRES * YRES * 3 * sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return;
	}

	/*if (iteration % 50 < 5) {
		FILE* pFile;
		char name[] = "fileXX.raw";
		name[4] = fc / 10 + '0';
		name[5] = fc % 10 + '0';
		pFile = fopen(name, "wb");
		fwrite(imgptr, sizeof(char), XRES * YRES * 3, pFile);
		fclose(pFile);
		printf("Saving...\n");
		fc++;
	}*/


	/*if (iteration >= 2000) {
		iteration = 0;
		cudaMemset(realImg, 0, XRES * YRES * 3 * sizeof(float));
		FILE* pFile;
		char name[] = "fileXX.raw";
		name[4] = fc / 10 + '0';
		name[5] = fc % 10 + '0';
		pFile = fopen(name, "wb");
		fwrite(imgptr, sizeof(char), XRES * YRES * 3, pFile);
		fclose(pFile);
		printf("Saving...\n");
		InitFrame();
		fc++;
	}*/
};

void InitDrawing(char *ptr) {
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

	cudaStatus = cudaMalloc((void**)&realImg, XRES * YRES * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		return;
	}

	cudaStatus = cudaMalloc((void**)&devSpheres, SPHC * sizeof(Sphere));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		return;
	}

	cudaStatus = cudaMalloc((void**)&devLights, LIGHTS * sizeof(Light));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		return;
	}

	cudaStatus = cudaMalloc((void**)&devState, sizeof(curandState) * RANDGENS);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		return;
	}

	cudaStatus = cudaMalloc((void**)&devTriangles, TRIS * sizeof(Triangle));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		return;
	}

	setup_kernel << <10, RANDGENS / 10 >> > (devState);

	InitFrame();
}

#else

__device__ float pointLit(Point &p, Vector n, GraphicsObject* self, Light *lights, Sphere *spheres, Triangle *triangles) {
	Ray ray;
	float lit = 0, t;
	bool col;
	for (int i = 0; i < LIGHTS; i++) {
		ray = Ray(p, lights[i].c);
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

__device__ bool findColPoint(Ray ray, Point *colPoint, Vector *colNormal, GraphicsObject **colObj, Sphere *spheres, Triangle *triangles, int iterations = 2) {

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

	if (mirror && iterations > 0) {
		return findColPoint(Ray(*colPoint, ray.d.Reflect(*colNormal)), colPoint, colNormal, colObj, spheres, triangles, iterations - 1);
	}

	if (nearest < INFINITY) return true;
	return false;
}


__global__ void drawPixelCUDA(char* ptr, Light *lights, Sphere *spheres, Triangle *triangles) {
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
		pix[0] = obj->color.r * light + 8 * (1 - light);
		pix[1] = obj->color.g * light + 24 * (1 - light);
		pix[2] = obj->color.b * light + 48 * (1 - light);
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

	cudaStatus = cudaMalloc((void**)&devLights, LIGHTS * sizeof(Light));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		return;
	}

	cudaStatus = cudaMalloc((void**)&devTriangles, TRIS * sizeof(Triangle));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		return;
	}
	started = true;
}

void DrawFrame()
{
	if (!started) return;
	InitFrame();
	
	cudaError_t cudaStatus;

	cudaStatus = cudaMemcpy(devSpheres, spheres, SPHC * sizeof(Sphere), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
		return;
	}

	cudaStatus = cudaMemcpy(devLights, lights, LIGHTS * sizeof(Light), cudaMemcpyHostToDevice);
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

#endif
