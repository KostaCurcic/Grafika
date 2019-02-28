#include "Drawing.h"

#ifdef CUDA

#include <math.h>
#include <Windows.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

/*#define SPHC 2
#define TRIS 3
#define LIGHTS 1*/
#define RANDGENS 1000

#define THRCOUNT 8

float angle = 0;

char *imgptr, *devImgPtr;
float *realImg = nullptr;

int iteration = 1;
bool started = false;
curandState *devState;
int fc = 0;

SceneData sd, devSdCopy;
SceneData *devSd;

Light *devLights;
Sphere *devSpheres;
Triangle *devTriangles;
Material *devMaterials;

void InitFrame()
{

	sd.genCameraCoords();
	devSdCopy = sd.genDeviceData(devSpheres, devTriangles, devLights, devMaterials);

	cudaError_t cudaStatus = cudaMemcpy(devSpheres, sd.spheres, sd.nSpheres * sizeof(Sphere), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
		return;
	}

	cudaStatus = cudaMemcpy(devLights, sd.lights, sd.nLights * sizeof(Light), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
		return;
	}

	cudaStatus = cudaMemcpy(devTriangles, sd.triangles, sd.nTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
		return;
	}

	cudaStatus = cudaMemcpy(devMaterials, sd.materials, sd.nMaterials * sizeof(Material), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
		return;
	}

	cudaStatus = cudaMemcpy(devSd, &devSdCopy, sizeof(SceneData), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
		return;
	}

	if (sd.reset) {
		sd.reset = false;
		iteration = 1;

		if(realImg != nullptr)
			cudaMemset(realImg, 0, XRES * YRES * 3 * sizeof(float));
	}
}

__device__ ColorReal traceRand(Ray ray, SceneData *sd, curandState *state, int iterations = 20) {
	float t1, nearest = INFINITY;
	ColorReal colorMultiplier(1, 1, 1);
	ColorReal colGet;
	Point colPoint;
	Vector colNormal;
	GraphicsObject *colObj;

	if (iterations <= 0) {
		return ColorReal(0, 0, 0);
	}

	for (int i = 0; i < sd->nSpheres; i++) {
		if (ray.intersects(sd->spheres[i], &colGet, &t1, nullptr)) {
			if (t1 < nearest && t1 > 0.001) {
				nearest = t1;
				colPoint = ray.getPointFromT(t1);
				colNormal = sd->spheres[i].Normal(colPoint);
				colObj = sd->spheres + i;
				colorMultiplier = colGet.getColorIntesity(sd->gamma);
			}
		}
	}

	for (int i = 0; i < sd->nLights; i++) {
		if (ray.intersects(sd->lights[i], &colGet, &t1)) {
			if (t1 < nearest && t1 > 0.001) {
				nearest = t1;
				colPoint = ray.getPointFromT(t1);
				colNormal = sd->lights[i].Normal(colPoint);
				colObj = sd->lights + i;
				colorMultiplier = colGet.getColorIntesity(sd->gamma) * sd->lights[i].intenisty;
			}
		}
	}

	for (int i = 0; i < sd->nTriangles; i++) {
		if (ray.intersects(sd->triangles[i], &colGet, &t1)) {
			if (t1 < nearest && t1 > 0.001) {
				nearest = t1;
				colPoint = ray.getPointFromT(t1);
				colNormal = sd->triangles[i].n;
				colObj = sd->triangles + i;
				colorMultiplier = colGet.getColorIntesity(sd->gamma);
			}
		}
	}

	if (nearest == INFINITY) {
		return sd->ambient.mat.color.getColorIntesity(sd->gamma) * sd->ambient.intenisty;
	}
	else if (colObj->shape == LIGHT) {
		return colorMultiplier;
	}
	else {
		if (colObj->mat.mirror) {
			return colorMultiplier *= traceRand(Ray(colPoint, ray.d.Reflect(colNormal)), sd, state, iterations - 1);
		}
		else if (colObj->mat.transparent) {
			return colorMultiplier *= traceRand(Ray(colPoint, ray.d.Refract(colNormal, colObj->mat.refIndex)), sd, state, iterations - 1);
		}
		else {
			ray.o = colPoint;
			if (ray.d * colNormal > 0) colNormal = -colNormal;
			do {
				ray.d.x = curand_uniform(state) * 2 - 1.0f;
				ray.d.y = curand_uniform(state) * 2 - 1.0f;
				ray.d.z = curand_uniform(state) * 2 - 1.0f;
				ray.d.Normalize();
				if (ray.d * colNormal <= 0) ray.d = -ray.d;
			} while (ray.d * colNormal <= curand_uniform(state));
			return colorMultiplier *= traceRand(ray, sd, state, iterations - 1);
		}
	}
}

__device__ bool findColPoint(Ray ray, Point *colPoint, Vector *colNormal, GraphicsObject **colObj, SceneData *sd, int iterations = 3) {

	float t1, nearest = INFINITY;
	bool mirror = false;
	bool transparent = false;

	for (int i = 0; i < sd->nSpheres; i++) {
		if (ray.intersects(sd->spheres[i], nullptr, &t1, nullptr)) {
			if (t1 < nearest && t1 > 0.001) {
				nearest = t1;
				*colPoint = ray.getPointFromT(t1);
				*colNormal = sd->spheres[i].Normal(*colPoint);
				*colObj = sd->spheres + i;
				mirror = (*colObj)->mat.mirror;
				transparent = (*colObj)->mat.transparent;
			}
		}
	}

	for (int i = 0; i < sd->nLights; i++) {
		if (ray.intersects(sd->lights[i], nullptr, &t1)) {
			if (t1 < nearest && t1 > 0.001) {
				nearest = t1;
				*colPoint = ray.getPointFromT(t1);
				*colNormal = sd->lights[i].Normal(*colPoint);
				*colObj = sd->lights + i;
			}
		}
	}

	for (int i = 0; i < sd->nTriangles; i++) {
		if (ray.intersects(sd->triangles[i], nullptr, &t1)) {
			if (t1 < nearest && t1 > 0.001) {
				nearest = t1;
				*colPoint = ray.getPointFromT(t1);
				*colNormal = sd->triangles[i].n;
				*colObj = sd->triangles + i;
				mirror = (*colObj)->mat.mirror;
				transparent = (*colObj)->mat.transparent;
			}
		}
	}

	if (mirror && iterations > 0) {
		return findColPoint(Ray(*colPoint, ray.d.Reflect(*colNormal)), colPoint, colNormal, colObj, sd, iterations - 1);
	}
	else if(transparent && iterations > 0){
		return findColPoint(Ray(*colPoint, ray.d.Refract(*colNormal, (*colObj)->mat.refIndex)), colPoint, colNormal, colObj, sd, iterations - 1);
	}

	if (nearest < INFINITY) return true;
	return false;
}

__global__ void setup_kernel(curandState *state) {

	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if (idx >= RANDGENS) return;
	curand_init(1234 + idx, idx, 0, &state[idx]);
}

__global__ void drawPixelCUDAR(char* ptr, float* realMap, SceneData *sd, int iter, curandState *state) {
	int xi = blockIdx.x * THRCOUNT + threadIdx.x;
	int yi = blockIdx.y * THRCOUNT + threadIdx.y;

	if (xi > XRES || yi > YRES) return;

	float x = (xi * 2.0f + curand_uniform(state + ((xi * 100 + yi + 3) % RANDGENS)) * 2.0f) / YRES - XRES / (float)YRES;
	float y = (yi * 2.0f + curand_uniform(state + ((xi * 100 + yi + 3) % RANDGENS)) * 2.0f) / YRES - 1.0;

	Color *pix = (Color*)(ptr + (yi * XRES + xi) * 3);
	ColorReal *rm = (ColorReal*)(realMap + (yi * XRES + xi) * 3);

	//Point pixelPoint = Point(10 + x, y, 0);

	Point pixelPoint = sd->camera + sd->c2S + sd->sR * x + sd->sD * y;

	float focalDistance = sd->focalDistance;

	Vector normal;
	GraphicsObject *obj = nullptr;

	Ray ray = Ray(sd->camera, pixelPoint);

	if (sd->dofStr > 0.000001f) {
		Point focalPoint = sd->camera + (Vector)(pixelPoint - sd->camera) * (1 + focalDistance / sd->camDist);

		float pointMove = sd->dofStr, xOff, yOff;

		float ang = curand_uniform(state + ((xi * 100 + yi) % RANDGENS)) * 6.28315f;
		pointMove *= curand_uniform(state + ((xi * 100 + yi) % RANDGENS));
		xOff = sinf(ang) * sqrtf(pointMove);
		yOff = cosf(ang) * sqrtf(pointMove);
		/*do {
			xOff = (curand_uniform(state + ((xi * 100 + yi) % RANDGENS)) * 2 - 1.0f) * pointMove;
			yOff = (curand_uniform(state + ((xi * 100 + yi) % RANDGENS)) * 2 - 1.0f) * pointMove;
		} while (sqrtf(xOff * xOff + yOff * yOff) > pointMove);*/
		Point passPoint = pixelPoint + sd->sR * xOff + sd->sD * yOff;
		ray = Ray(passPoint, focalPoint);
	}

	float light;
	float ra, c1, c2, c3;

	Point colPoint;

	*rm += traceRand(ray, sd, state + ((xi * XRES + yi + 3) + (iter* 123)) % RANDGENS, sd->bounces);
	*pix = rm->getPixColor(sd->gamma, sd->expMultiplier / iter);

	return;
}

__device__ float pointLit(Point &p, Vector n, GraphicsObject* self, SceneData *sd) {
	Ray ray;
	float lit = 0, t;
	bool col;
	for (int i = 0; i < sd->nLights; i++) {
		ray = Ray(p, sd->lights[i].c);
		if (n * ray.d > 0) {
			col = false;
			for (int j = 0; j < sd->nSpheres; j++) {
				if (sd->spheres + j != self && ray.intersects(sd->spheres[j], nullptr, &t) && t > 0.0001) {
					col = true;
					break;
				}
			}
			if (!col) {
				for (int j = 0; j < sd->nTriangles; j++) {
					if (sd->triangles + j != self && ray.intersects(sd->triangles[j], nullptr, &t) && t > 0.0001) {
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

__global__ void drawPixelCUDA(char* ptr, SceneData *sd) {
	int xi = blockIdx.x * THRCOUNT + threadIdx.x;
	int yi = blockIdx.y * THRCOUNT + threadIdx.y;

	if (xi > XRES || yi > YRES) return;

	float x = xi * 2.0f / YRES - XRES / (float)YRES;
	float y = yi * 2.0 / YRES - 1.0;

	Color *pix = (Color*)(ptr + (yi * XRES + xi) * 3);

	Point pixelPoint = sd->camera + sd->c2S + sd->sR * x + sd->sD * y;

	ColorReal color;

	Vector normal;
	GraphicsObject *obj;

	Ray ray = Ray(sd->camera, pixelPoint);

	float light = 1.0f;

	Point colPoint;

	if (findColPoint(ray, &colPoint, &normal, &obj, sd)) {
		if (obj->shape == LIGHT) light == 1.0f;
		else light = pointLit(colPoint, normal, obj, sd);

		if (obj->shape == TRIANGLE && ((Triangle*)obj)->mat.texture.width != 0) {
			float coords[] = { 0, 0 };
			((Triangle*)obj)->interpolatePoint(colPoint, (float*)&(((Triangle*)obj)->t0), (float*)&(((Triangle*)obj)->t1), (float*)&(((Triangle*)obj)->t2), coords, 2);
			ColorReal c = obj->mat.getColor(coords[0], coords[1]);

			color = c * light;
		}
		else {
			color = obj->mat.getColor(0, 0) * light;
		}
	}
	else{
		color = sd->ambient.mat.color;
	}
	*pix = color.getPixColor();
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

	cudaStatus = cudaMalloc((void**)&realImg, XRES * YRES * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		return;
	}

	cudaStatus = cudaMalloc((void**)&devSpheres, sd.nSpheres * sizeof(Sphere));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		return;
	}

	cudaStatus = cudaMalloc((void**)&devLights, sd.nLights * sizeof(Light));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		return;
	}

	cudaStatus = cudaMalloc((void**)&devTriangles, sd.nTriangles * sizeof(Triangle));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		return;
	}

	cudaStatus = cudaMalloc((void**)&devMaterials, sd.nMaterials * sizeof(Material));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		return;
	}

	cudaStatus = cudaMalloc((void**)&devSd, sizeof(SceneData));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		return;
	}

	cudaStatus = cudaMalloc((void**)&devState, sizeof(curandState) * RANDGENS);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
		return;
	}

	setup_kernel << <10, RANDGENS / 10 >> > (devState);

	InitFrame();

}

void DrawFrame()
{
	if (sd.realTime) {
		InitFrame();

		cudaError_t cudaStatus;

		dim3 thrds(THRCOUNT, THRCOUNT);
		dim3 blocks(XRES / THRCOUNT, YRES / THRCOUNT);

		drawPixelCUDA << <blocks, thrds >> > (devImgPtr, devSd);

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
	else {
		dim3 thrds(THRCOUNT, THRCOUNT);
		dim3 blocks(XRES / THRCOUNT, YRES / THRCOUNT);

		cudaError_t cudaStatus;

		drawPixelCUDAR << <blocks, thrds >> > (devImgPtr, realImg, devSd, iteration, devState);

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

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(imgptr, devImgPtr, XRES * YRES * 3 * sizeof(char), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return;
		}

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
	}
}

DEVICE_PREFIX void SceneData::genCameraCoords()
{
	if (camXang > 6.28318f) camXang -= 6.28318f;
	if (camXang < 0.0f) camXang += 6.28318f;
	if (camYang >= 1.5707f && camYang < 3.14159f) camYang = 1.57f;
	else {
		if (camYang < 0.0f) camYang += 6.28318f;
		//TODO doesnt work
		if (camYang > 3.141592f && camYang <= 4.712388f) camYang = 4.714f;
	}

	c2S = Vector(0, 0, 1);

	c2S = Vector(-sinf(camXang), tanf(camYang), cosf(camXang));

	c2S = c2S.Normalize() * camDist;

	sR = Vector(cosf(camXang), 0, sinf(camXang));

	sD = (c2S / camDist) % sR;

}

DEVICE_PREFIX SceneData SceneData::genDeviceData(Sphere *devS, Triangle *devTr, Light *devL, Material *devMa)
{
	SceneData ret = *this;
	/*for (int i = 0; i < nTriangles; i++) {
		ret.triangles[i]. = devTe + triangles[i].texIndex;
	}*/

	ret.lights = devL;
	ret.triangles = devTr;
	ret.spheres = devS;
	ret.materials = devMa;
	return ret;
}

void SceneData::assignPointersHost() {};

#include "Point.cpp"
#include "Ray.cpp"
#include "Sphere.cpp"
#include "Texture.cpp"
#include "Triangle.cpp"
#include "Vector.cpp"
#include "Color.cpp"
#include "Material.cpp"

#endif
