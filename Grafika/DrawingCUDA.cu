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

#define ITER

float angle = 0;

char *imgptr, *devImgPtr;
float *realImg = nullptr;

int iteration = 1;
bool started = false;
int fc = 0;

SceneData sd, devSdCopy;
SceneData *devSd;

Light *devLights;
Sphere *devSpheres;
Triangle *devTriangles;
Material *devMaterials;

LARGE_INTEGER timer;

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


__device__ __forceinline__ unsigned int pcg_hash(unsigned int x) {
	x = x * 747796405u + 2891336453u;
	unsigned int w = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
	return (w >> 22u) ^ w;
}

// each time you need a uniform float in [0,1):
// State is advanced by a full-period LCG (visits all 2^32 values in one cycle, so it
// can never get stuck on a constant); the returned value is a hash of the state.
__device__ __forceinline__ float next(unsigned int& state) {
	state = state * 747796405u + 2891336453u;          // LCG advance, full period 2^32
	unsigned int w = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return ((w >> 22u) ^ w) * (1.0f / 4294967296.0f);  // output = hash of state
}

__device__ ColorReal getProbabilisticLight(Point& p, Vector n, GraphicsObject* self, SceneData* sd, unsigned int& rng) {
	Ray ray;
	float t;
	ColorReal totalLight = ColorReal(0, 0, 0);
	bool col;
	for (int i = 0; i < sd->nLights; i++) {
		float r = sd->lights[i].r;
		// Pick a random point on the surface of the light sphere. lightNormal is the
		// unit direction from the light's center to that point, i.e. the light's own
		// surface normal there.
		Vector lightNormal = Vector(next(rng) * 2 - 1.0f, next(rng) * 2 - 1.0f, next(rng) * 2 - 1.0f).Normalize();
		Point lightPoint = sd->lights[i].c + lightNormal * r;
		ray = Ray(p, lightPoint);

		float cosX = n * ray.d;   // how slanted the light arrives at our surface
		if (cosX > 0) {
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
				// How the light's surface is angled toward us. Points on the far side of
				// the sphere face away (cosY <= 0) and contribute nothing.
				float cosY = -(lightNormal * ray.d);
				if (cosY > 0) {
					Vector toLight = lightPoint - p;
					float dist2 = toLight * toLight;
					// (1/pi) BRDF * cosX * cosY * area(4*pi*r^2) / dist^2; the pi's cancel
					// to leave 4*r^2. The surface's own colour is applied later via the
					// path throughput (newPixelColor), so it is not multiplied in here.
					totalLight += sd->lights[i].mat.color.getColorIntesity(sd->gamma)
						* sd->lights[i].intenisty
						* (cosX * cosY * 4.0f * r * r / dist2);
				}
			}
		}

	}
	return totalLight;
}


__device__ ColorReal traceRandIter(Ray ray, SceneData* sd, unsigned int& rng, Ray* newRay, ColorReal* predictedLight, bool *predictRan) {
	float t1, nearest = INFINITY;
	ColorReal colorMultiplier(1, 1, 1);
	ColorReal colGet;
	Point colPoint;
	Vector colNormal;
	GraphicsObject* colObj;
	*predictRan = false;

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
		*newRay = Ray(Point(-INFINITY, -INFINITY, -INFINITY), Vector(0, 0, 0));
		return sd->ambient.mat.color.getColorIntesity(sd->gamma) * sd->ambient.intenisty;
	}
	else if (colObj->shape == LIGHT) {
		*newRay = Ray(Point(-INFINITY, -INFINITY, INFINITY), Vector(0, 0, 0));
		return colorMultiplier;
	}
	else {
		if (colObj->mat.mirror) {
			*newRay = Ray(colPoint, ray.d.Reflect(colNormal));
			return colorMultiplier;
		}
		else if (colObj->mat.transparent) {
			*newRay = Ray(colPoint, ray.d.Refract(colNormal, colObj->mat.refIndex));
			return colorMultiplier;
		}
		else {
			ray.o = colPoint;
			if (ray.d * colNormal > 0) colNormal = -colNormal;
			if (sd->useLightPredict) {
				*predictedLight = getProbabilisticLight(colPoint, colNormal, colObj, sd, rng);
				*predictRan = true;
			}
			do {
				ray.d.x = next(rng) * 2 - 1.0f;
				ray.d.y = next(rng) * 2 - 1.0f;
				ray.d.z = next(rng) * 2 - 1.0f;
				ray.d.Normalize();
				if (ray.d * colNormal <= 0) ray.d = -ray.d;
			} while (ray.d * colNormal <= next(rng));
			*newRay = ray;
			return colorMultiplier;
		}
	}
}

__device__ ColorReal traceRand(Ray ray, SceneData *sd, unsigned int& rng, int iterations = 20) {
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
			return colorMultiplier *= traceRand(Ray(colPoint, ray.d.Reflect(colNormal)), sd, rng, iterations - 1);
		}
		else if (colObj->mat.transparent) {
			return colorMultiplier *= traceRand(Ray(colPoint, ray.d.Refract(colNormal, colObj->mat.refIndex)), sd, rng, iterations - 1);
		}
		else {
			ray.o = colPoint;
			if (ray.d * colNormal > 0) colNormal = -colNormal;
			do {
				ray.d.x = next(rng) * 2 - 1.0f;
				ray.d.y = next(rng) * 2 - 1.0f;
				ray.d.z = next(rng) * 2 - 1.0f;
				ray.d.Normalize();
				if (ray.d * colNormal <= 0) ray.d = -ray.d;
			} while (ray.d * colNormal <= next(rng));
			return colorMultiplier *= traceRand(ray, sd, rng, iterations - 1);
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

__global__ void drawPixelCUDAR(char* ptr, float* realMap, SceneData *sd, int iter) {
	int xi = blockIdx.x * THRCOUNT + threadIdx.x;
	int yi = blockIdx.y * THRCOUNT + threadIdx.y;

	unsigned int rng = pcg_hash((xi * XRES + yi + 3) ^ pcg_hash(iter));

	if (xi > XRES || yi > YRES) return;

	float x = (xi * 2.0f + next(rng) * 2.0f) / YRES - XRES / (float)YRES;
	float y = (yi * 2.0f + next(rng) * 2.0f) / YRES - 1.0;

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
		float pointBack = ((Vector)(sd->camera - pixelPoint)).Length();

		float ang = next(rng) * 6.28315f;
		pointMove *= next(rng);
		xOff = sinf(ang) * sqrtf(pointMove);
		yOff = cosf(ang) * sqrtf(pointMove);
		/*do {
			xOff = (curand_uniform(state + ((xi * 100 + yi) % RANDGENS)) * 2 - 1.0f) * pointMove;
			yOff = (curand_uniform(state + ((xi * 100 + yi) % RANDGENS)) * 2 - 1.0f) * pointMove;
		} while (sqrtf(xOff * xOff + yOff * yOff) > pointMove);*/
		Point passPoint = pixelPoint + sd->sR * xOff + sd->sD * yOff;
		ray = Ray(passPoint, focalPoint);
		ray.o = ray.getPointFromT(-pointBack);
	}

	float light;
	float ra, c1, c2, c3;

	Point colPoint;

#ifdef ITER
	Ray newRay;
	ColorReal newPixelColor = ColorReal(1, 1, 1);
	ColorReal newPixelColorNew;
	ColorReal predictedLight = ColorReal(0, 0, 0);
	ColorReal predictedLightNew = ColorReal(0, 0, 0);
	bool predictRan, lastPredictRan = false;
	for (int i = 0; i <= sd->bounces; i++) {
		if(i == sd->bounces){
			newPixelColor = ColorReal(0, 0, 0);
			break;
		}

		newPixelColorNew = traceRandIter(ray, sd, rng, &newRay, &predictedLightNew, &predictRan);

		if (sd->useLightPredict && lastPredictRan && newRay.o.z == INFINITY) {
			newPixelColor = ColorReal(0, 0, 0);
		}
		else {
			newPixelColor *= newPixelColorNew;
		}

		if (sd->useLightPredict) {
			predictedLight += predictedLightNew * newPixelColor;   // now includes current albedo
			predictedLightNew = ColorReal(0, 0, 0);
			lastPredictRan = predictRan;
		}

		if (newRay.o.x == -INFINITY) break;
		ray = newRay;
	}
	*rm += newPixelColor + predictedLight;
#else
	* rm += traceRand(ray, sd, rng, sd->bounces);
#endif // ITER


	*pix = rm->getPixColorDesat(sd->gamma, sd->expMultiplier / iter);

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
		if (obj->shape == LIGHT) light = 1.0f;
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

	// traceRand recurses once per bounce (sd.bounces, default 20). The default
	// per-thread stack (1 KB) overflows that deep recursion and the path-tracer
	// kernel aborts with cudaErrorLaunchFailure (719). Raise the stack to fit.
	/*cudaStatus = cudaDeviceSetLimit(cudaLimitStackSize, 32 * 1024);
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSetLimit(stack) failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}*/

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

		drawPixelCUDAR << <blocks, thrds >> > (devImgPtr, realImg, devSd, iteration);

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

		LARGE_INTEGER freq, now;
		QueryPerformanceFrequency(&freq);
		QueryPerformanceCounter(&now);

		double itps = (double)freq.QuadPart / (now.QuadPart - timer.QuadPart);

		timer = now;

		printf("It %4d  %.2f it/s | Exp %.0f  Gamma %.3f  Bounces %d | Focal %.2f  DOF %.4f  FOV %.2f | LightPrefict %d          \r",
			iteration, itps, sd.expMultiplier, sd.gamma, sd.bounces, sd.focalDistance, sd.dofStr, sd.camDist, sd.useLightPredict);

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
