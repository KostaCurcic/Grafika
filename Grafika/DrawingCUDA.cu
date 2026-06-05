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

// What kind of surface a trace step landed on. Used by the accumulation loop to decide
// double-count suppression: a light reached through a mirror was already accounted for by
// the diffuse vertex's mirror NEE, so the flag must survive a HIT_MIRROR step.
#define HIT_DIFFUSE     0
#define HIT_MIRROR      1
#define HIT_TRANSPARENT 2
#define HIT_LIGHT       3
#define HIT_SKY         4

// ---- Exact mirror NEE (Alhazen's problem) -------------------------------------------------
// Given a diffuse point P and a point Q on a light, we want the point M on a mirror sphere
// where the ray P->M reflects exactly toward Q. The reflection point always lies in the plane
// through C (sphere centre), P and Q, so we reduce to a 1D search over the angle phi around
// that circle. The reflection law in that plane is: the unit vectors M->P and M->Q sum to a
// vector parallel to the (radial) surface normal -- i.e. their combined tangential component
// is zero. alhazenF returns that tangential component; its roots are the candidate points.
//
// (Px,0) and (Qx,Qy) are P and Q in 2D plane coordinates with C at the origin and P on the
// +x axis. rho is the mirror radius.
__device__ __forceinline__ float alhazenF(float phi, float Px, float Qx, float Qy, float rho) {
	float c = cosf(phi), s = sinf(phi);
	float mx = rho * c, my = rho * s;
	float dpx = Px - mx, dpy = -my;       // P - M   (Py = 0)
	float dqx = Qx - mx, dqy = Qy - my;   // Q - M
	float lp = sqrtf(dpx * dpx + dpy * dpy);
	float lq = sqrtf(dqx * dqx + dqy * dqy);
	float hx = dpx / lp + dqx / lq;       // unit(P-M) + unit(Q-M)
	float hy = dpy / lp + dqy / lq;
	return hx * (-s) + hy * c;            // component along the tangent (-s, c)
}

// Returns the half-vector . normal at phi if this is a real outward reflection (both points on
// the lit side and the half-vector points outward), else -1. Used to pick the physical root.
__device__ __forceinline__ float alhazenScore(float phi, float Px, float Qx, float Qy, float rho) {
	float c = cosf(phi), s = sinf(phi);
	float mx = rho * c, my = rho * s;
	float dpx = Px - mx, dpy = -my;
	float dqx = Qx - mx, dqy = Qy - my;
	float lp = sqrtf(dpx * dpx + dpy * dpy);
	float lq = sqrtf(dqx * dqx + dqy * dqy);
	float upx = dpx / lp, upy = dpy / lp;
	float uqx = dqx / lq, uqy = dqy / lq;
	float upn = upx * c + upy * s;
	float uqn = uqx * c + uqy * s;
	float hn  = (upx + uqx) * c + (upy + uqy) * s;
	if (upn > 0 && uqn > 0 && hn > 0) return hn;
	return -1.0f;
}

// Solve Alhazen for the mirror sphere (centre C, radius rho), points P and Q. On success fills
// Mout with the 3D reflection point. When useHint is set we pick the valid root whose 3D point
// is closest to Mhint (keeps the finite-difference solves on the same reflection branch as the
// base solve); otherwise we pick the most physical root (largest outward half-vector).
__device__ bool solveAlhazen(const Point& P, const Point& Q, const Point& C, float rho,
                             Point& Mout, bool useHint, Point Mhint) {
	Vector a = P - C;
	Vector b = Q - C;
	Vector w = a % b;                 // plane normal (cross product)
	if (w.Length() < 1e-7f) return false;   // P, Q, C colinear -> degenerate, skip

	Vector e1 = a; e1.Normalize();
	Vector e2 = w % e1; e2.Normalize();     // in-plane axis, perpendicular to e1

	float Px = a * e1;                 // = |a|  (P lies on +e1)
	float Qx = b * e1;
	float Qy = b * e2;

	const float TWO_PI = 6.2831853f;
	const int   N = 48;

	float rootPhi[6];
	int   nr = 0;

	float fprev = alhazenF(0.0f, Px, Qx, Qy, rho);
	for (int sIdx = 1; sIdx <= N && nr < 6; sIdx++) {
		float phi = TWO_PI * sIdx / N;
		float f = alhazenF(phi, Px, Qx, Qy, rho);
		if ((fprev < 0 && f >= 0) || (fprev > 0 && f <= 0)) {
			// Bracketed a root in [lo, hi]; refine by bisection.
			float lo = TWO_PI * (sIdx - 1) / N, hi = phi, flo = fprev;
			for (int it = 0; it < 30; it++) {
				float mid = 0.5f * (lo + hi);
				float fm = alhazenF(mid, Px, Qx, Qy, rho);
				if ((flo < 0 && fm < 0) || (flo > 0 && fm > 0)) { lo = mid; flo = fm; }
				else hi = mid;
			}
			float pr = 0.5f * (lo + hi);
			if (alhazenScore(pr, Px, Qx, Qy, rho) > 0) rootPhi[nr++] = pr;
		}
		fprev = f;
	}
	if (nr == 0) return false;

	int sel = 0;
	if (useHint) {
		float best = 1e30f;
		for (int i = 0; i < nr; i++) {
			float c = cosf(rootPhi[i]), s = sinf(rootPhi[i]);
			Point Mi = C + (e1 * (rho * c) + e2 * (rho * s));
			float d = Vector(Mi - Mhint).Length();
			if (d < best) { best = d; sel = i; }
		}
	}
	else {
		float best = -1.0f;
		for (int i = 0; i < nr; i++) {
			float sc = alhazenScore(rootPhi[i], Px, Qx, Qy, rho);
			if (sc > best) { best = sc; sel = i; }
		}
	}

	float c = cosf(rootPhi[sel]), s = sinf(rootPhi[sel]);
	Mout = C + (e1 * (rho * c) + e2 * (rho * s));
	return true;
}

// Exact mirror NEE: for each mirror sphere and light, sample a point Q on the light, solve for
// the reflection point M, and connect P->M->Q. Because M is found exactly, the path always
// reaches the light -- so every sample contributes (no waiting for a lucky reflection). The
// curved mirror stretches the light's image, so we weight by a numerically estimated Jacobian
// dOmega_P / dArea_Q (how much light-surface area maps to solid angle at P through the mirror).
__device__ ColorReal getMirrorLight(Point& p, Vector n, GraphicsObject* self, SceneData* sd, unsigned int& rng) {
	ColorReal total = ColorReal(0, 0, 0);
	float t;

	for (int m = 0; m < sd->nSpheres; m++) {
		if (!sd->spheres[m].mat.mirror) continue;

		float rho = sd->spheres[m].r;
		Point  cm = sd->spheres[m].c;
		ColorReal mirrorCol = sd->spheres[m].mat.getColor(0, 0).getColorIntesity(sd->gamma);

		for (int k = 0; k < sd->nLights; k++) {
			float rL = sd->lights[k].r;
			Point cL = sd->lights[k].c;

			// Sample a point Q on the light's surface; qN is the light's outward normal there.
			Vector qN = Vector(next(rng) * 2 - 1.0f, next(rng) * 2 - 1.0f, next(rng) * 2 - 1.0f).Normalize();
			Point  Q = cL + qN * rL;

			Point M;
			if (!solveAlhazen(p, Q, cm, rho, M, false, Point())) continue;

			Vector toM = M - p;
			float distPM = toM.Length();
			Vector wP = toM / distPM;                 // unit P->M
			float cosP = n * wP;
			if (cosP <= 0) continue;                  // mirror image is below our surface

			Vector MtoQ = Q - M;
			float dMQ = MtoQ.Length();
			// Light's surface must face M: qN . (M - Q) > 0, i.e. qN . (Q - M) < 0.
			if ((qN * MtoQ) >= 0) continue;

			// Visibility P -> M (ignore the mirror itself and the surface we are on).
			Ray rPM = Ray(p, M);
			bool blocked = false;
			for (int j = 0; j < sd->nSpheres && !blocked; j++) {
				if (sd->spheres + j == self || j == m) continue;
				if (rPM.intersects(sd->spheres[j], nullptr, &t) && t > 0.0001f && t < distPM - 0.001f) blocked = true;
			}
			for (int j = 0; j < sd->nTriangles && !blocked; j++) {
				if (sd->triangles + j == self) continue;
				if (rPM.intersects(sd->triangles[j], nullptr, &t) && t > 0.0001f && t < distPM - 0.001f) blocked = true;
			}
			if (blocked) continue;

			// Visibility M -> Q (ignore the mirror itself).
			Ray rMQ = Ray(M, Q);
			bool occ = false;
			for (int j = 0; j < sd->nSpheres && !occ; j++) {
				if (j == m) continue;
				if (rMQ.intersects(sd->spheres[j], nullptr, &t) && t > 0.0001f && t < dMQ - 0.001f) occ = true;
			}
			for (int j = 0; j < sd->nTriangles && !occ; j++) {
				if (rMQ.intersects(sd->triangles[j], nullptr, &t) && t > 0.0001f && t < dMQ - 0.001f) occ = true;
			}
			if (occ) continue;

			// Numerical Jacobian: nudge Q along two tangent directions, re-solve M, and measure
			// how far the reflected direction (seen from P) fans out per unit light area.
			Vector tg1 = qN % Vector(0, 1, 0);
			if (tg1.Length() < 0.01f) tg1 = qN % Vector(1, 0, 0);
			tg1.Normalize();
			Vector tg2 = qN % tg1; tg2.Normalize();
			float eps = rL * 0.01f + 1e-4f;

			Point M1, M2;
			if (!solveAlhazen(p, Q + tg1 * eps, cm, rho, M1, true, M)) continue;
			if (!solveAlhazen(p, Q + tg2 * eps, cm, rho, M2, true, M)) continue;

			Vector w0 = toM;        w0.Normalize();
			Vector w1 = M1 - p;     w1.Normalize();
			Vector w2 = M2 - p;     w2.Normalize();
			float J = ((w1 - w0) % (w2 - w0)).Length() / (eps * eps);

			ColorReal lightRad = sd->lights[k].mat.color.getColorIntesity(sd->gamma) * sd->lights[k].intenisty;

			// (1/pi BRDF) * cosP * L * J * area(4*pi*rL^2)  ->  4*rL^2 after the pi's cancel.
			total += mirrorCol * lightRad * (cosP * J * 4.0f * rL * rL);
		}
	}

	return total;
}


__device__ ColorReal traceRandIter(Ray ray, SceneData* sd, unsigned int& rng, Ray* newRay, ColorReal* predictedLight, int *hitType) {
	float t1, nearest = INFINITY;
	ColorReal colorMultiplier(1, 1, 1);
	ColorReal colGet;
	Point colPoint;
	Vector colNormal;
	GraphicsObject* colObj;
	*predictedLight = ColorReal(0, 0, 0);

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
		*hitType = HIT_SKY;
		*newRay = Ray(Point(-INFINITY, -INFINITY, -INFINITY), Vector(0, 0, 0));
		return sd->ambient.mat.color.getColorIntesity(sd->gamma) * sd->ambient.intenisty;
	}
	else if (colObj->shape == LIGHT) {
		*hitType = HIT_LIGHT;
		*newRay = Ray(Point(-INFINITY, -INFINITY, INFINITY), Vector(0, 0, 0));
		return colorMultiplier;
	}
	else {
		if (colObj->mat.mirror) {
			*hitType = HIT_MIRROR;
			*newRay = Ray(colPoint, ray.d.Reflect(colNormal));
			return colorMultiplier;
		}
		else if (colObj->mat.transparent) {
			*hitType = HIT_TRANSPARENT;
			*newRay = Ray(colPoint, ray.d.Refract(colNormal, colObj->mat.refIndex));
			return colorMultiplier;
		}
		else {
			*hitType = HIT_DIFFUSE;
			ray.o = colPoint;
			if (ray.d * colNormal > 0) colNormal = -colNormal;
			if (sd->useLightPredict) {
				*predictedLight = getProbabilisticLight(colPoint, colNormal, colObj, sd, rng);
				if (sd->useMirrorPredict)
					*predictedLight = *predictedLight + getMirrorLight(colPoint, colNormal, colObj, sd, rng);
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
	int hitType;
	bool lastPredictRan = false;
	for (int i = 0; i <= sd->bounces; i++) {
		if(i == sd->bounces){
			newPixelColor = ColorReal(0, 0, 0);
			break;
		}

		newPixelColorNew = traceRandIter(ray, sd, rng, &newRay, &predictedLightNew, &hitType);

		// A light reached after a diffuse vertex was already counted by that vertex's NEE
		// (direct light or, through a mirror bounce, its mirror NEE) -- drop the duplicate.
		if (sd->useLightPredict && lastPredictRan && hitType == HIT_LIGHT) {
			newPixelColor = ColorReal(0, 0, 0);
		}
		else {
			newPixelColor *= newPixelColorNew;
		}

		if (sd->useLightPredict) {
			predictedLight += predictedLightNew * newPixelColor;   // now includes current albedo
			predictedLightNew = ColorReal(0, 0, 0);
			// A diffuse vertex arms suppression; glass clears it. A mirror passes it through
			// ONLY when mirror NEE is on (it covered the reflected light); otherwise the mirror
			// must clear it so the bounce path is still allowed to gather light through it.
			if (hitType == HIT_DIFFUSE) lastPredictRan = true;
			else if (hitType == HIT_TRANSPARENT) lastPredictRan = false;
			else if (hitType == HIT_MIRROR && !sd->useMirrorPredict) lastPredictRan = false;
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

		printf("It %4d  %.2f it/s | Exp %.0f  Gamma %.3f  Bounces %d | Focal %.2f  DOF %.4f  FOV %.2f | LightPredict %d  Mirror %d          \r",
			iteration, itps, sd.expMultiplier, sd.gamma, sd.bounces, sd.focalDistance, sd.dofStr, sd.camDist, sd.useLightPredict, sd.useMirrorPredict);

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
