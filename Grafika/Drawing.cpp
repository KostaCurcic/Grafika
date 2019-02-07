#include "Drawing.h"
#include "Ray.h"
#include <math.h>

#define SPHC 2
#define LIGHTS 1

Point camera = Point(0, 0, -2.0f);
Sphere spheres[SPHC];
Point lights[LIGHTS];
float angle = 0;

void InitFrame()
{
	spheres[0] = Sphere(Point(sinf(angle) * 3, 0, 10 + cosf(angle) * 3), 1);
	angle += 0.01;
	spheres[1] = Sphere(Point(0, -1000, 10), 995);
	lights[0] = Point(0, 2, 10);
	//lights[1] = Point(1000, 0, 0);
}

void drawPixel(float x, float y, char *pix) {
	Point pixelPoint(x, y, 0);

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
				if (spheres[i].Normal(colPoint) * shadowRay.d > 0) {
					for (int s = 0; s < SPHC; s++) {
						if (s == i) continue;
						if (shadowRay.intersects(spheres[s], nullptr, nullptr)) {
							sCollided = true;
							break;
						}
					}
					if (!sCollided) {
						pix[0] = 50;
						pix[1] = 200;
						pix[2] = 100;
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
		if (ray.d.y > 0) {
			pix[0] = 41;
			pix[1] = 119;
			pix[2] = 240;
		}
		else {
			pix[0] = 137;
			pix[1] = 71;
			pix[2] = 0;
		}
	}
}
