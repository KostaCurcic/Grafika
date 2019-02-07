#include "Drawing.h"
#include "Ray.h"

#define SPHC 1

Point camera = Point(0, 0, -2.0f);
Sphere spheres[SPHC];

void InitFrame()
{
	spheres[0] = Sphere(Point(0, 0, 10), 1);
}

void drawPixel(float x, float y, char *pix) {
	Point pixelPoint(x, y, 0);

	Ray ray = Ray(camera, pixelPoint);
	bool collided = false;

	for (int i = 0; i < SPHC; i++) {
		if (ray.intersects(spheres[0])) {
			pix[0] = 20;
			pix[1] = 200;
			pix[2] = 100;
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
