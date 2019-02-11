#include <stdio.h>
#include <Windows.h>
#include <gl\GL.h>
#include <gl\GLU.h>
#include <math.h>
#include "OpenGL3.1.h"

#include "GLInit.h"
#include "Drawing.h"

GLuint tex;
char *arr, *tarr;

void initial(WPARAM wParam, LPARAM lParam) {

	LoadShaderFunctions();

	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, XRES, YRES, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	arr = (char*)malloc(XRES * YRES * 3);

	sd.camera = Point(0, 0, -2);
	sd.expMultiplier = 1000;
	sd.dofStr = 0.005f;
	sd.focalDistance = 5.0f;
	sd.gamma = 2.224f;

	sd.nLights = 1;
	sd.nSpheres = 2;
	sd.nTriangles = 3;

	sd.spheres = (Sphere*)malloc(sd.nSpheres * sizeof(Sphere));
	sd.triangles = (Triangle*)malloc(sd.nTriangles * sizeof(Triangle));
	sd.lights = (Light*)malloc(sd.nSpheres * sizeof(Sphere));

	sd.spheres[0] = Sphere(Point(sinf(0) * 3, -1, 8 + cosf(0) * 3), 1);
	sd.spheres[0].mirror = true;

	sd.spheres[1] = Sphere(Point(5, -1, 5), 1);
	sd.spheres[1].color.r = 50;
	sd.spheres[1].color.g = 200;
	sd.spheres[1].color.b = 100;

	sd.lights[0] = Light(Sphere(Point(-100, 100, -50), 10), .2f);
	sd.lights[0].color.r = 239;
	sd.lights[0].color.g = 163;
	sd.lights[0].color.b = 56;

	sd.triangles[0] = Triangle(Point(10, -2, 0), Point(-10, -2, 0), Point(10, -2, 20));
	sd.triangles[1] = Triangle(Point(-10, -2, 0), Point(-10, -2, 20), Point(10, -2, 20));

	sd.triangles[2] = Triangle(Point(0, 2, 5), Point(2, 0, 5), Point(-2, 0, 5));

	InitDrawing(arr);


}

void draw(WPARAM wParam, LPARAM lParam) {

	glBindTexture(GL_TEXTURE_2D, tex);

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, XRES, YRES, GL_RGB, GL_UNSIGNED_BYTE, arr);
	DrawFrame();

	glClearColor(1, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_TEXTURE_2D);

	glBegin(GL_QUADS);
		glTexCoord2f(0, 1);
		glVertex2f(-1, 1);

		glTexCoord2f(0, 0);
		glVertex2f(-1, -1);

		glTexCoord2f(1, 0);
		glVertex2f(1, -1);

		glTexCoord2f(1, 1);
		glVertex2f(1, 1);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);
}

void key(WPARAM wParam, LPARAM lParam) {
	switch (wParam)
	{
	case VK_DOWN:
		sd.spheres[0].c.z -= 0.05;
		break;
	case VK_UP:
		sd.spheres[0].c.z += 0.05;
		break;
	case VK_LEFT:
		sd.spheres[0].c.x -= 0.05;
		break;
	case VK_RIGHT:
		sd.spheres[0].c.x += 0.05;
		break;
	case VK_NUMPAD4:
		sd.expMultiplier *= 1.1;
		break;
	case VK_NUMPAD1:
		sd.expMultiplier /= 1.1;
		break;
	case VK_OEM_PLUS:
		sd.gamma *= 1.1f;
		break;
	case VK_OEM_MINUS:
		sd.gamma /= 1.1f;
		break;
	case 0x31: //1
		break;
	case 0x57: //W
		break;
	case 0x53: //S
		break;
	}
	InitFrame();
}

int main() {

	EVENTFUNC functions[]{
		EVENTFUNC {WM_PAINT, draw},
		EVENTFUNC {WM_KEYDOWN, key},
		EVENTFUNC {WM_CREATE, initial},
	};

	DoGL(3, functions, 1920, 1080);
}