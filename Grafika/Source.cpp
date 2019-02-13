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
	sd.genCameraCoords();

	sd.expMultiplier = 1000;
	sd.dofStr = 0.005f;
	sd.focalDistance = 5.0f;
	sd.gamma = 2.224f;

	sd.nLights = 1;
	sd.nSpheres = 2;
	sd.nTriangles = 3;
	sd.nTextures = 1;

	sd.ambient.color.r = 18;
	sd.ambient.color.g = 42;
	sd.ambient.color.b = 55;
	sd.ambient.intenisty = .01f;

	sd.spheres = (Sphere*)malloc(sd.nSpheres * sizeof(Sphere));
	sd.triangles = (Triangle*)malloc(sd.nTriangles * sizeof(Triangle));
	sd.lights = (Light*)malloc(sd.nSpheres * sizeof(Sphere));
	sd.textures = (Texture*)malloc(sd.nTextures * sizeof(Texture));

	sd.textures[0].load(R"(..\tile.bmp)");

	sd.spheres[0] = Sphere(Point(sinf(0) * 3, -1, 8 + cosf(0) * 3), 1);
	sd.spheres[0].mirror = true;

	sd.spheres[1] = Sphere(Point(5, -1, 5), 1);
	sd.spheres[1].color.r = 50;
	sd.spheres[1].color.g = 200;
	sd.spheres[1].color.b = 100;
	sd.spheres[1].mirror = true;

	sd.lights[0] = Light(Sphere(Point(-100, 100, -50), 10), .2f);
	sd.lights[0].color.r = 239;
	sd.lights[0].color.g = 163;
	sd.lights[0].color.b = 56;

	sd.triangles[0] = Triangle(Point(7, -2, 0), Point(-7, -2, 0), Point(7, -2, 21));
	sd.triangles[0].textured = true;
	sd.triangles[0].texIndex = 0;
	sd.triangles[0].t0 = Point(1, 0, 0);
	sd.triangles[0].t1 = Point(0, 0, 0);
	sd.triangles[0].t2 = Point(1, 1, 0);

	sd.triangles[1] = Triangle(Point(-7, -2, 0), Point(-7, -2, 21), Point(7, -2, 21));
	sd.triangles[1].textured = true;
	sd.triangles[1].texIndex = 0;
	sd.triangles[1].t0 = Point(0, 0, 0);
	sd.triangles[1].t1 = Point(0, 1, 0);
	sd.triangles[1].t2 = Point(1, 1, 0);

	sd.triangles[2] = Triangle(Point(-5, -2, 4), Point(-5.5f, 2, 6), Point(-5, -2, 8));
	sd.triangles[2].mirror = true;

	InitDrawing(arr);


}

void testKeys() {
	bool changed = false;
	float speed = 0.1f, angSpeed = 0.04f;
	if (GetFocus() == win) {
		if (GetAsyncKeyState(VK_DOWN)) {
			sd.camYang += angSpeed;
			changed = true;
		}
		if (GetAsyncKeyState(VK_UP)) {
			sd.camYang -= angSpeed;
			changed = true;
		}
		if (GetAsyncKeyState(VK_LEFT)) {
			sd.camXang += angSpeed;
			changed = true;
		}
		if (GetAsyncKeyState(VK_RIGHT)) {
			sd.camXang -= angSpeed;
			changed = true;
		}
		if (GetAsyncKeyState(VK_NUMPAD4)) {
			sd.expMultiplier *= 1.1;
			changed = true;
		}
		if (GetAsyncKeyState(VK_NUMPAD1)) {
			sd.expMultiplier /= 1.1;
			changed = true;
		}
		if (GetAsyncKeyState(VK_NUMPAD5)) {
			sd.focalDistance *= 1.03;
			changed = true;
		}
		if (GetAsyncKeyState(VK_NUMPAD2)) {
			sd.focalDistance /= 1.03;
			changed = true;
		}
		if (GetAsyncKeyState(VK_NUMPAD6)) {
			sd.dofStr *= 1.1;
			changed = true;
		}
		if (GetAsyncKeyState(VK_NUMPAD3)) {
			sd.dofStr /= 1.1;
			changed = true;
		}
		if (GetAsyncKeyState(VK_OEM_PLUS)) {
			sd.gamma *= 1.01f;
			changed = true;
		}
		if (GetAsyncKeyState(VK_OEM_MINUS)) {
			sd.gamma /= 1.01f;
			changed = true;
		}
		if (GetAsyncKeyState(0x31)) { //1
			changed = true;
		}
		if (GetAsyncKeyState(0x57)) { //W
			sd.camera = sd.camera + sd.c2S * (speed / sd.camDist);
			changed = true;
		}
		if (GetAsyncKeyState(0x53)) { //S
			sd.camera = sd.camera - sd.c2S * (speed / sd.camDist);
			changed = true;
		}
		if (GetAsyncKeyState(0x41)) { //A
			sd.camera = sd.camera - sd.sR * speed;
			changed = true;
		}
		if (GetAsyncKeyState(0x44)) { //D
			sd.camera = sd.camera + sd.sR * speed;
			changed = true;
		}
		if (GetAsyncKeyState(0x43)) { //C
			sd.camera = sd.camera + sd.sD * speed;
			changed = true;
		}
		if (GetAsyncKeyState(0x5A)) { //Z
			sd.camera = sd.camera - sd.sD * speed;
			changed = true;
		}
		if (GetAsyncKeyState(0x52)) { //R
			sd.reset = true;
			changed = true;
		}
		if (GetAsyncKeyState(0x51)) { //Q
			sd.camDist *= 1.1f;
			changed = true;
		}
		if (GetAsyncKeyState(0x45)) { //E
			sd.camDist /= 1.1f;
			changed = true;
		}
		if (GetAsyncKeyState(VK_RBUTTON)) {
			POINT p;
			GetCursorPos(&p);
			sd.camXang = (p.x * 3.1415f) / (XRES / 4) - 3.1415f;
			sd.camYang = (p.y * 3.1415f) / (YRES / 4) - 3.1415f;
		}
	}
	if (changed) {
		InitFrame();
	}
}

void key(WPARAM wParam, LPARAM lParam) {
	if (wParam == 0x46) { // F
		sd.realTime = !sd.realTime;
	}
	if (wParam == 0x42) { //B
		sd.bilinearTexture = !sd.bilinearTexture;
	}
	InitFrame();
}

void draw(WPARAM wParam, LPARAM lParam) {

	testKeys();

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


int main() {

	EVENTFUNC functions[]{
		EVENTFUNC {WM_PAINT, draw},
		EVENTFUNC {WM_KEYDOWN, key},
		EVENTFUNC {WM_CREATE, initial},
	};

	DoGL(3, functions, 1920, 1080);
}