#include <stdio.h>
#include <Windows.h>
#include <gl\GL.h>
#include <gl\GLU.h>
#include <math.h>
#include "OpenGL3.1.h"

#include "GLInit.h"
#include "Drawing.h"
#include "SceneLoader.h"

GLuint tex;
char *arr, *tarr;

void initial(WPARAM wParam, LPARAM lParam) {

#ifdef CUDA
	cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
#endif

	LoadShaderFunctions();

	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, XRES, YRES, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	arr = (char*)malloc(XRES * YRES * 3);

	SceneLoader sl;
	sl.loadScene(R"(..\house.scene)", sd);

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
		/*if (GetAsyncKeyState('1')) {
			changed = true;
		}*/
		if (GetAsyncKeyState('W')) {
			sd.camera = sd.camera + sd.c2S * (speed / sd.camDist);
			changed = true;
		}
		if (GetAsyncKeyState('S')) {
			sd.camera = sd.camera - sd.c2S * (speed / sd.camDist);
			changed = true;
		}
		if (GetAsyncKeyState('A')) {
			sd.camera = sd.camera - sd.sR * speed;
			changed = true;
		}
		if (GetAsyncKeyState('D')) {
			sd.camera = sd.camera + sd.sR * speed;
			changed = true;
		}
		if (GetAsyncKeyState('C')) {
			sd.camera = sd.camera + sd.sD * speed;
			changed = true;
		}
		if (GetAsyncKeyState('Z')) {
			sd.camera = sd.camera - sd.sD * speed;
			changed = true;
		}
		if (GetAsyncKeyState('R')) {
			sd.reset = true;
			changed = true;
		}
		if (GetAsyncKeyState('Q')) {
			sd.camDist *= 1.1f;
			changed = true;
		}
		if (GetAsyncKeyState('E')) {
			sd.camDist /= 1.1f;
			changed = true;
		}
		// These keys move spheres[0] and tweak the refraction index of the lens
		// (spheres[1]). The house.scene has no spheres, so they're commented out
		// to avoid indexing past the end of the (empty) sphere array. Re-enable
		// them when loading a scene that actually contains spheres.
		/*if (GetAsyncKeyState('I')) {
			sd.spheres[0].c.z += 0.03f;
			changed = true;
		}
		if (GetAsyncKeyState('K')) {
			sd.spheres[0].c.z -= 0.03f;
			changed = true;
		}
		if (GetAsyncKeyState('J')) {
			sd.spheres[0].c.x += 0.03f;
			changed = true;
		}
		if (GetAsyncKeyState('L')) {
			sd.spheres[0].c.x -= 0.03f;
			changed = true;
		}
		if (GetAsyncKeyState('O')) {
			sd.spheres[0].c.y += 0.03f;
			changed = true;
		}
		if (GetAsyncKeyState('U')) {
			sd.spheres[0].c.y -= 0.03f;
			changed = true;
		}
		if (GetAsyncKeyState('1')) {
			sd.spheres[1].mat.refIndex *= 1.03f;
			changed = true;
		}
		if (GetAsyncKeyState('2')) {
			sd.spheres[1].mat.refIndex /= 1.03f;
			changed = true;
		}*/
		/*if (GetAsyncKeyState(VK_RBUTTON)) {
			POINT p;
			GetCursorPos(&p);
			sd.camXang = (p.x * 3.1415f) / (XRES / 4) - 3.1415f;
			sd.camYang = (p.y * 3.1415f) / (YRES / 4) - 3.1415f;
		}*/
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
	if (wParam == 0x30) { //0
		sd.bounces += 1;
	}
	if (wParam == 0x39) { //9
		sd.bounces -= 1;
	}
	if(wParam == 'N') {
		sd.useLightPredict = !sd.useLightPredict;
		if (!sd.useLightPredict) sd.useMirrorPredict = false; // mirror NEE requires light NEE
	}
	if(wParam == 'M') { // mirror-caustic NEE; only meaningful while light predict (N) is on
		if (sd.useLightPredict) sd.useMirrorPredict = !sd.useMirrorPredict;
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

	printf(R"(
RAY-TRACER
Keys:
	Arrows - Camera rotation
	WASD - movement (camera relative)
	C, Z - move up - down (camera relative)
	Q, E - FOV
	Num 4, 1 - Exposure control
	Num 5, 2 - Focal distance
	Num 6, 3 - DOF (Aparature size)
	-, + - gamma control
	9, 0 - Ray bounces (less / more)
	F - Toggle real-time / accumulation mode
	B - Toggle bilinear texture filtering
	R - Reset render
---------------------------------------------
)");

	EVENTFUNC functions[]{
		EVENTFUNC {WM_PAINT, draw},
		EVENTFUNC {WM_KEYDOWN, key},
		EVENTFUNC {WM_CREATE, initial},
	};

	DoGL(3, functions, 1920, 1080);
}