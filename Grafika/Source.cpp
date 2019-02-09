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

	arr = (char*)malloc(XRES * YRES * 3);

	InitDrawing(arr);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);


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

int main() {

	EVENTFUNC functions[]{
		EVENTFUNC {WM_PAINT, draw},
		EVENTFUNC {WM_CREATE, initial},
	};

	DoGL(2, functions, 1600, 900);
}