#include <stdio.h>
#include <Windows.h>
#include <gl\GL.h>
#include <gl\GLU.h>

#include "GLInit.h"

HGLRC glrc = 0;
HWND win;
int pxFormat = 0;
HDC hdc = 0;
PIXELFORMATDESCRIPTOR pfd;
EVENTFUNC *functions;
int fCount;

void doFunctionFromMessage(UINT Msg, WPARAM wParam, LPARAM lParam) {
	int i = 0;
	for (i = 0; i < fCount; i++) {
		if (functions[i].eventNum == Msg) {
			(*functions[i].func)(wParam, lParam);
			return;
		}
	}
}

LRESULT CALLBACK msgProc(HWND hWnd, UINT Msg, WPARAM wParam, LPARAM lParam) {
	switch (Msg) {
	case WM_CREATE:
		pfd =
		{
			sizeof(PIXELFORMATDESCRIPTOR),
			1,
			PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,    // Flags
			PFD_TYPE_RGBA,        // The kind of framebuffer. RGBA or palette.
			32,                   // Colordepth of the framebuffer.
			0, 0, 0, 0, 0, 0,
			0,
			0,
			0,
			0, 0, 0, 0,
			24,                   // Number of bits for the depthbuffer
			8,                    // Number of bits for the stencilbuffer
			0,                    // Number of Aux buffers in the framebuffer.
			PFD_MAIN_PLANE,
			0,
			0, 0, 0
		};

		hdc = GetDC(hWnd);

		pxFormat = ChoosePixelFormat(hdc, &pfd);

		SetPixelFormat(hdc, pxFormat, &pfd);

		glrc = wglCreateContext(hdc);

		wglMakeCurrent(hdc, glrc);

		doFunctionFromMessage(Msg, wParam, lParam);

		break;
	case WM_PAINT:

		doFunctionFromMessage(Msg, wParam, lParam);

		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
		break;
	case WM_CLOSE:
		wglMakeCurrent(hdc, NULL);
		wglDeleteContext(glrc);
		PostQuitMessage(0);
	default:

		doFunctionFromMessage(Msg, wParam, lParam);

		DefWindowProc(hWnd, Msg, wParam, lParam);
	}
	return 1;
}

void DoGL(int count, EVENTFUNC fcns[], int xres, int yres) {
	fCount = count;
	functions = fcns;
	WNDCLASS myGLClass;
	memset(&myGLClass, 0, sizeof(WNDCLASS));

	myGLClass.lpfnWndProc = msgProc;
	myGLClass.lpszClassName = "myGLClass";
	myGLClass.style = CS_OWNDC;

	int a = RegisterClass(&myGLClass);

	win = CreateWindow("myGLClass", "Test", WS_OVERLAPPED | WS_POPUP, 0, 0, xres, yres, 0, 0, 0, 0);

	ShowWindow(win, SW_SHOW);

	MSG msg;

	while (GetMessage(&msg, NULL, 0, 0)) {

		if (msg.message == WM_QUIT) return;

		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
}