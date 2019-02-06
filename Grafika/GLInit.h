#pragma once

typedef struct EventFunc {
	UINT eventNum;
	void (*func)(WPARAM, LPARAM);
} EVENTFUNC;

void DoGL(int count, EVENTFUNC fcns[], int xres, int yres);
