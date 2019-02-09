#pragma once

#define XRES 3200
#define YRES 1800

void drawPixel(float x, float y, char *pix);

void InitFrame();

void InitDrawing(char *ptr);

void DrawFrame();