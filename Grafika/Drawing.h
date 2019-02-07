#pragma once

#define XRES 1280
#define YRES 720

void drawPixel(float x, float y, char *pix);

void InitFrame();

void InitDrawing(char *ptr);

void DrawFrame();