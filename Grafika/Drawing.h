#pragma once

#define XRES 1920
#define YRES 1080

#define NONRT

void drawPixel(float x, float y, char *pix);

void InitFrame();

void InitDrawing(char *ptr);

void DrawFrame();