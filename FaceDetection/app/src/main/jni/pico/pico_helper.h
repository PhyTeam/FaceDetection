#ifndef PICO_HELPER_H
#define PICO_HELPER_H
#include <opencv2/opencv.hpp>
#include <vector>

void process_image(IplImage* frame, int draw, int isUsePy, std::vector<cv::Rect>&);
void init(const char* npdCascadePath);

#endif // PICO_HELPER_H
