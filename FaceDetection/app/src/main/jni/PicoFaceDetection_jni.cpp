//
// Created by bbphuc on 8/30/17.
//
#include <jni.h>

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

#include "PicoFaceDetection_jni.h"
#include "pico/pico_helper.h"
#include <android/log.h>

#ifndef LOGD
#define LOG_TAG "FaceDetection/PICO"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#endif

using namespace std;
using namespace cv;


JNIEXPORT void JNICALL
Java_com_example_bbphuc_facedetection_FdActivity_nativeLoadTrainedModel(JNIEnv * jenv, jclass type,
                                                                        jstring jpath)
{
    const char* path = jenv->GetStringUTFChars(jpath, 0);
    init(path); // Initialize default params
}

void init(VideoWriter& writer, Size size) {
    writer.open("/sdcard/DCIM/test/test.avi", CV_FOURCC('D','I','V','X'), 120, size, true );
}


JNIEXPORT void JNICALL
Java_com_example_bbphuc_facedetection_FdActivity_nativePICDetect(JNIEnv *env, jclass type,
                                                                 jlong rgb, jlong grayImage, jlong addrFaces, int rot)
{
    Mat img = *((Mat *) grayImage);//imread("/sdcard/DCIM/test/FB_IMG_14763484633.jpg");
    cv::TickMeter t;
    t.start();
    float scale = std::max(img.cols, img.rows)/160;
    resize(img, img, Size(), 1/scale, 1/scale);
    // scale frame
    IplImage* iImg = new IplImage(img);
    vector<Rect> result;
    // Generate rect
    process_image(iImg, 1, 0, result);
    delete iImg;
    // Process result
    for (int i = 0; i < result.size(); i++) {
        Rect& rect = result[i];
        rect.x *= scale;
        rect.y *= scale;
        rect.width *= scale;
        rect.height *= scale;
    }


    t.stop();
    LOGD("Run time : %f" , t.getTimeMilli());
    *((Mat *) addrFaces) = Mat(result, true);
}
