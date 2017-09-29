//
// Created by bbphuc on 8/30/17.
//
#include <jni.h>

#ifndef FACEDETECTION_PICOFACEDETECTION_JNI_H
#define FACEDETECTION_PICOFACEDETECTION_JNI_H



#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL
        Java_com_example_bbphuc_facedetection_FdActivity_nativeLoadTrainedModel(JNIEnv *env, jclass type,
        jstring path);


JNIEXPORT void JNICALL
Java_com_example_bbphuc_facedetection_FdActivity_nativePICDetect(JNIEnv *env, jclass type,
        jlong grayImageAddr, jlong addrFaces, int rot);

#ifdef __cplusplus
}
#endif
#endif //FACEDETECTION_PICOFACEDETECTION_JNI_H
