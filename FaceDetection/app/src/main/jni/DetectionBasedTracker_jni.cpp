#include <DetectionBasedTracker_jni.h>
#include <npd/npddetect.h>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

#include <string>
#include <vector>

#include <android/log.h>
#ifndef LOGD
#define LOG_TAG "FaceDetection/DetectionBasedTracker"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#endif
using namespace std;
using namespace cv;

inline void vector_Rect_to_Mat(vector<Rect> &v_rect, Mat &mat) {
    mat = Mat(v_rect, true);
}

class CascadeDetectorAdapter : public DetectionBasedTracker::IDetector {
public:
    CascadeDetectorAdapter(cv::Ptr<cv::CascadeClassifier> detector) :
            IDetector(),
            Detector(detector) {
        LOGD("CascadeDetectorAdapter::Detect::Detect");
        CV_Assert(detector);
    }

    void detect(const cv::Mat &Image, std::vector<cv::Rect> &objects) {
        LOGD("CascadeDetectorAdapter::Detect: begin");
        LOGD("CascadeDetectorAdapter::Detect: scaleFactor=%.2f, minNeighbours=%d, minObjSize=(%dx%d), maxObjSize=(%dx%d)",
             scaleFactor, minNeighbours, minObjSize.width, minObjSize.height, maxObjSize.width,
             maxObjSize.height);
        Detector->detectMultiScale(Image, objects, scaleFactor, minNeighbours, 0, minObjSize,
                                   maxObjSize);
        LOGD("CascadeDetectorAdapter::Detect: end");
    }

    virtual ~CascadeDetectorAdapter() {
        LOGD("CascadeDetectorAdapter::Detect::~Detect");
    }

private:
    CascadeDetectorAdapter();

    cv::Ptr<cv::CascadeClassifier> Detector;
};

struct DetectorAgregator {
    cv::Ptr<CascadeDetectorAdapter> mainDetector;
    cv::Ptr<CascadeDetectorAdapter> trackingDetector;

    cv::Ptr<DetectionBasedTracker> tracker;

    DetectorAgregator(cv::Ptr<CascadeDetectorAdapter> &_mainDetector,
                      cv::Ptr<CascadeDetectorAdapter> &_trackingDetector) :
            mainDetector(_mainDetector),
            trackingDetector(_trackingDetector) {
        CV_Assert(_mainDetector);
        CV_Assert(_trackingDetector);

        DetectionBasedTracker::Parameters DetectorParams;
        tracker = makePtr<DetectionBasedTracker>(mainDetector, trackingDetector, DetectorParams);
    }
};

JNIEXPORT jlong JNICALL
Java_com_example_bbphuc_facedetection_DetectionBasedTracker_nativeCreateObject
        (JNIEnv *jenv, jclass, jstring jFileName, jint faceSize) {
    LOGD("Java_com_example_bbphuc_facedetection_DetectionBasedTracker_nativeCreateObject enter");
    const char *jnamestr = jenv->GetStringUTFChars(jFileName, NULL);
    string stdFileName(jnamestr);
    jlong result = 0;

    LOGD("Java_com_example_bbphuc_facedetection_DetectionBasedTracker_nativeCreateObject");

    try {
        cv::Ptr<CascadeDetectorAdapter> mainDetector = makePtr<CascadeDetectorAdapter>(
                makePtr<CascadeClassifier>(stdFileName));
        cv::Ptr<CascadeDetectorAdapter> trackingDetector = makePtr<CascadeDetectorAdapter>(
                makePtr<CascadeClassifier>(stdFileName));
        result = (jlong) new DetectorAgregator(mainDetector, trackingDetector);
        if (faceSize > 0) {
            mainDetector->setMinObjectSize(Size(faceSize, faceSize));
            //trackingDetector->setMinObjectSize(Size(faceSize, faceSize));
        }
    }
    catch (cv::Exception &e) {
        LOGD("nativeCreateObject caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if (!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...) {
        LOGD("nativeCreateObject caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je,
                       "Unknown exception in JNI code of DetectionBasedTracker.nativeCreateObject()");
        return 0;
    }

    LOGD("Java_com_example_bbphuc_facedetection_DetectionBasedTracker_nativeCreateObject exit");
    return result;
}

JNIEXPORT void JNICALL
Java_com_example_bbphuc_facedetection_DetectionBasedTracker_nativeDestroyObject
        (JNIEnv *jenv, jclass, jlong thiz) {
    LOGD("Java_com_example_bbphuc_facedetection_DetectionBasedTracker_nativeDestroyObject");

    try {
        if (thiz != 0) {
            ((DetectorAgregator *) thiz)->tracker->stop();
            delete (DetectorAgregator *) thiz;
        }
    }
    catch (cv::Exception &e) {
        LOGD("nativeestroyObject caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if (!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...) {
        LOGD("nativeDestroyObject caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je,
                       "Unknown exception in JNI code of DetectionBasedTracker.nativeDestroyObject()");
    }
    LOGD("Java_com_example_bbphuc_facedetection_DetectionBasedTracker_nativeDestroyObject exit");
}

JNIEXPORT void JNICALL Java_com_example_bbphuc_facedetection_DetectionBasedTracker_nativeStart
        (JNIEnv *jenv, jclass, jlong thiz) {
    LOGD("Java_com_example_bbphuc_facedetection_DetectionBasedTracker_nativeStart");

    try {
        ((DetectorAgregator *) thiz)->tracker->run();
    }
    catch (cv::Exception &e) {
        LOGD("nativeStart caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if (!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...) {
        LOGD("nativeStart caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeStart()");
    }
    LOGD("Java_com_example_bbphuc_facedetection_DetectionBasedTracker_nativeStart exit");
}

JNIEXPORT void JNICALL Java_com_example_bbphuc_facedetection_DetectionBasedTracker_nativeStop
        (JNIEnv *jenv, jclass, jlong thiz) {
    LOGD("Java_com_example_bbphuc_facedetection_DetectionBasedTracker_nativeStop");

    try {
        ((DetectorAgregator *) thiz)->tracker->stop();
    }
    catch (cv::Exception &e) {
        LOGD("nativeStop caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if (!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...) {
        LOGD("nativeStop caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeStop()");
    }
    LOGD("Java_com_example_bbphuc_facedetection_DetectionBasedTracker_nativeStop exit");
}

JNIEXPORT void JNICALL Java_com_example_bbphuc_facedetection_DetectionBasedTracker_nativeSetFaceSize
        (JNIEnv *jenv, jclass, jlong thiz, jint faceSize) {
    LOGD("Java_com_example_bbphuc_facedetection_DetectionBasedTracker_nativeSetFaceSize -- BEGIN");

    try {
        if (faceSize > 0) {
            ((DetectorAgregator *) thiz)->mainDetector->setMinObjectSize(Size(faceSize, faceSize));
//((DetectorAgregator*)thiz)->trackingDetector->setMinObjectSize(Size(faceSize, faceSize));
        }
    }
    catch (cv::Exception &e) {
        LOGD("nativeStop caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if (!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...) {
        LOGD("nativeSetFaceSize caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je,
                       "Unknown exception in JNI code of DetectionBasedTracker.nativeSetFaceSize()");
    }
    LOGD("Java_com_example_bbphuc_facedetection_DetectionBasedTracker_nativeSetFaceSize -- END");
}


JNIEXPORT void JNICALL Java_com_example_bbphuc_facedetection_DetectionBasedTracker_nativeDetect
        (JNIEnv *jenv, jclass, jlong thiz, jlong imageGray, jlong faces) {
    LOGD("Java_com_example_bbphuc_facedetection_DetectionBasedTracker_nativeDetect");

    try {
        vector<Rect> RectFaces;
        ((DetectorAgregator *) thiz)->tracker->process(*((Mat *) imageGray));
        ((DetectorAgregator *) thiz)->tracker->getObjects(RectFaces);
        *((Mat *) faces) = Mat(RectFaces, true);
    }
    catch (cv::Exception &e) {
        LOGD("nativeCreateObject caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if (!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...) {
        LOGD("nativeDetect caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code DetectionBasedTracker.nativeDetect()");
    }
    LOGD("Java_com_example_bbphuc_facedetection_DetectionBasedTracker_nativeDetect END");
}


JNIEXPORT void JNICALL Java_com_example_bbphuc_facedetection_FdActivity_nativeNPDDetect
        (JNIEnv *jenv, jclass, jlong imageGray, jlong faces, jstring jpath)
{
    static npd::npddetect npd;
    Mat img = *((Mat *) imageGray);
    const char* path = jenv->GetStringUTFChars(jpath, 0);
    static bool isLoaded = false;
    if (!isLoaded) {
        npd.load(path);
        isLoaded = true;
    }
    //TIMER_BEGIN
    double t = (double)cvGetTickCount();
    int n = npd.detect(img.data, img.cols, img.rows);
    t = ((double)cvGetTickCount() - t) / ((double)cvGetTickFrequency()*1000.) ;

    LOGD("Detect num: %d (%lf ms)\n", n, t);
    //LOG("Detect %d face, cost %lf(ms)", n, TIMER_NOW);
    //TIMER_END

    vector< int >& Xs = npd.getXs();
    vector< int >& Ys = npd.getYs();
    vector< int >& Ss = npd.getSs();
    //vector< float >& Scores = npd.getScores();
    vector<Rect> RectFaces;
    for(int i = 0; i < n; i++)
    {
        //int deta = Ss[i]/10;
        //if(Scores[i] < 22.38)
        //    continue;

        //cv::rectangle(img, cv::Rect(Xs[i], Ys[i], Ss[i], Ss[i]), cv::Scalar(255, 0, 0));
        RectFaces.push_back(cv::Rect(Xs[i], Ys[i], Ss[i], Ss[i]));
    }

    *((Mat *) faces) = Mat(RectFaces, true);
}