#include <string>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

#ifndef PICO_HELPER_H
#define PICO_HELPER_H

#define USED 1
#define UNUSED 0

#define DRAW_REC 0
#define DRAW_CIR 1

struct detection_result {
    cv::Rect rect;
    float score;
};

struct classifier_params {

    int max_size;
    int min_size;

    float angle;

    float scaleFactor;
    float strideFactor;
    float qthreshold;
    int usepyr;
    int noclustering;

    void* cascade = 0;

    classifier_params() {
        // Default parameters
        min_size = 30;
        max_size = 400;

        angle = 0.0f;
        scaleFactor = 1.1;
        strideFactor = 0.1;

        qthreshold = 10.0f;
        usepyr = UNUSED;
        noclustering = UNUSED;
    }

    ~classifier_params() {
    }
};

void process_image(cv::Mat frame, classifier_params params, std::vector<detection_result>& result, bool verbose = false);
void process_image(IplImage* frame, int draw, int isUsePy, std::vector<cv::Rect>&);

void init(const char* path, bool verbose = false);
void load_pico_cascade(const char* path, classifier_params* params, bool verbose = false);

class FaceDetector
{
private:
    std::string cascadePath;
public:
    FaceDetector() {}
    virtual ~FaceDetector() {}

    virtual void loadClassifierData(const std::string& cascadePath) = 0;
    virtual void detect(Mat img, vector<Rect>& result)
    {
        TickMeter t;
        t.start();

        _detect(img, result);

        t.stop();
        //cout << "Estimate time is " << t.getTimeMilli() << "milisecs" << endl;
    }

    virtual void detect(Mat img, vector<detection_result> &result)
    {
        _detect(img, result);
    }

    virtual void drawRect(const Mat& src, Mat& output, vector<Rect>& faces, int mode = DRAW_REC);

protected:
    virtual void _detect(Mat img, vector<Rect>& result) = 0;
    virtual void _detect(Mat img, vector<detection_result> &result) = 0;
};

class PICO_FaceDetecor: public FaceDetector
{
private:
    classifier_params params;
public:
    ~PICO_FaceDetecor();
    void loadClassifierData(const string& path);
protected:
    void _detect(Mat img, vector<Rect>& result);
    void _detect(Mat img, vector<detection_result>& result);
};



#endif // PICO_HELPER_H
