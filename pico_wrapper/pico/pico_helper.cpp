/*
 *  This code is released under the MIT License.
 *  Copyright (c) 2013 Nenad Markus
 */

#include <stdio.h>

// OpenCV 3.x required
// depending on your computer configuration (OpenCV install path), the following line might need modifications
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <vector>

using namespace std;
using namespace cv;

#include <time.h>
#include <unistd.h>
#include "pico_helper.h"

extern "C"{
#include "pico/picornt.h"
}
/*
    a portable time function
*/

#ifdef __GNUC__
#include <time.h>
float getticks()
{
    return cvGetTickCount();
}
#else
#include <windows.h>
float getticks()
{
    static double freq = -1.0;
    LARGE_INTEGER lint;

    if(freq < 0.0)
    {
        if(!QueryPerformanceFrequency(&lint))
            return -1.0f;

        freq = lint.QuadPart;
    }

    if(!QueryPerformanceCounter(&lint))
        return -1.0f;

    return (float)( lint.QuadPart/freq );
}
#endif

/*

*/

void* cascade = 0;

int minsize;
int maxsize;

float angle;

float scalefactor;
float stridefactor;

float qthreshold;

int usepyr;
int noclustering;
int verbose;

void process_image(Mat frame, classifier_params params, std::vector<detection_result>& result, bool verbose) {
    int i, j;
    float t;

    uint8_t* pixels;
    int nrows, ncols, ldim;

    #define MAXNDETECTIONS 2048
    int ndetections;
    float rcsq[4*MAXNDETECTIONS];
    /* static */
    Mat gray;
    /* static */
    Mat pyr[5];

    if(!pyr[0].data)
    {
        //
        gray = Mat(Size(frame.cols, frame.rows), CV_8UC1);
        //
        pyr[0] = gray;
        pyr[1] = Mat(Size(frame.cols / 2, frame.rows / 2), CV_8UC1);
        pyr[2] = Mat(Size(frame.cols / 4, frame.rows / 4), CV_8UC1);
        pyr[3] = Mat(Size(frame.cols / 8, frame.rows / 8), CV_8UC1);
        pyr[4] = Mat(Size(frame.cols / 16, frame.rows / 16), CV_8UC1);
    }

    // get grayscale image
    if(frame.channels() == 3)
        cvtColor(frame, gray, CV_RGB2GRAY);
    else
        frame.copyTo(gray);

    // perform detection with the pico library
    t = getticks();

    if(params.usepyr)
    {
        int nd;

        pyr[0] = gray;

        pixels = (uint8_t*)pyr[0].data;
        nrows = pyr[0].rows;
        ncols = pyr[0].cols;
        ldim = pyr[0].step;

        ndetections = find_objects(rcsq, MAXNDETECTIONS, params.cascade, params.angle, pixels, nrows, ncols, ldim, params.scaleFactor, params.strideFactor,
                                   MAX(16, params.min_size), MIN(128, params.max_size));

        for(i=1; i<5; ++i)
        {
            resize(pyr[i-1], pyr[i], pyr[i].size(), CV_INTER_LINEAR);

            pixels = (uint8_t*)pyr[i].data;
            nrows = pyr[i].rows;
            ncols = pyr[i].cols;
            ldim = pyr[i].step;

            nd = find_objects(&rcsq[4*ndetections], MAXNDETECTIONS-ndetections, params.cascade, params.angle, pixels, nrows, ncols, ldim,
                    params.scaleFactor, params.strideFactor,
                    MAX(64, (params.min_size)>>i), MIN(128, (params.max_size)>>i));

            for(j=ndetections; j<ndetections+nd; ++j)
            {
                rcsq[4*j+0] = (1<<i)*rcsq[4*j+0];
                rcsq[4*j+1] = (1<<i)*rcsq[4*j+1];
                rcsq[4*j+2] = (1<<i)*rcsq[4*j+2];
            }

            ndetections = ndetections + nd;
        }
    }
    else
    {
        //
        pixels = (uint8_t*)gray.data;
        nrows = gray.rows;
        ncols = gray.cols;
        ldim = gray.step;

        // Get max faces
        ndetections = find_objects(rcsq, MAXNDETECTIONS, params.cascade, params.angle,
                                   pixels, nrows, ncols, ldim,
                                   params.scaleFactor, params.strideFactor, params.min_size, MIN(nrows, ncols));
    }

    if(!params.noclustering)
        ndetections = cluster_detections(rcsq, ndetections);

    t = getticks() - t;

    // if the `verbose` flag is set, print the results to standard output
    if(verbose)
    {
        for(i=0; i<ndetections; ++i)
            if(rcsq[4*i+3]>=params.qthreshold) // check the confidence threshold
                printf("%d %d %d %f\n", (int)rcsq[4*i+0], (int)rcsq[4*i+1], (int)rcsq[4*i+2], rcsq[4*i+3]);

        //
        //printf("# %f\n", 1000.0f*t); // use '#' to ignore this line when parsing the output of the program
    }

    for(i=0; i<ndetections; ++i)
        if(rcsq[4*i+3]>=params.qthreshold) // check the confidence threshold
        {
            detection_result ret;
            ret.rect = cv::Rect(rcsq[4*i+1] - rcsq[4*i+2]/2, rcsq[4*i+0] - rcsq[4*i+2]/2, rcsq[4*i+2], rcsq[4*i+2]);
            ret.score = rcsq[4*i+3];
            result.push_back(ret);
        }
}

void process_image(IplImage* frame, int draw, int isUsePy, std::vector<cv::Rect>& result)
{
    usepyr = isUsePy;
    int i, j;
    float t;

    uint8_t* pixels;
    int nrows, ncols, ldim;

    #define MAXNDETECTIONS 2048
    int ndetections;
    float rcsq[4*MAXNDETECTIONS];
    /* static */
    IplImage* gray = 0;
    /* static */
    IplImage* pyr[5] = {0, 0, 0, 0, 0};

    /*
        ...
    */

    //
    if(!pyr[0])
    {
        //
        gray = cvCreateImage(cvSize(frame->width, frame->height), frame->depth, 1);

        //
        pyr[0] = gray;
        pyr[1] = cvCreateImage(cvSize(frame->width/2, frame->height/2), frame->depth, 1);
        pyr[2] = cvCreateImage(cvSize(frame->width/4, frame->height/4), frame->depth, 1);
        pyr[3] = cvCreateImage(cvSize(frame->width/8, frame->height/8), frame->depth, 1);
        pyr[4] = cvCreateImage(cvSize(frame->width/16, frame->height/16), frame->depth, 1);
    }

    // get grayscale image
    if(frame->nChannels == 3)
        cvCvtColor(frame, gray, CV_RGB2GRAY);
    else
        cvCopy(frame, gray, 0);

    // perform detection with the pico library
    t = getticks();

    if(usepyr)
    {
        int nd;

        //
        pyr[0] = gray;

        pixels = (uint8_t*)pyr[0]->imageData;
        nrows = pyr[0]->height;
        ncols = pyr[0]->width;
        ldim = pyr[0]->widthStep;

        ndetections = find_objects(rcsq, MAXNDETECTIONS, cascade, angle, pixels, nrows, ncols, ldim, scalefactor, stridefactor, MAX(16, minsize), MIN(128, maxsize));

        for(i=1; i<5; ++i)
        {
            cvResize(pyr[i-1], pyr[i], CV_INTER_LINEAR);

            pixels = (uint8_t*)pyr[i]->imageData;
            nrows = pyr[i]->height;
            ncols = pyr[i]->width;
            ldim = pyr[i]->widthStep;

            nd = find_objects(&rcsq[4*ndetections], MAXNDETECTIONS-ndetections, cascade, angle, pixels, nrows, ncols, ldim, scalefactor, stridefactor, MAX(64, minsize>>i), MIN(128, maxsize>>i));

            for(j=ndetections; j<ndetections+nd; ++j)
            {
                rcsq[4*j+0] = (1<<i)*rcsq[4*j+0];
                rcsq[4*j+1] = (1<<i)*rcsq[4*j+1];
                rcsq[4*j+2] = (1<<i)*rcsq[4*j+2];
            }

            ndetections = ndetections + nd;
        }
    }
    else
    {
        //
        pixels = (uint8_t*)gray->imageData;
        nrows = gray->height;
        ncols = gray->width;
        ldim = gray->widthStep;

        //
        ndetections = find_objects(rcsq, MAXNDETECTIONS, cascade, angle, pixels, nrows, ncols, ldim, scalefactor, stridefactor, minsize, MIN(nrows, ncols));
    }

    if(!noclustering)
        ndetections = cluster_detections(rcsq, ndetections);

    t = getticks() - t;

    // if the flag is set, draw each detection
    if(draw)
        for(i=0; i<ndetections; ++i)
            if(rcsq[4*i+3]>=qthreshold) // check the confidence threshold
                cvCircle(frame, cvPoint(rcsq[4*i+1], rcsq[4*i+0]), rcsq[4*i+2]/2, CV_RGB(255, 0, 0), 4, 8, 0); // we draw circles here since height-to-width ratio of the detected face regions is 1.0f

    // if the `verbose` flag is set, print the results to standard output
    if(verbose)
    {
        //
        for(i=0; i<ndetections; ++i)
            if(rcsq[4*i+3]>=qthreshold) // check the confidence threshold
                printf("%d %d %d %f\n", (int)rcsq[4*i+0], (int)rcsq[4*i+1], (int)rcsq[4*i+2], rcsq[4*i+3]);

        //
        //printf("# %f\n", 1000.0f*t); // use '#' to ignore this line when parsing the output of the program
    }

    for(i=0; i<ndetections; ++i)
        if(rcsq[4*i+3]>=qthreshold) // check the confidence threshold
            result.push_back(cv::Rect(rcsq[4*i+1] - rcsq[4*i+2]/2, rcsq[4*i+0] - rcsq[4*i+2]/2, rcsq[4*i+2], rcsq[4*i+2]));
}

void load_pico_cascade(const char* path, classifier_params* params, bool verbal)
{
    if (!params) {
        return;
    }

    size_t size;
    FILE* file;
    file = fopen(path, "rb");

    if(!file)
    {
        printf("# cannot read cascade from '%s'\n", path);
        return;
    }

    fseek(file, 0L, SEEK_END);
    size = ftell(file);
    fseek(file, 0L, SEEK_SET);

    params->cascade = malloc(size);

    if(!(params->cascade) || size!=fread(params->cascade, 1, size, file)) {
        if (file) fclose(file);
        return;
    }
    //
    fclose(file);

    if(verbal)
    {
        printf("# cascade parameters:\n");
        printf("#	tsr = %f\n", ((float*)params->cascade)[0]);
        printf("#	tsc = %f\n", ((float*)params->cascade)[1]);
        printf("#	tdepth = %d\n", ((int*)params->cascade)[2]);
        printf("#	ntrees = %d\n", ((int*)params->cascade)[3]);
        printf("# detection parameters:\n");
        printf("#	minsize = %d\n", params->min_size);
        printf("#	maxsize = %d\n", params->max_size);
        printf("#	scalefactor = %f\n", params->scaleFactor);
        printf("#	stridefactor = %f\n", params->strideFactor);
        printf("#	qthreshold = %f\n", params->qthreshold);
        printf("#	usepyr = %d\n", params->usepyr);
    }
}


void init(const char* npdCascadePath, bool verbal)
{
    char input[1024], output[1024];

    int size;
    FILE* file;

    //
    file = fopen(npdCascadePath, "rb");

    if(!file)
    {
        printf("# cannot read cascade from '%s'\n", npdCascadePath);
        return;
    }

    //
    fseek(file, 0L, SEEK_END);
    size = ftell(file);
    fseek(file, 0L, SEEK_SET);

    //
    cascade = malloc(size);

    if(!cascade || size!=fread(cascade, 1, size, file))
        return;

    //
    fclose(file);

    // set default parameters
    minsize = 10;
    maxsize = 1024;

    angle = 0.0f;

    scalefactor = 1.1f;
    stridefactor = 0.1f;

    qthreshold = 5.0f;

    usepyr = 0;
    noclustering = 0;
    verbose = 1;

    //
    input[0] = 0;
    output[0] = 0;

    if(verbal)
    {
        printf("# cascade parameters:\n");
        printf("#	tsr = %f\n", ((float*)cascade)[0]);
        printf("#	tsc = %f\n", ((float*)cascade)[1]);
        printf("#	tdepth = %d\n", ((int*)cascade)[2]);
        printf("#	ntrees = %d\n", ((int*)cascade)[3]);
        printf("# detection parameters:\n");
        printf("#	minsize = %d\n", minsize);
        printf("#	maxsize = %d\n", maxsize);
        printf("#	scalefactor = %f\n", scalefactor);
        printf("#	stridefactor = %f\n", stridefactor);
        printf("#	qthreshold = %f\n", qthreshold);
        printf("#	usepyr = %d\n", usepyr);
    }
}
/* =============================== FaceDetector class ============================*/

void FaceDetector::drawRect(const Mat &img, Mat& output,vector<Rect> &faces, int mode)
{
    output = img.clone();
    const static Scalar colors[] =
    {
        Scalar(255,0,0),
        Scalar(255,128,0),
        Scalar(255,255,0),
        Scalar(0,255,0),
        Scalar(0,128,255),
        Scalar(0,255,255),
        Scalar(0,0,255),
        Scalar(255,0,255)
    };
    output = img.clone();
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect r = faces[i];
        Point center;
        Scalar color = colors[i%8];
        int radius;


        double scale = 1.0f;

        double aspect_ratio = (double)r.width/r.height;

        if (mode == DRAW_REC) {
            rectangle( output, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
                       cvPoint(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)),
                       color, 3, 8, 0);
        } else if (mode == DRAW_CIR){
            if( 0.75 < aspect_ratio && aspect_ratio < 1.3 ) {
                center.x = cvRound((r.x + r.width*0.5)*scale);
                center.y = cvRound((r.y + r.height*0.5)*scale);
                radius = cvRound((r.width + r.height)*0.25*scale);
                circle( output, center, radius, color, 2, 8, 0 );
            }
        }
    }
}

void PICO_FaceDetecor::loadClassifierData(const string &path)
{
    load_pico_cascade(path.c_str(), &params, false);
}

void PICO_FaceDetecor::_detect(Mat img, vector<Rect> &result)
{
    vector<detection_result> r;
    process_image(img, params, r, false);
    for (unsigned int i = 0; i < r.size(); i++) {
        result.push_back(r[i].rect);
    }
}

void PICO_FaceDetecor::_detect(Mat img, vector<detection_result> &result)
{
    process_image(img, params, result, false);
}

PICO_FaceDetecor::~PICO_FaceDetecor() {
    if (this->params.cascade) {
        free(this->params.cascade);
    }
}
