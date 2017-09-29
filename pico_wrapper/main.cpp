#include <iostream>
#include <pico/pico_helper.h>
#include <opencv2/opencv.hpp>

#include <fstream>

using namespace std;
using namespace cv;

void read_input(const char* path, vector<string>& files)
{
    string line;
    ifstream myfile (path);
    if (myfile.is_open())
    {
        while (getline(myfile,line))
        {
          files.push_back(line);
        }
        myfile.close();
    }
}

void simple_test()
{
    string path = "/Users/bbphuc/Desktop/myface.jpg"
    vector<detection_result> result;
    FaceDetector* fd = new PICO_FaceDetecor();
    fd->detect(img, result);
    fd->drawRect(img, img, result, DRAW_REC);
    delete fd;
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        return -1;
    }
    char* cascade = argv[1];
    char* inputfile = argv[2];

    vector<detection_result> result;
    FaceDetector* fd = new PICO_FaceDetecor();
    fd->loadClassifierData(cascade);
    vector<string> listFiles;
    read_input(inputfile, listFiles);

    for (int i = 0; i < listFiles.size(); ++i) {
        Mat img = imread(listFiles[i] + ".jpg");
        fd->detect(img, result);

        cout << listFiles[i] << endl;
        cout << result.size() << endl;
        for (auto it = result.begin(); it != result.end(); ++it) {
            Rect& rect = (*it).rect;
            printf("%d %d %d %d %.3f\n", rect.x, rect.y, rect.width, rect.height, (*it).score);
        }
        result.clear();
    }

    fd->drawRect(img, img, result, DRAW_REC);
    delete fd;

    //imshow("Result", img);
    //waitKey(0);

    return 0;
}
