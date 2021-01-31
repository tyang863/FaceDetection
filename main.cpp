/*
This program detects human faces in pictures, label the faces with rectangles, and add a hat image to the top of each face.
The pictures are taken by the camera of the device running this program.
Implemented with OpenCV, the detection procedure use cascaded Adaboost classifiers to detect human faces based on Harr features.
*/
#include "opencv2/opencv.hpp"
#include <iostream>
using namespace std;
using namespace cv;

void DetectAndHat(Mat &img, CascadeClassifier &classifier, double scale)
{
    vector<Rect> faces;
    const static Scalar color = CV_RGB(0, 0, 255);

    //Preprocessing of the image: graying, shrinking and histogram equalizing
    Mat gray, ShrinkedImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8SC1);
    cvtColor(img, gray, CV_BGR2GRAY);
    resize(gray, ShrinkedImg, ShrinkedImg.size(), 0, 0, INTER_AREA);
    equalizeHist(ShrinkedImg, ShrinkedImg);

    //Detect human faces in multiple scales
    classifier.detectMultiScale(ShrinkedImg, faces, 1.1, 3, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30));

    //Label the detected faces with rectangles and add one hat image to the top of each face
    Mat hat, AdaptiveHat;
    hat = imread("C://Users/narut/Desktop/hat.png");
    uchar *data_target;
    uchar *data_source;
    for (vector<Rect>::const_iterator r = faces.begin(), int i = 0; r != faces.end(); r++, i++)
    {
        //Add rectangles
        rectangle(img,
                  cvPoint(cvRound(r->x * scale), cvRound(r->y * scale)),
                  cvPoint(cvRound((r->x + r->width - 1) * scale), cvRound((r->y + r->height - 1) * scale)),
                  color, 2, 8, 0);

        //Adjust the hat image to proper size and add it to the top of the faces
        resize(hat, AdaptiveHat, Size(cvRound(r->width * scale), cvRound(r->width * scale * hat.rows / hat.cols)));
        if (cvRound(r->y * scale) - AdaptiveHat.rows > 0)
        {
            for (int j_target = cvRound(r->y * scale) - AdaptiveHat.rows, j_source = 0; j_target < cvRound(r->y * scale); j_target++, j_source++)
            {
                data_target=img.ptr<uchar>(j_target);
                data_source=AdaptiveHat.ptr<uchar>(j_source);
                for (int i_target = cvRound(r->x * scale) * 3, i_source=0; i_target < cvRound(r->x * scale) * 3 + AdaptiveHat.cols * 3; i_target+=3, i_source+=3)
                {
                    if (data_source[i_source] < 255 && data_source[i_source + 1] < 255 && data_source[i_source + 2] < 255) //Exclude white background
                    {
                        //Set color value for each channel of RGB
                        data_target[i_target] = data_source[i_source];
                        data_target[i_target + 1] = data_source[i_source + 1];
                        data_target[i_target + 2] = data_source[i_source + 2];
                    }
                }
            }
        }
    }

    imshow("result", img);
}

int main()
{
    //Open the camera
    VideoCapture cap(0);
    if(!cap.isOpened())
    {
        cout << "Fail to open the camera!" << endl;
        return -1;
    }

    //Load the Cascade Adaboost Classifier
    Mat frame;
    CascadeClassifier classifier;
    classifier.load("D://codeblocks/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml");

    //Process the pictures
    int key;
    while(true)
    {
        cap >> frame;
        DetectAndHat(frame, classifier, 2);
        key = waitKey(10);
        if (key == 27) //Stop detecting when pressing 'ESC'
            break;
    }
    return 0;
}
