/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    cout << "This program has " << argc << " arguments:" << endl;
    for (int i = 0; i < argc; ++i) {
        cout << argv[i] << endl;
    }
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)
    vector <int> total_vehicle_Keypoints;
    vector <int>total_keypoints;
    vector <float> average_size_detectedKP;
    vector <int> total_matched_keypoints;
    double Average_Time = 0.0;
    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = true;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize
        DataFrame frame;
        frame.cameraImg = imgGray;
        /* Skeleton :
            if ( data buffer is full )
            {
                remove the oldest data frame , then continue to adding a new frame
            }else
            {
                just add a new frame
            }
        */

        if(dataBuffer.size() > dataBufferSize)
        {
            dataBuffer.erase(dataBuffer.begin());
            dataBuffer.push_back(frame);
        }
        else
        {
            dataBuffer.push_back(frame);
        }
        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        string detectorType = argv[1];

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, false,&Average_Time);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, imgGray, false,&Average_Time);
        }
        else if (detectorType.compare("BRISK") == 0 || detectorType.compare("SIFT") == 0 ||
                   detectorType.compare("AKAZE") == 0 || detectorType.compare("ORB") == 0 ||
                   detectorType.compare("FAST") == 0)
        {
            detKeypointsModern(keypoints, imgGray, detectorType ,false,&Average_Time);
        }
        
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        std::vector<cv::KeyPoint> vehicle_Keypoints;
        if (bFocusOnVehicle)
        {
            // Remove keypoints outside of the vehicleRect
            for (auto it=keypoints.begin(); it != keypoints.end(); it++ ) {
              if (vehicleRect.contains(it->pt)) {
                //keypoints.erase(it);
                vehicle_Keypoints.push_back(*it);
              }
            }
        }

        //Neighborhood distribution calculation
        double average_size = 0;
        for (auto kp : vehicle_Keypoints) {
            average_size += kp.size;
        }
        average_size /= vehicle_Keypoints.size();
        total_keypoints.push_back(keypoints.size());
        total_vehicle_Keypoints.push_back(vehicle_Keypoints.size());
        average_size_detectedKP.push_back(average_size);
        
        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 20;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        string descriptorType = argv[2]; // BRIEF, ORB, FREAK, AKAZE, SIFT
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType,&Average_Time);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            string descriptorType = "DES_HOG"; // DES_BINARY, DES_HOG
            string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType, matcherType, selectorType);
            total_matched_keypoints.push_back(matches.size());

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                // cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    } // eof loop over all images
    cout << "DATA BUFFER ORIGINAL SIZE : "<<dataBuffer.size();
    

                        /* Run Task 7 script * Uncomment this part    */

    // std::ofstream Task7;
    // Task7.open("../TASK MP.7.txt", std::ios_base::app); // append instead of overwrite
    // Task7 <<"For Detector \""<<argv[1]<<"\""<<" number of keypoints on vehicle detected :\n";
    // for(int i = 0;i < total_vehicle_Keypoints.size();i++)
    // {
    //     Task7 << "\t *)In Image("<<i<<") , "<<total_keypoints[i]<<" Keypoints Were Detected , "\
    //     << total_vehicle_Keypoints[i] <<" Were On The Vehicle , Average size of neigbourhood : "<< average_size_detectedKP[i]<<".\n";
    // }



                        /* Run task 8 script & uncomment this part */

    // std::ofstream Task8;
    // Task8.open("../TASK MP.8.txt", std::ios_base::app); // append instead of overwrite
    // Task8 <<"For Detector \""<<argv[1]<<"\""<<" & Descriptor \""<<argv[2]<<"\" combination : \n";
    // for(int i = 0;i < total_matched_keypoints.size();i++)
    // {
    //     Task8 << "\t *)In Image("<<i<<") , number of matched keypoints : "<< total_matched_keypoints[i]<<"\n";
    // }



                        /* Run task 9 script & uncomment this part */
    // Average_Time *=100;
    // std::ofstream Task9;
    // Task9.open("../TASK MP.9.csv", std::ios_base::app);
    // Task9 << argv[1]<<"/" << argv[2]<<"," << Average_Time <<"\n";

    // cout <<"Value for Average_Time :"<<Average_Time;

    return 0;
}
