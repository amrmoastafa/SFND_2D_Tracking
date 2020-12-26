#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        // TODO : implement k-nearest-neighbor matching
        vector<vector<cv::DMatch>> knn_matches;
        
        matcher->knnMatch(descSource, descRef, knn_matches, 2); // finds the 2 best matches
        
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {

            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }

    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType,double *Average_Time)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
        double t = (double)cv::getTickCount();
        extractor = cv::BRISK::create(threshold, octaves, patternScale);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        *Average_Time += t;
    }
    else if (descriptorType.compare("BRIEF") == 0) {
        double t = (double)cv::getTickCount();
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        *Average_Time += t;
    }
    else if (descriptorType.compare("ORB") == 0) {
        double t = (double)cv::getTickCount();
        extractor = cv::ORB::create();
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        *Average_Time += t;
    }
    else if (descriptorType.compare("FREAK") == 0) {
        double t = (double)cv::getTickCount();
        extractor = cv::xfeatures2d::FREAK::create();
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        *Average_Time += t;
    }
    else if (descriptorType.compare("AKAZE") == 0) {
        double t = (double)cv::getTickCount();
        extractor = cv::AKAZE::create();
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        *Average_Time += t;
    }
    else if (descriptorType.compare("SIFT") == 0) {
        double t = (double)cv::getTickCount();
        extractor = cv::SIFT::create();
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        *Average_Time += t;
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis,double *Average_Time)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    *Average_Time += t;
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}


//// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
//// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis,double *Average_Time)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)
    double t = (double)cv::getTickCount();
    // Normalize the output & detecting Harris Corners
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
    double Overlap_Threshold = 0.0; // Maximum allowed Overlapping between two features
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            
            if (response > minResponse)
            { // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    // The following function draws a circle around the keypoint , if the new keypoint circle coincides with another circle , they are the same
                    // Keypoint , then it return 1 , if they are a totally different circles , we get 0 , the overlap is somewhere between 0 ~ 1

                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    // cout<<kptOverlap<<"#*#*#*#*#*#*#*#*#*#*"<<endl;

                    // If the calculated overlap value is beyond certain value which is 0 here , we remove the point.
                    if (kptOverlap > Overlap_Threshold)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      // if overlap is >threshold AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }     // eof loop over rows
    
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    *Average_Time += t;
    cout << "Harris Detection with n=" << keypoints.size()<<" Keypoints." << endl;
            
    // visualize keypoints
    if (bVis) {
        string windowName = "Harris Corner Detection Results";
        cv::namedWindow(windowName);
        cv::Mat visImage = dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}


void detKeypointsModern(vector<cv::KeyPoint> &keypoints, cv::Mat &img,string detectorType, bool bVis,double *Average_Time)
{
    if (detectorType.compare("FAST") == 0)
    {
        int threshold = 10;
        bool nonmaxSuppression = true;
        cv::Ptr<cv::FeatureDetector> fast_detector = cv::FastFeatureDetector::create(threshold, nonmaxSuppression, cv::FastFeatureDetector::TYPE_9_16 );
        double t = (double)cv::getTickCount();
        fast_detector->detect(img,keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        *Average_Time += t;
        cout << "Fast Detection with n = " << keypoints.size() <<" Keypoints."<< endl;
    }
    else if (detectorType.compare("BRISK") == 0){
        cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        *Average_Time += t;
        cout << "BRISK Detection with n = " << keypoints.size() <<" Keypoints."<< endl;
    }
    else if (detectorType.compare("ORB") == 0) {
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        *Average_Time += t;
        cout << "ORB Detection with n = " << keypoints.size() <<" Keypoints."<< endl;
    }
    else if (detectorType.compare("AKAZE") == 0) {
        cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        *Average_Time += t;
        cout << "AKAZE Detection with n = " << keypoints.size() <<" Keypoints."<< endl;
    }
    else if (detectorType.compare("SIFT") == 0) {
        cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        *Average_Time += t;
        cout << "SIFT Detection with n = " << keypoints.size() <<" Keypoints."<< endl;
    }




    // Visualization part is common so place it outside the if conditions
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}