
#include <iostream>
#include <vector>
#include <algorithm>

#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <brisk/brisk.h>


// parameters
// the following parameters allow us to start with small-scope project
#define DOWNSAMPLE_RATE    10
#define TIME_WINDOW_BEGIN  "1403636649313555456"
#define TIME_WINDOW_END    "1403636658963555584"

#define BRISK_DETECTION_THRESHOLD             50.0
#define BRISK_DETECTION_OCTAVES               0
#define BRISK_DETECTION_ABSOLUTE_THRESHOLD    800.0
#define BRISK_DETECTION_MAX_KEYPOINTS         450

#define BRISK_DESCRIPTION_ROTATION_INVARIANCE true
#define BRISK_DESCRIPTION_SCALE_INVARIANCE    false



class CameraData {
 public:
  CameraData(std::string timeStampStr, std::string dataFilePath) {
    time_ = timeStampStr;
    image_ = cv::imread(dataFilePath,cv::IMREAD_GRAYSCALE);
  }

  std::string getTime() {
    return time_;
  }

  cv::Mat getImage() {
    return image_;
  }

 private:
  std::string time_;   // we don't have to process time at this moment
  cv::Mat image_;
};




int main(int argc, char **argv) {

  /*** Step 1. Reading image files ***/

  // the folder path
  // std::string path(argv[1]);
  std::string path("../../../dataset/mav0/");
  std::string camera_data_folder("cam0/data/");

  std::vector<std::string> image_names;

  // boost allows us to work on image files directly
  for (auto iter = boost::filesystem::directory_iterator(path + camera_data_folder);
        iter != boost::filesystem::directory_iterator(); iter++) {

    if (!boost::filesystem::is_directory(iter->path())) {          //we eliminate directories
      image_names.push_back(iter->path().filename().string());
    } 
    else
      continue;
  }

  std::sort(image_names.begin(), image_names.end());


  int downsample_rate = DOWNSAMPLE_RATE;
  std::string time_window_begin = TIME_WINDOW_BEGIN;
  std::string time_window_end = TIME_WINDOW_END;

  std::vector<CameraData> camera_observation_data;   // image and timestep

  int counter = 0;
  for (auto& image_names_iter: image_names) {	
  
    if (counter % downsample_rate == 0) {            // downsample images for testing

    	std::string time_stamp_str = image_names_iter.substr(0,19);  // remove ".png"

    	if(time_window_begin <= time_stamp_str && time_stamp_str <= time_window_end) {
    	  std::string dataFilePath = path + camera_data_folder + image_names_iter;

        std::string time_stamp_str = image_names_iter.substr(0,19);
        camera_observation_data.push_back(CameraData(time_stamp_str, dataFilePath));

        // cv::imshow(time_stamp_str, camera_observation_data.back().getImage());
        // cv::waitKey(100);
      }
    }

    counter++;

  }


  /*** Step 2. Extracting features ***/

  std::shared_ptr<cv::FeatureDetector> brisk_detector = 
    std::shared_ptr<cv::FeatureDetector>(
      new brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>(
        BRISK_DETECTION_THRESHOLD, 
        BRISK_DETECTION_OCTAVES, 
        BRISK_DETECTION_ABSOLUTE_THRESHOLD, 
        BRISK_DETECTION_MAX_KEYPOINTS));

  // you can try to use ORB feature as well
  // std::shared_ptr<cv::FeatureDetector> orb_detector = cv::ORB::create();

  std::vector<cv::KeyPoint> keypoints;

  for (int i=0; i<camera_observation_data.size(); i++) {	

  	keypoints.clear();
    brisk_detector->detect(camera_observation_data.at(i).getImage(), keypoints);
    // orb_detector->detect(camera_observation_data.at(i).getImage(), keypoints);

    cv::Mat img_w_keypoints;
    cv::drawKeypoints(camera_observation_data.at(i).getImage(), keypoints, img_w_keypoints);
    cv::imshow(camera_observation_data.at(i).getTime(), img_w_keypoints);
    cv::waitKey(100);

  }


  /*** Step 3. Matching images ***/

  // descriptor
  std::vector<cv::KeyPoint> keypoints_1;
  std::vector<cv::KeyPoint> keypoints_2;
  keypoints_1.clear();
  keypoints_2.clear();

  brisk_detector->detect(camera_observation_data.at(1).getImage(), keypoints_1);
  brisk_detector->detect(camera_observation_data.at(2).getImage(), keypoints_2);


  cv::Mat descriptors_1;         ///< we store the descriptors using OpenCV's matrices
  cv::Mat descriptors_2;         ///< we store the descriptors using OpenCV's matrices
  descriptors_1.resize(0);
  descriptors_2.resize(0);

  std::shared_ptr<cv::FeatureDetector> brisk_extractor = 
    std::shared_ptr<cv::DescriptorExtractor>(
      new brisk::BriskDescriptorExtractor(
        BRISK_DESCRIPTION_ROTATION_INVARIANCE,
        BRISK_DESCRIPTION_SCALE_INVARIANCE));

  brisk_extractor->compute(camera_observation_data.at(1).getImage(), keypoints_1, descriptors_1);
  brisk_extractor->compute(camera_observation_data.at(2).getImage(), keypoints_2, descriptors_2);

  std::shared_ptr<cv::DescriptorMatcher> matcher = 
    std::shared_ptr<cv::DescriptorMatcher>(
      cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE));
  std::vector<cv::DMatch> matches;

  matcher->match(descriptors_1, descriptors_2, matches);

  cv::Mat img_w_matches;
  cv::drawMatches(camera_observation_data.at(1).getImage(), keypoints_1,
                  camera_observation_data.at(2).getImage(), keypoints_2,
                  matches, img_w_matches);
  cv::imshow("Matches", img_w_matches);
  cv::waitKey(100);


  /*** Step 4. Testing RANSAC ***/

  // try RANSAC
  std::vector<cv::Point2f> scene_1;
  std::vector<cv::Point2f> scene_2;
  std::vector<cv::DMatch> good_matches;
  for (int i=0; i<matches.size(); i++) {
    if (matches[i].distance < 400) {
      good_matches.push_back(matches[i]);

      scene_1.push_back(keypoints_1[matches[i].queryIdx].pt);
      scene_2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }
  }

  cv::drawMatches(camera_observation_data.at(1).getImage(), keypoints_1,
                  camera_observation_data.at(2).getImage(), keypoints_2,
                  good_matches, img_w_matches);
  cv::imshow("Matches with only small distance", img_w_matches);
  cv::waitKey();

  cv::Mat H = cv::findHomography(scene_1, scene_2, cv::RANSAC);

  std::cout << H;

  std::getchar();
  return 0;
}
