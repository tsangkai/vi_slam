
#include <iostream>
#include <vector>
#include <algorithm>

#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <brisk/brisk.h>


// parameters
// the following parameters allow us to start with small-scope project
#define DOWNSAMPLE_RATE    10
#define TIME_WINDOW_BEGIN  "1403636649313555456"
#define TIME_WINDOW_END    "1403636658963555584"

#define BRISK_DETECTION_THRESHOLD           50.0
#define BRISK_DETECTION_OCTAVES             0
#define BRISK_DETECTION_ABSOLUTE_THRESHOLD  800.0
#define BRISK_DETECTION_MAX_KEYPOINTS       450



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

  // the folder path
  // std::string path(argv[1]);
  std::string path("../../../dataset/mav0/");
  std::string camera_data_folder("cam0/data/");

  std::vector<std::string> image_names;

  for (auto iter = boost::filesystem::directory_iterator(path + camera_data_folder);
        iter != boost::filesystem::directory_iterator(); iter++) {
    if (!boost::filesystem::is_directory(iter->path())) {          //we eliminate directories
      image_names.push_back(iter->path().filename().string());
    } 
    else {
      continue;
    }
  }

  std::sort(image_names.begin(), image_names.end());

  std::vector<CameraData> camera_observation_data;


  int downsample_rate = DOWNSAMPLE_RATE;
  std::string time_window_begin = TIME_WINDOW_BEGIN;
  std::string time_window_end = TIME_WINDOW_END;


  int counter = 0;
  for (auto& image_names_iter: image_names) {	
  
    if (counter % downsample_rate == 0) {            // enable downsample of the images for testing

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


  // feature extraction 
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




  std::getchar();
  return 0;
}
