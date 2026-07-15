#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <filesystem>

#ifdef ENABLE_VULKAN
#include <gpu.h>
#endif

namespace OCR
{

int GetThreads(const int threads);

int GetPreferredGpuDevice();

#ifdef _WIN32
std::wstring Utf8ToWString(const std::string &utf8);
std::string WStringToUtf8(const std::wstring &wstr);
#endif

std::filesystem::path Utf8ToPath(const std::string &utf8);

std::vector<cv::Point2f> GetMinBoxes(const cv::RotatedRect &rrect, int &max_side_len);

float BoxScoreFast(const std::vector<cv::Point2f> &boxes, const cv::Mat &binary);

float GetUnclipDistance(const std::vector<cv::Point2f> &boxes, const float unclip_ratio);

cv::RotatedRect Unclip(const std::vector<cv::Point2f> &boxes, const float unclip_ratio);

cv::Mat GetRotatedCropImage(const cv::Mat &image, std::vector<cv::Point> points);

void Trim(std::string &s);

}   // namespace OCR

#endif // UTILS_H
