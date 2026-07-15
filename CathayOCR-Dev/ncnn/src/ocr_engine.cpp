#include <utility>
#include <fstream>
#include <stdexcept>
#include <filesystem>

#include "json/json.hpp"
#include "plog/Log.h"

#include "utils.h"
#include "ocr_engine.h"

namespace
{

template <typename T>
T GetJValue(const nlohmann::json &json, const std::vector<std::string> &keys, const T &dft)
{
    try
    {
        const nlohmann::json *current = &json;
        for (const auto &key : keys)
        {
            if (current->contains(key))
            {
                current = &((*current)[key]);
            }
            else
            {
                PLOGW << "Failed to find key: " << key << ", use default value: " << dft;
                return dft;
            }
        }
        return current->get<T>();
    }
    catch (const std::exception &e)
    {
        PLOGW << "Failed to parsing json: " << e.what() << ", use default value: " << dft;
        return dft;
    }
}

}   // unnamed namespace

namespace OCR
{

OCREngine::OCREngine(const std::string &config_path)
{
    Initialize(config_path);
}

OCREngine::OCREngine(OCREngine &&other) noexcept
    : config_(std::exchange(other.config_, {}))
    , det_net_(std::move(other.det_net_))
    , cls_net_(std::move(other.cls_net_))
    , rec_net_(std::move(other.rec_net_))
    , run_mutex_(std::move(other.run_mutex_))
{

}

OCREngine & OCREngine::operator = (OCREngine &&other) noexcept
{
    if (this != &other)
    {
        config_ = std::exchange(other.config_, {});
        det_net_ = std::move(other.det_net_);
        cls_net_ = std::move(other.cls_net_);
        rec_net_ = std::move(other.rec_net_);
        run_mutex_ = std::move(other.run_mutex_);
    }
    return *this;
}

bool OCREngine::Initialize(const std::string &config_path)
{
    // load json
    nlohmann::json j{};
    try
    {
#ifdef _WIN32
        std::ifstream config_file(Utf8ToWString(config_path).c_str());
#else
        std::ifstream config_file(config_path);
#endif
        j = nlohmann::json::parse(config_file, nullptr, true, true);
    }
    catch(const nlohmann::json::exception &e)
    {
        PLOGE << "Failed to read JSON config from " << config_path << ": " << e.what();
        return false;
    }

    // read config
    config_.is_save = GetJValue(j, {"save"}, false);

    // Get the directory where config.json is located
    std::filesystem::path config_file_path = std::filesystem::absolute(Utf8ToPath(config_path));
    std::filesystem::path config_dir = config_file_path.parent_path();
    
    PLOGD << "Config file: " << config_file_path.string();
    PLOGD << "Config dir: " << config_dir.string();

    DetConfig &det_config = config_.det_config;
    det_config.infer_threads = GetThreads(GetJValue(j, {"det", "infer_threads"}, 1));
    
    // Support both nested and root-level model paths for UMI-OCR compatibility
    std::string det_model_path = GetJValue(j, {"det", "model_path"}, std::string());
    if (det_model_path.empty()) {
        det_model_path = GetJValue(j, {"det_model_path"}, std::string());
    }
    // If path is relative, make it absolute based on config file location
    if (!det_model_path.empty() && !Utf8ToPath(det_model_path).is_absolute()) {
        det_model_path = (config_dir / Utf8ToPath(det_model_path)).lexically_normal().string();
        PLOGD << "Det model path (relative->absolute): " << det_model_path;
    } else if (!det_model_path.empty()) {
        // Check if absolute path exists, if not try relative to config dir
        if (!std::filesystem::exists(Utf8ToPath(det_model_path))) {
            PLOGW << "Absolute path not found: " << det_model_path;
            std::string relative_path = Utf8ToPath(det_model_path).filename().string();
            std::string new_path = (config_dir / "models" / Utf8ToPath(relative_path)).lexically_normal().string();
            if (std::filesystem::exists(Utf8ToPath(new_path + ".param"))) {
                det_model_path = new_path;
                PLOGI << "Using relative path instead: " << det_model_path;
            }
        }
    }
    det_config.model_path = det_model_path;

    det_config.padding = GetJValue(j, {"det","padding"}, 50);
    det_config.max_side_len = GetJValue(j, {"det","max_side_len"}, 50);
    det_config.box_thres = GetJValue(j, {"det", "box_thres"}, 0.4f);
    det_config.bitmap_thres = GetJValue(j, {"det", "bitmap_thres"}, 0.3f);
    det_config.unclip_ratio = GetJValue(j, {"det", "unclip_ratio"}, 1.6f);
    det_config.is_fp16 = GetJValue(j, {"det", "fp16"}, false);
    det_config.use_vulkan = GetJValue(j, {"det", "use_vulkan"}, false);
    det_config.gpu_device_index = GetJValue(j, {"det", "gpu_device_index"}, -1);

    ClsConfig &cls_config = config_.cls_config;
    cls_config.infer_threads = GetThreads(GetJValue(j, {"cls", "infer_threads"}, 1));
    cls_config.reco_threads = GetThreads(GetJValue(j, {"cls", "reco_threads"}, 1));
    
    std::string cls_model_path = GetJValue(j, {"cls", "model_path"}, std::string());
    if (cls_model_path.empty()) {
        cls_model_path = GetJValue(j, {"cls_model_path"}, std::string());
    }
    // If path is relative, make it absolute based on config file location
    if (!cls_model_path.empty() && !Utf8ToPath(cls_model_path).is_absolute()) {
        cls_model_path = (config_dir / Utf8ToPath(cls_model_path)).lexically_normal().string();
        PLOGD << "Cls model path (relative->absolute): " << cls_model_path;
    } else if (!cls_model_path.empty()) {
        // Check if absolute path exists, if not try relative to config dir
        if (!std::filesystem::exists(Utf8ToPath(cls_model_path))) {
            PLOGW << "Absolute path not found: " << cls_model_path;
            std::string relative_path = Utf8ToPath(cls_model_path).filename().string();
            std::string new_path = (config_dir / "models" / Utf8ToPath(relative_path)).lexically_normal().string();
            if (std::filesystem::exists(Utf8ToPath(new_path + ".param"))) {
                cls_model_path = new_path;
                PLOGI << "Using relative path instead: " << cls_model_path;
            }
        }
    }
    cls_config.model_path = cls_model_path;

    cls_config.enable = GetJValue(j, {"cls", "enable"}, true);
    cls_config.most_angle = GetJValue(j, {"cls", "most_angle"}, true);
    cls_config.is_fp16 = GetJValue(j, {"cls", "fp16"}, false);
    cls_config.use_vulkan = GetJValue(j, {"cls", "use_vulkan"}, false);
    cls_config.gpu_device_index = GetJValue(j, {"cls", "gpu_device_index"}, -1);

    RecConfig &rec_config = config_.rec_config;
    rec_config.infer_threads = GetThreads(GetJValue(j, {"rec", "infer_threads"}, 1));
    rec_config.reco_threads = GetThreads(GetJValue(j, {"rec", "reco_threads"}, 1));
    
    std::string rec_model_path = GetJValue(j, {"rec", "model_path"}, std::string());
    if (rec_model_path.empty()) {
        rec_model_path = GetJValue(j, {"rec_model_path"}, std::string());
    }
    // If path is relative, make it absolute based on config file location
    if (!rec_model_path.empty() && !Utf8ToPath(rec_model_path).is_absolute()) {
        rec_model_path = (config_dir / Utf8ToPath(rec_model_path)).lexically_normal().string();
        PLOGD << "Rec model path (relative->absolute): " << rec_model_path;
    } else if (!rec_model_path.empty()) {
        // Check if absolute path exists, if not try relative to config dir
        if (!std::filesystem::exists(Utf8ToPath(rec_model_path))) {
            PLOGW << "Absolute path not found: " << rec_model_path;
            std::string relative_path = Utf8ToPath(rec_model_path).filename().string();
            std::string new_path = (config_dir / "models" / Utf8ToPath(relative_path)).lexically_normal().string();
            if (std::filesystem::exists(Utf8ToPath(new_path + ".param"))) {
                rec_model_path = new_path;
                PLOGI << "Using relative path instead: " << rec_model_path;
            }
        }
    }
    rec_config.model_path = rec_model_path;

    std::string keys_path = GetJValue(j, {"rec", "keys_path"}, std::string());
    if (keys_path.empty()) {
        keys_path = GetJValue(j, {"rec_char_dict_path"}, std::string());
    }
    // If path is relative, make it absolute based on config file location
    if (!keys_path.empty() && !Utf8ToPath(keys_path).is_absolute()) {
        keys_path = (config_dir / Utf8ToPath(keys_path)).lexically_normal().string();
        PLOGD << "Keys path (relative->absolute): " << keys_path;
    } else if (!keys_path.empty()) {
        // Check if absolute path exists, if not try relative to config dir
        if (!std::filesystem::exists(Utf8ToPath(keys_path))) {
            PLOGW << "Absolute path not found: " << keys_path;
            std::string relative_path = Utf8ToPath(keys_path).filename().string();
            std::string new_path = (config_dir / "models" / Utf8ToPath(relative_path)).lexically_normal().string();
            if (std::filesystem::exists(Utf8ToPath(new_path))) {
                keys_path = new_path;
                PLOGI << "Using relative path instead: " << keys_path;
            }
        }
    }
    rec_config.keys_path = keys_path;

    rec_config.is_fp16 = GetJValue(j, {"rec", "fp16"}, false);
    rec_config.use_vulkan = GetJValue(j, {"rec", "use_vulkan"}, false);
    rec_config.gpu_device_index = GetJValue(j, {"rec", "gpu_device_index"}, -1);

    // Vulkan ncnn::Net is not thread-safe for multiple concurrent extractors.
    // Disable internal OpenMP parallelism in cls/rec when any model uses Vulkan.
    if (det_config.use_vulkan || cls_config.use_vulkan || rec_config.use_vulkan)
    {
        cls_config.reco_threads = 1;
        rec_config.reco_threads = 1;
    }

    // initialize runtime lock
    run_mutex_ = std::make_unique<std::mutex>();

    // show configs
    ShowConfig();

    // create nets
    det_net_ = std::make_unique<DBNet>();
    cls_net_ = std::make_unique<AngleNet>();
    rec_net_ = std::make_unique<CRNNNet>();

    if (!det_net_->Initialize(det_config) ||
        !cls_net_->Initialize(cls_config) ||
        !rec_net_->Initialize(rec_config))
    {
        det_net_.reset();
        cls_net_.reset();
        rec_net_.reset();
        return false;
    }

    return true;
}

std::vector<OCRResult> OCREngine::Run(const cv::Mat &image) const
{
    if (!det_net_ || !cls_net_ || !rec_net_)
    {
        PLOGW << "Return an empty result since ( "
            << (!det_net_ ? "det_net " : "")
            << (!cls_net_ ? "cls_net " : "")
            << (!rec_net_ ? "rec_net " : "")
            << ") == nullptr";
        return {};
    }

    // ncnn Vulkan net/extractor is not thread-safe; serialize concurrent Run() calls.
    std::lock_guard<std::mutex> lock(*run_mutex_);

    // timers
    double det_time{}, cls_time{}, rec_time{}, total_time{};

    // 1. Text Detection
    total_time = det_time = static_cast<double>(cv::getTickCount());

    auto text_boxes = det_net_->Det(image);

    det_time = (cv::getTickCount() - det_time) / cv::getTickFrequency() * 1000.0;

    // rotate and crop images
    std::vector<cv::Mat> text_images(text_boxes.size());
    for (size_t i = 0; i < text_boxes.size(); ++i)
    {
        text_images[i] = GetRotatedCropImage(image, text_boxes[i].points);
    }

    // 2. Handle Angle
    cls_time = static_cast<double>(cv::getTickCount());

    auto angles = cls_net_->Cls(text_images);

    cls_time = (cv::getTickCount() - cls_time) / cv::getTickFrequency() * 1000.0;

    // rotate images
    for (size_t i = 0; i < text_images.size(); ++i)
    {
        if (angles[i].is_rot)
            cv::rotate(text_images[i], text_images[i], cv::ROTATE_180);
    }

    // 3. Recognize Text
    rec_time = static_cast<double>(cv::getTickCount());

    auto text_lines = rec_net_->Rec(text_images);

    rec_time = (cv::getTickCount() - rec_time) / cv::getTickFrequency() * 1000.0;

    std::vector<OCRResult> results(text_lines.size());
    for (size_t i = 0; i < text_lines.size(); ++i)
    {
        results[i].line = text_lines[i];
        results[i].angle = angles[i];
        results[i].box = text_boxes[i];
    }

    // timer
    total_time = (cv::getTickCount() - total_time) / cv::getTickFrequency() * 1000.0;
    PLOGI.printf("det_time(%.2fms), cls_time(%.2fms), rec_time(%.2fms), total(%.2fms)",
        det_time, cls_time, rec_time, total_time);

    // save results for debugging
    SaveResults(image, text_boxes, text_images, results);

    return results;
}

void OCREngine::ShowConfig() const
{
    const DetConfig &det_config = config_.det_config;
    const ClsConfig &cls_config = config_.cls_config;
    const RecConfig &rec_config = config_.rec_config;

    PLOGD << "--------------- Configs ---------------";

    PLOGD << "Det config";
    PLOGD.printf("  infer_threads(%d) padding(%d) max_side_len(%d) box_thres(%.2f) "
        "bitmap_thres(%.2f) unclip_ratio(%.2f) fp16(%d) use_vulkan(%d) gpu_device_index(%d)",
        det_config.infer_threads, det_config.padding, det_config.max_side_len,
        det_config.box_thres, det_config.bitmap_thres, det_config.unclip_ratio, det_config.is_fp16,
        det_config.use_vulkan, det_config.gpu_device_index);

    PLOGD << "Cls config";
    PLOGD.printf("  infer_threads(%d) reco_threads(%d) enable(%d) most_angle(%d) fp16(%d) use_vulkan(%d) gpu_device_index(%d)",
        cls_config.infer_threads, cls_config.reco_threads, cls_config.enable, cls_config.most_angle, cls_config.is_fp16,
        cls_config.use_vulkan, cls_config.gpu_device_index);

    PLOGD << "Rec config";
    PLOGD.printf("  infer_threads(%d) reco_threads(%d) fp16(%d) use_vulkan(%d) gpu_device_index(%d)",
        rec_config.infer_threads, rec_config.reco_threads, rec_config.is_fp16,
        rec_config.use_vulkan, rec_config.gpu_device_index);

    PLOGD << "---------------------------------------";
}

void OCREngine::SaveResults(const cv::Mat &image, std::vector<TextBox> &text_boxes,
    std::vector<cv::Mat> &text_images, std::vector<OCRResult> &results,
    const std::string folder_name) const
{
    if (config_.is_save)
    {
        // create results folder
        std::filesystem::create_directories(folder_name);

        // save det results
        cv::Mat det_image = image.clone();
        for (size_t i = 0; i < text_boxes.size(); ++i)
            cv::polylines(det_image, text_boxes[i].points, true, cv::Scalar(255.0, 0.0, 0.0), 2);
        cv::imwrite(folder_name + "/det.jpg", det_image);

        // print boxes
        for (size_t i = 0; i < results.size(); ++i)
        {
            auto &box = text_boxes[i];
            auto &angle = results[i].angle;
            PLOGD.printf("Box[%zu] (%d, %d) (%d, %d) (%d, %d) (%d, %d) score: %.2f | Rotate: %d, score: %.2f",
                i,
                box.points[0].x, box.points[0].y, box.points[1].x, box.points[1].y,
                box.points[2].x, box.points[2].y, box.points[3].x, box.points[3].y,
                box.score * 100.0f, angle.is_rot, angle.score * 100.0f
            );
        }

        // save rec inputs
        for (size_t i = 0; i < text_images.size(); ++i)
            cv::imwrite(folder_name + std::string("/text") + std::to_string(i) + std::string(".jpg"), text_images[i]);

        PLOGI << "Results saved to ./" << folder_name;
    }
}

}   // namespace OCR