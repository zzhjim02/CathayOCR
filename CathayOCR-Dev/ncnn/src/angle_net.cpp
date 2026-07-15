#include <utility>
#include <algorithm>

#include "plog/Log.h"

#include "angle_net.h"
#include "utils.h"

#ifdef ENABLE_VULKAN
#include <gpu.h>
#endif

namespace OCR
{

AngleNet::AngleNet(const ClsConfig &config)
{
    Initialize(config);
}

AngleNet::~AngleNet()
{
#ifdef ENABLE_VULKAN
    // Release dedicated allocators before destroying the net so that
    // ncnn can tear down the Vulkan device cleanly.
    if (vk_device_)
    {
        if (blob_allocator_)
        {
            vk_device_->reclaim_blob_allocator(blob_allocator_);
            blob_allocator_ = nullptr;
        }
        if (staging_allocator_)
        {
            vk_device_->reclaim_staging_allocator(staging_allocator_);
            staging_allocator_ = nullptr;
        }
        vk_device_ = nullptr;
    }
#endif

    // Destroy the net after all Vulkan resources have been released.
    net_.reset();
}

AngleNet::AngleNet(AngleNet &&other) noexcept
    : config_(std::exchange(other.config_, {}))
    , net_(std::move(other.net_))
#ifdef ENABLE_VULKAN
    , vk_device_(std::exchange(other.vk_device_, nullptr))
    , blob_allocator_(std::exchange(other.blob_allocator_, nullptr))
    , staging_allocator_(std::exchange(other.staging_allocator_, nullptr))
#endif
{

}

AngleNet & AngleNet::operator = (AngleNet &&other) noexcept
{
    if (this != &other)
    {
        config_ = std::exchange(other.config_, {});
        net_ = std::move(other.net_);
#ifdef ENABLE_VULKAN
        vk_device_ = std::exchange(other.vk_device_, nullptr);
        blob_allocator_ = std::exchange(other.blob_allocator_, nullptr);
        staging_allocator_ = std::exchange(other.staging_allocator_, nullptr);
#endif
    }
    return *this;
}

bool AngleNet::Initialize(const ClsConfig &config)
{
    config_ = config;

    net_ = std::make_unique<ncnn::Net>();
    net_->opt.num_threads = config_.infer_threads;
    net_->opt.use_fp16_packed = config_.is_fp16;
    net_->opt.use_fp16_storage = config_.is_fp16;
    net_->opt.use_fp16_arithmetic = config_.is_fp16;

#ifdef ENABLE_VULKAN
    if (config_.use_vulkan)
    {
        ncnn::create_gpu_instance();

        int gpu_count = ncnn::get_gpu_count();
        if (gpu_count <= 0)
        {
            PLOGE << "Vulkan enabled but no GPU found";
            net_.reset();
            return false;
        }

        net_->opt.use_vulkan_compute = true;

        int device_index = config_.gpu_device_index;
        if (device_index < 0 || device_index >= gpu_count)
        {
            device_index = GetPreferredGpuDevice();
        }
        if (device_index < 0 || device_index >= gpu_count)
        {
            device_index = 0;
        }

        net_->set_vulkan_device(device_index);

        // Acquire dedicated Vulkan allocators for this net to avoid
        // sharing default allocators across nets/extractors.
        vk_device_ = net_->vulkan_device();
        if (vk_device_)
        {
            blob_allocator_ = vk_device_->acquire_blob_allocator();
            staging_allocator_ = vk_device_->acquire_staging_allocator();
            net_->opt.blob_vkallocator = blob_allocator_;
            net_->opt.workspace_vkallocator = blob_allocator_;
            net_->opt.staging_vkallocator = staging_allocator_;

            PLOGW << "AngleNet using Vulkan device " << device_index
                  << ": " << ncnn::get_gpu_info(device_index).device_name();
        }
        else
        {
            PLOGE << "Failed to acquire Vulkan device " << device_index;
            net_.reset();
            return false;
        }
    }
#endif

    if (net_->load_param((config_.model_path + ".param").c_str()) ||
        net_->load_model((config_.model_path + ".bin").c_str()))
    {
        PLOGE << "Failed to load model from path: " << config_.model_path;
        PLOGE << "Tried loading: " << (config_.model_path + ".param") << " and " << (config_.model_path + ".bin");
        net_.reset();
        return false;
    }

    return true;
}

std::vector<Angle> AngleNet::Cls(const std::vector<cv::Mat> &text_images) const
{
    std::vector<Angle> angles(text_images.size());
    if (!config_.enable || text_images.empty())
    {
        for (auto &angle : angles)
            angle = Angle{false, 0.0f};
        return angles;
    }

    // get angles
    int num_images = static_cast<int>(text_images.size());
    #pragma omp parallel for num_threads(config_.reco_threads) schedule(static)
    for (int i = 0; i < num_images; ++i)
    {
        angles[i] = Cls(text_images[i]);
    }

    // vote for rotation decisions
    if (config_.most_angle)
    {
        float rot_weight = 0.0f;
        float no_rot_weight = 0.0f;
        for (const auto &angle : angles)
        {
            if (angle.is_rot)
                rot_weight += angle.score;
            else
                no_rot_weight += angle.score;
        }
        bool decision = rot_weight > no_rot_weight;
        for (auto &angle : angles)
            angle.is_rot = decision;
    }

    return angles;
}

Angle AngleNet::Cls(const cv::Mat &image) const
{
    // resize image
    cv::Mat rsz_image = SmartResize(image, 3.0f);

    ncnn::Mat blob = ncnn::Mat::from_pixels(rsz_image.data, ncnn::Mat::PIXEL_BGR, rsz_image.cols, rsz_image.rows);
    blob.substract_mean_normalize(mean_values_, norm_values_);

    // inference
    ncnn::Extractor ex = net_->create_extractor();
#ifdef ENABLE_VULKAN
    if (config_.use_vulkan)
    {
        ex.set_blob_vkallocator(net_->opt.blob_vkallocator);
        ex.set_workspace_vkallocator(net_->opt.workspace_vkallocator);
        ex.set_staging_vkallocator(net_->opt.staging_vkallocator);
    }
#endif
    ex.input("input", blob);
    ncnn::Mat out;
    ex.extract("output", out);
    ex.clear();

    // score to angle
    float *arr = reinterpret_cast<float *>(out.data);
    std::vector<float> scores(arr, arr + out.w);

    auto max_it = std::max_element(scores.begin(), scores.end());
    int max_i = static_cast<int>(std::distance(scores.begin(), max_it));
    float max_score = *max_it;

    return {max_i == 1, max_score};
}

cv::Mat AngleNet::SmartResize(const cv::Mat &image, const float max_downscale) const
{
    float ratio = static_cast<float>(target_h_) / image.rows;
    int rsz_w = static_cast<int>(image.cols * ratio);
    cv::Mat rsz_image;

    if (rsz_w < target_w_)
    {
        // resize then padding with gray pixels
        cv::resize(image, rsz_image, cv::Size(rsz_w, target_h_));
        cv::copyMakeBorder(rsz_image, rsz_image, 0, 0, 0, target_w_ - rsz_w,
            cv::BORDER_CONSTANT, cv::Scalar(114.0, 114.0, 114.0));
    }
    else if (rsz_w < target_w_ * max_downscale)
    {
        // resize with width compression
        cv::resize(image, rsz_image, cv::Size(target_w_, target_h_));
    }
    else
    {
        // resize with width compression and crop
        cv::Mat crop_image = image(cv::Rect(0, 0, static_cast<int>(max_downscale * target_w_ / ratio), image.rows));
        cv::resize(crop_image, rsz_image, cv::Size(target_w_, target_h_));
    }

    return rsz_image;
}

}   // namespace OCR