#include <utility>
#include <algorithm>

#include "plog/Log.h"

#include "utils.h"
#include "db_net.h"

#ifdef ENABLE_VULKAN
#include <gpu.h>
#endif

namespace OCR
{

DBNet::DBNet(const DetConfig &config)
{
    Initialize(config);
}

DBNet::~DBNet()
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

DBNet::DBNet(DBNet &&other) noexcept
    : config_(std::exchange(other.config_, {}))
    , net_(std::move(other.net_))
#ifdef ENABLE_VULKAN
    , vk_device_(std::exchange(other.vk_device_, nullptr))
    , blob_allocator_(std::exchange(other.blob_allocator_, nullptr))
    , staging_allocator_(std::exchange(other.staging_allocator_, nullptr))
#endif
{

}

DBNet & DBNet::operator = (DBNet &&other) noexcept
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

bool DBNet::Initialize(const DetConfig &config)
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

            PLOGW << "DBNet using Vulkan device " << device_index
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

std::vector<TextBox> DBNet::Det(const cv::Mat &image) const
{
    // padding
    const int padding = config_.padding;
    cv::Mat pad_image;
    cv::copyMakeBorder(image, pad_image, padding, padding, padding, padding,
        cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, cv::Scalar(255.0, 255.0, 255.0));

    // resize
    const int target_size = std::min(config_.max_side_len + 2 * padding,
        std::max(pad_image.rows, pad_image.cols));

    int img_rows = pad_image.rows, img_cols = pad_image.cols;
    float ratio = static_cast<float>(target_size) / std::max(img_rows, img_cols);
    int rsz_rows = std::max(static_cast<int>(img_rows * ratio) / target_stride_ * target_stride_, target_stride_);
    int rsz_cols = std::max(static_cast<int>(img_cols * ratio) / target_stride_ * target_stride_, target_stride_);
    float ratio_rows = static_cast<float>(rsz_rows) / img_rows;
    float ratio_cols = static_cast<float>(rsz_cols) / img_cols;

    PLOGD.printf("src_w(%d), src_h(%d), dst_w(%d), dst_h(%d), ratio_w(%f), ratio_h(%f)",
        img_cols, img_rows, rsz_cols, rsz_rows, ratio_cols, ratio_rows);

    ncnn::Mat blob = ncnn::Mat::from_pixels_resize(
        pad_image.data, ncnn::Mat::PIXEL_BGR, img_cols, img_rows, rsz_cols, rsz_rows);
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

    // binarization
    const float denorm_values[1] = {255.0f};
    out.substract_mean_normalize(0, denorm_values);

    cv::Mat pred(out.h, out.w, CV_8UC1);
    out.to_pixels(pred.data, ncnn::Mat::PIXEL_GRAY);
    cv::Mat bitmap = pred > static_cast<uint8_t>(config_.bitmap_thres * 255.0f);

    // get boxes from bitmap
    auto text_boxes = FindBoxesFromBitmap(pred, bitmap, img_rows, img_cols, ratio_rows, ratio_cols);

    return text_boxes;
}

std::vector<TextBox> DBNet::FindBoxesFromBitmap(const cv::Mat &pred, const cv::Mat &bitmap,
    const int img_rows, const int img_cols, const float ratio_rows, const float ratio_cols) const
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bitmap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    size_t num_contours = std::min(contours.size(), max_candidates_);

    std::vector<TextBox> text_boxes;
    for (size_t i = 0; i < num_contours; ++i)
    {
        if (contours[i].size() <= 2)
            continue;

        cv::RotatedRect min_area_rect = cv::minAreaRect(contours[i]);

        int long_side;
        auto min_boxes = GetMinBoxes(min_area_rect, long_side);
        if (long_side < min_size_)
            continue;

        float box_score = BoxScoreFast(min_boxes, pred);
        if (box_score < config_.box_thres)
            continue;

        // unclip
        cv::RotatedRect unclip_rect = Unclip(min_boxes, config_.unclip_ratio);
        if (unclip_rect.size.height <= 1.0f || unclip_rect.size.width <= 1.0f)
            continue;

        min_boxes = GetMinBoxes(unclip_rect, long_side);
        if (long_side < min_size_ + 2)
            continue;

        std::vector<cv::Point> text_points;
        for (size_t j = 0; j < min_boxes.size(); ++j)
        {
            int x = std::clamp(static_cast<int>(min_boxes[j].x / ratio_cols) - config_.padding,
                0, img_cols - 2 * config_.padding - 1);
            int y = std::clamp(static_cast<int>(min_boxes[j].y / ratio_rows) - config_.padding,
                0, img_rows - 2 * config_.padding - 1);
            text_points.emplace_back(cv::Point{x, y});
        }
        text_boxes.emplace_back(TextBox{std::move(text_points), box_score});
    }
    std::reverse(text_boxes.begin(), text_boxes.end());

    return text_boxes;
}

}   // namespace OCR