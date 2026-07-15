#include <utility>
#include <fstream>
#include <algorithm>
#include <iterator>

#include "plog/Log.h"

#include "crnn_net.h"
#include "utils.h"

#ifdef ENABLE_VULKAN
#include <gpu.h>
#endif

namespace OCR
{

CRNNNet::CRNNNet(const RecConfig &config)
{
    Initialize(config);
}

CRNNNet::~CRNNNet()
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

CRNNNet::CRNNNet(CRNNNet &&other) noexcept
    : config_(std::exchange(other.config_, {}))
    , net_(std::move(other.net_))
    , keys_(std::move(other.keys_))
#ifdef ENABLE_VULKAN
    , vk_device_(std::exchange(other.vk_device_, nullptr))
    , blob_allocator_(std::exchange(other.blob_allocator_, nullptr))
    , staging_allocator_(std::exchange(other.staging_allocator_, nullptr))
#endif
{

}

CRNNNet & CRNNNet::operator = (CRNNNet &&other) noexcept
{
    if (this != &other)
    {
        config_ = std::exchange(other.config_, {});
        net_ = std::move(other.net_);
        keys_ = std::move(other.keys_);
#ifdef ENABLE_VULKAN
        vk_device_ = std::exchange(other.vk_device_, nullptr);
        blob_allocator_ = std::exchange(other.blob_allocator_, nullptr);
        staging_allocator_ = std::exchange(other.staging_allocator_, nullptr);
#endif
    }
    return *this;
}

bool CRNNNet::Initialize(const RecConfig &config)
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

            PLOGW << "CRNNNet using Vulkan device " << device_index
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

    // load keys
    std::string line;
    std::ifstream ifs{config_.keys_path};
    if (!ifs.is_open())
    {
        PLOGE << "Failed to load keys " << config_.keys_path;
        return false;
    }
    keys_.clear();
    while (std::getline(ifs, line))
        keys_.emplace_back(line);
    PLOGD << "Total keys: " << keys_.size();

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

std::vector<TextLine> CRNNNet::Rec(const std::vector<cv::Mat> &text_images) const
{
    std::vector<TextLine> text_lines(text_images.size());

    int num_lines = static_cast<int>(text_images.size());
    #pragma omp parallel for num_threads(config_.reco_threads) schedule(dynamic)
    for (int i = 0; i < num_lines; ++i)
    {
        text_lines[i] = Rec(text_images[i]);
    }

    return text_lines;
}

TextLine CRNNNet::Rec(const cv::Mat &text_image) const
{
    // resize
    float ratio = static_cast<float>(target_h_) / text_image.rows;
    int rsz_w = static_cast<int>(text_image.cols * ratio);

    ncnn::Mat blob = ncnn::Mat::from_pixels_resize(text_image.data, ncnn::Mat::PIXEL_BGR,
        text_image.cols, text_image.rows, rsz_w, target_h_);
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

    // decode output and get TextLine
    float *arr = reinterpret_cast<float *>(out.data);
    std::vector<float> scores(arr, arr + out.h * out.w);

    return Score2TextLine(scores, out.h, out.w);
}

TextLine CRNNNet::Score2TextLine(const std::vector<float> &scores, const int rows, const int cols) const
{
    if (cols != static_cast<int>(keys_.size()))
    {
        PLOGE << "Unmatched scores: " << cols << " != " << keys_.size();
        return TextLine{};
    }

    std::string text;
    std::vector<float> text_scores;
    int prev_i = -1;
    const int blank_i = 0;

    for (int i = 0; i < rows; ++i)
    {
        auto max_it = std::max_element(scores.begin() + i * cols, scores.begin() + (i + 1) * cols);
        int max_i = static_cast<int>(std::distance(scores.begin(), max_it));
        max_i %= cols;
        float max_v = *max_it;

        if (max_i != blank_i && max_i != prev_i)
        {
            text.append(keys_[max_i]);
            text_scores.emplace_back(max_v);
        }
        prev_i = max_i;
    }

    Trim(text);

    return {std::move(text), std::move(text_scores)};
}

}   // namespace OCR