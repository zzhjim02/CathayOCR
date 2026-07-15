#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include "ocr_engine.h"
#include "common.h"

namespace py = pybind11;

// numpy数组 → cv::Mat
static cv::Mat numpy_to_mat(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> input)
{
    py::buffer_info buf = input.request();

    int cv_type = CV_8UC1;
    if (buf.ndim == 3)
    {
        int ch = static_cast<int>(buf.shape[2]);
        if (ch == 3)      cv_type = CV_8UC3;
        else if (ch == 4) cv_type = CV_8UC4;
    }

    cv::Mat mat(static_cast<int>(buf.shape[0]),
                static_cast<int>(buf.shape[1]),
                cv_type,
                static_cast<uchar *>(buf.ptr));

    return mat.clone();
}

// std::vector<OCR::OCRResult> → Python list of dict
static py::list convert_results(const std::vector<OCR::OCRResult> &results) // 这里加了 OCR::
{
    py::list py_results;
    for (const auto &r : results)
    {
        py::dict d;

        // 1. 识别文本
        d["text"] = r.line.text;

        // 2. 识别置信度
        d["scores"] = r.line.scores;

        // 3. 检测框的置信度
        d["det_score"] = r.box.score;

        // 4. 角度信息
        d["is_rotated"] = r.angle.is_rot;
        d["angle_score"] = r.angle.score;

        // 5. 文本框四个角点坐标
        py::list box;
        for (const cv::Point &pt : r.box.points) // 这里明确了 cv::Point
        {
            box.append(py::make_tuple(pt.x, pt.y));
        }
        d["box"] = box;

        py_results.append(d);
    }
    return py_results;
}

// pybind11 模块定义
PYBIND11_MODULE(ocr_engine, m)
{
    m.doc() = "PaddleOCR-ncnn Python Bindings";

    py::class_<OCR::OCREngine>(m, "OCREngine")

        .def(py::init<>(),
             "创建空的 OCREngine 实例")

        .def(py::init<const std::string &>(),
             py::arg("config_path"),
             "创建 OCREngine 并用 config.json 初始化")

        .def("initialize",
             &OCR::OCREngine::Initialize,
             py::arg("config_path"),
             "加载配置文件初始化引擎，返回 True 表示成功")

        .def("run",
             [](OCR::OCREngine &self,
                py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img)
             {
                 cv::Mat mat = numpy_to_mat(img);
                 auto results = self.Run(mat);
                 return convert_results(results);
             },
             py::arg("image"),
R"(运行 OCR 识别。
参数:
    image: numpy 数组，(H,W) 灰度图 或 (H,W,3) BGR 彩色图
返回:
    list[dict], 每个 dict 包含:
      - "text"       : str         识别出的文字
      - "scores"     : list[float] 每个字符的识别置信度
      - "det_score"  : float       文本框检测置信度
      - "is_rotated" : bool        是否被旋转过
      - "angle_score": float       旋转角度置信度
      - "box"        : list        文本框四个角点 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
)");
}
