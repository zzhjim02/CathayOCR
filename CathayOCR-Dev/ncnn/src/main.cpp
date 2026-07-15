#include "logging.h"
#include "ocr_engine.h"
#include "utils.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <algorithm>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <shellapi.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#endif

#include <opencv2/opencv.hpp>
#include "3rdparty/json/json.hpp"

using json = nlohmann::json;

enum class RunMode
{
    CLI,
    PIPE,
    TCP
};

struct AppConfig
{
    RunMode mode = RunMode::CLI;
    std::string config_path;
    std::string model_path;
    int tcp_port = 18043;
    bool verbose = false;
    bool use_vulkan = false;
};

AppConfig g_config;
std::unique_ptr<OCR::OCREngine> g_ocr_engine;
std::mutex g_engine_mutex;
std::atomic<bool> g_running{true};

#ifdef _WIN32
BOOL WINAPI ConsoleCtrlHandler(DWORD dwCtrlType)
{
    switch (dwCtrlType)
    {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_LOGOFF_EVENT:
    case CTRL_SHUTDOWN_EVENT:
        g_running = false;
        return TRUE;
    default:
        return FALSE;
    }
}
#else
#include <csignal>
void signal_handler(int)
{
    g_running = false;
}
#endif

namespace
{

std::string base64_decode(const std::string &input)
{
    static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string output;
    output.reserve(input.size() * 3 / 4);

    std::vector<int> T(256, -1);
    for (int i = 0; i < 64; i++)
        T[table[i]] = i;

    int val = 0, bits = 0;
    for (unsigned char c : input)
    {
        if (T[c] == -1 && c != '=')
            continue;
        val = (val << 6) | T[c];
        bits += 6;
        if (bits >= 8)
        {
            output.push_back(static_cast<char>((val >> (bits - 8)) & 0xFF));
            bits -= 8;
        }
    }
    return output;
}

bool load_config(const std::string &config_path, json &config)
{
#ifdef _WIN32
    std::ifstream file(OCR::Utf8ToWString(config_path).c_str());
#else
    std::ifstream file(config_path);
#endif
    if (!file.is_open())
    {
        std::cerr << "Failed to open config file: " << config_path << std::endl;
        return false;
    }

    try
    {
        file >> config;
        return true;
    }
    catch (const json::parse_error &e)
    {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat load_image_from_file(const std::string &path)
{
#ifdef _WIN32
    std::ifstream file(OCR::Utf8ToWString(path).c_str(), std::ios::binary);
#else
    std::ifstream file(path, std::ios::binary);
#endif
    if (!file)
    {
        std::cerr << "Failed to open image file: " << path << std::endl;
        return cv::Mat();
    }

    std::vector<uchar> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (data.empty())
    {
        std::cerr << "Image file is empty: " << path << std::endl;
        return cv::Mat();
    }

    return cv::imdecode(data, cv::IMREAD_COLOR);
}

cv::Mat load_image_from_json(const json &input)
{
    cv::Mat image;

    if (input.contains("img_base64"))
    {
        std::string base64_str = input["img_base64"];
        std::string decoded = base64_decode(base64_str);
        std::vector<uchar> data(decoded.begin(), decoded.end());
        image = cv::imdecode(data, cv::IMREAD_COLOR);
    }
    else if (input.contains("img_bytes"))
    {
        if (input["img_bytes"].is_array())
        {
            std::vector<uchar> data = input["img_bytes"].get<std::vector<uchar>>();
            image = cv::imdecode(data, cv::IMREAD_COLOR);
        }
    }
    else if (input.contains("img_path"))
    {
        std::string path = input["img_path"];
        image = load_image_from_file(path);
    }

    return image;
}

json create_success_response(const std::vector<OCR::OCRResult> &results)
{
    json response;
    response["code"] = 200;
    response["data"] = json::array();

    for (const auto &result : results)
    {
        json item;
        item["score"] = result.line.scores.empty() ? 0.0f : result.line.scores[0];
        item["text"] = result.line.text;

        json box = json::array();
        for (const auto &pt : result.box.points)
        {
            box.push_back({pt.x, pt.y});
        }
        item["box"] = box;

        response["data"].push_back(item);
    }

    if (results.empty())
    {
        response["code"] = 300;
        response["data"] = json::array();
    }

    return response;
}

json create_error_response(int code, const std::string &error_msg)
{
    json response;
    response["code"] = code;
    response["data"] = json::array();
    response["error"] = error_msg;
    return response;
}

json create_init_response(bool success, const std::string &message)
{
    json response;
    response["code"] = success ? 200 : 400;
    response["data"] = json::array();
    response["message"] = message;
    return response;
}

bool initialize_engine()
{
    std::lock_guard<std::mutex> lock(g_engine_mutex);

    if (g_ocr_engine)
    {
        g_ocr_engine.reset();
    }

    if (g_config.config_path.empty())
    {
        std::cerr << "Config path not specified" << std::endl;
        return false;
    }

    g_ocr_engine = std::make_unique<OCR::OCREngine>(g_config.config_path);
    if (!g_ocr_engine)
    {
        return false;
    }

    return true;
}

std::string process_request(const json &input)
{
    if (!g_ocr_engine)
    {
        json resp = create_error_response(400, "Engine not initialized");
        return resp.dump();
    }

    cv::Mat image = load_image_from_json(input);

    if (image.empty())
    {
        json resp = create_error_response(400, "Failed to load image");
        return resp.dump();
    }

    try
    {
        std::vector<OCR::OCRResult> results = g_ocr_engine->Run(image);
        json resp = create_success_response(results);
        return resp.dump();
    }
    catch (const std::exception &e)
    {
        json resp = create_error_response(400, std::string("OCR processing failed: ") + e.what());
        return resp.dump();
    }
}

bool set_nonblocking(
#ifdef _WIN32
    SOCKET fd
#else
    int fd
#endif
)
{
#ifdef _WIN32
    u_long mode = 1;
    return ioctlsocket(fd, FIONBIO, &mode) == 0;
#else
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags == -1)
        return false;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
#endif
}

}

void print_usage(const char *program_name)
{
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  -c, --config <path>     Path to config.json (required)\n"
              << "  -m, --mode <mode>       Run mode: cli, pipe, tcp (default: cli)\n"
              << "  -p, --port <port>       TCP port for tcp mode (default: 18043)\n"
              << "  -v, --verbose           Enable verbose output\n"
              << "  --vulkan                Use Vulkan GPU acceleration\n"
              << "  -h, --help              Show this help message\n\n"
              << "Run modes:\n"
              << "  cli   - Command line: " << program_name << " -c config.json image.jpg\n"
              << "  pipe  - Pipe mode:   echo '{\"img_path\":\"test.jpg\"}' | " << program_name << " -m pipe -c config.json\n"
              << "  tcp   - TCP mode:     " << program_name << " -m tcp -c config.json -p 18043\n";
}

bool parse_arguments(int argc, char *argv[])
{
    g_config.mode = RunMode::CLI;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "-c" || arg == "--config")
        {
            if (i + 1 < argc)
                g_config.config_path = argv[++i];
        }
        else if (arg == "-m" || arg == "--mode")
        {
            if (i + 1 < argc)
            {
                std::string mode = argv[++i];
                if (mode == "cli")
                    g_config.mode = RunMode::CLI;
                else if (mode == "pipe")
                    g_config.mode = RunMode::PIPE;
                else if (mode == "tcp")
                    g_config.mode = RunMode::TCP;
            }
        }
        else if (arg == "-p" || arg == "--port")
        {
            if (i + 1 < argc)
                g_config.tcp_port = std::stoi(argv[++i]);
        }
        else if (arg == "-v" || arg == "--verbose")
        {
            g_config.verbose = true;
        }
        else if (arg == "--vulkan")
        {
            g_config.use_vulkan = true;
        }
        else if (arg == "-h" || arg == "--help")
        {
            print_usage(argv[0]);
            return false;
        }
        else if (arg[0] != '-')
        {
            g_config.model_path = arg;
        }
    }

    return true;
}

int run_cli_mode(const std::string &image_path)
{
    if (!initialize_engine())
    {
        std::cerr << "Failed to initialize OCR engine" << std::endl;
        return 1;
    }

    cv::Mat image = load_image_from_file(image_path);
    if (image.empty())
    {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return 1;
    }

    auto results = g_ocr_engine->Run(image);

    for (const auto &res : results)
    {
        std::cout << res.line.text << std::endl;
    }

    return 0;
}

int run_pipe_mode()
{
    if (!initialize_engine())
    {
        std::cerr << "Failed to initialize OCR engine" << std::endl;
        return 1;
    }

    std::string buffer;
    buffer.reserve(8192);

    while (g_running)
    {
        char ch;
        std::cin.read(&ch, 1);

        if (std::cin.eof())
        {
            break;
        }

        if (ch == '\n' || ch == '\r')
        {
            if (!buffer.empty())
            {
                buffer.erase(remove(buffer.begin(), buffer.end(), '\r'), buffer.end());

                if (!buffer.empty())
                {
                    try
                    {
                        auto input = json::parse(buffer);
                        std::string output = process_request(input);
                        std::cout << output << std::endl;
                    }
                    catch (const json::parse_error &e)
                    {
                        json resp = create_error_response(400, std::string("JSON parse error: ") + e.what());
                        std::cout << resp.dump() << std::endl;
                    }
                }
                buffer.clear();
            }
        }
        else
        {
            buffer.push_back(ch);
        }
    }

    return 0;
}

int run_tcp_mode()
{
    if (!initialize_engine())
    {
        std::cerr << "Failed to initialize OCR engine" << std::endl;
        return 1;
    }

#ifdef _WIN32
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0)
    {
        std::cerr << "WSAStartup failed" << std::endl;
        return 1;
    }
#endif

#ifdef _WIN32
    SOCKET server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == INVALID_SOCKET)
#else
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0)
#endif
    {
        std::cerr << "Failed to create socket" << std::endl;
        return 1;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt, sizeof(opt));

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(static_cast<unsigned short>(g_config.tcp_port));

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0)
    {
        std::cerr << "Failed to bind to port " << g_config.tcp_port << std::endl;
#ifdef _WIN32
        closesocket(server_fd);
        WSACleanup();
#else
        close(server_fd);
#endif
        return 1;
    }

    if (listen(server_fd, 5) < 0)
    {
        std::cerr << "Failed to listen on port" << std::endl;
#ifdef _WIN32
        closesocket(server_fd);
        WSACleanup();
#else
        close(server_fd);
#endif
        return 1;
    }

    if (!set_nonblocking(server_fd))
    {
        std::cerr << "Failed to set server socket to non-blocking" << std::endl;
#ifdef _WIN32
        closesocket(server_fd);
        WSACleanup();
#else
        close(server_fd);
#endif
        return 1;
    }

    std::cout << "TCP server listening on port " << g_config.tcp_port << std::endl;

    std::vector<std::thread> threads;

    while (g_running)
    {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
#ifdef _WIN32
        SOCKET client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd == INVALID_SOCKET)
        {
            int err = WSAGetLastError();
            if (err == WSAEWOULDBLOCK)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            if (g_running)
                std::cerr << "Failed to accept connection: " << err << std::endl;
            break;
        }
#else
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0)
        {
            if (errno == EAGAIN || errno == EWOULDBLOCK)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            if (g_running)
                std::cerr << "Failed to accept connection: " << errno << std::endl;
            break;
        }
#endif

#ifdef _WIN32
        threads.emplace_back([client_fd]() {
            u_long mode = 1;
            ioctlsocket(client_fd, FIONBIO, &mode);

            std::string buffer;
            buffer.reserve(4096);

            char recv_buf[4096];
            int retries = 0;
            const int max_retries = 10;

            while (retries < max_retries)
            {
                int bytes = recv(client_fd, recv_buf, sizeof(recv_buf) - 1, 0);
                if (bytes > 0)
                {
                    recv_buf[bytes] = '\0';
                    buffer += recv_buf;

                    try
                    {
                        auto j = json::parse(buffer);
                        std::string output = process_request(j);

                        send(client_fd, output.c_str(), static_cast<int>(output.size()), 0);
                    }
                    catch (const json::parse_error &)
                    {
                        if (!buffer.empty())
                            retries++;
                        else
                            retries = max_retries;
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                        continue;
                    }
                    break;
                }
                else if (bytes == 0)
                {
                    break;
                }
                else
                {
                    int err = WSAGetLastError();
                    if (err == WSAEWOULDBLOCK)
                    {
                        retries++;
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                        continue;
                    }
                    break;
                }
            }

            closesocket(client_fd);
        });
#else
        threads.emplace_back([client_fd]() {
            int flags = fcntl(client_fd, F_GETFL, 0);
            fcntl(client_fd, F_SETFL, flags | O_NONBLOCK);

            std::string buffer;
            buffer.reserve(4096);

            char recv_buf[4096];
            int retries = 0;
            const int max_retries = 10;

            while (retries < max_retries)
            {
                int bytes = recv(client_fd, recv_buf, sizeof(recv_buf) - 1, 0);
                if (bytes > 0)
                {
                    recv_buf[bytes] = '\0';
                    buffer += recv_buf;

                    try
                    {
                        auto j = json::parse(buffer);
                        std::string output = process_request(j);

                        send(client_fd, output.c_str(), output.size(), 0);
                    }
                    catch (const json::parse_error &)
                    {
                        if (!buffer.empty())
                            retries++;
                        else
                            retries = max_retries;
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                        continue;
                    }
                    break;
                }
                else if (bytes == 0)
                {
                    break;
                }
                else
                {
                    if (errno == EAGAIN || errno == EWOULDBLOCK)
                    {
                        retries++;
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                        continue;
                    }
                    break;
                }
            }

            close(client_fd);
        });
#endif
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

#ifdef _WIN32
    closesocket(server_fd);
    WSACleanup();
#else
    close(server_fd);
#endif

    return 0;
}

int main(int argc, char *argv[])
{
#ifdef _WIN32
    // Retrieve command-line arguments as UTF-16 and convert to UTF-8 so that
    // Chinese paths passed via CLI are decoded correctly.
    std::vector<std::string> utf8_args;
    std::vector<char *> utf8_arg_ptrs;
    {
        int wargc = 0;
        wchar_t **wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
        if (wargv)
        {
            utf8_args.reserve(wargc);
            utf8_arg_ptrs.reserve(wargc);
            for (int i = 0; i < wargc; ++i)
            {
                utf8_args.emplace_back(OCR::WStringToUtf8(wargv[i]));
                utf8_arg_ptrs.emplace_back(utf8_args.back().data());
            }
            LocalFree(wargv);
            argc = wargc;
            argv = utf8_arg_ptrs.data();
        }
    }
#endif

    // Initialize logging (output to stderr to not interfere with JSON responses on stdout)
    // Use warning level to minimize log output during pipe/tcp mode
    static plog::ConsoleAppender<plog::MyFormatter> appender(plog::streamStdErr);
    plog::init(plog::warning, &appender);

    if (!parse_arguments(argc, argv))
        return 0;

    if (g_config.config_path.empty())
    {
        std::cerr << "Error: Config file not specified. Use -c option." << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    if (g_config.verbose)
    {
        std::cout << "Mode: ";
        switch (g_config.mode)
        {
        case RunMode::CLI:
            std::cout << "CLI";
            break;
        case RunMode::PIPE:
            std::cout << "PIPE";
            break;
        case RunMode::TCP:
            std::cout << "TCP";
            break;
        }
        std::cout << std::endl;
        std::cout << "Config: " << g_config.config_path << std::endl;
        std::cout << "Vulkan: " << (g_config.use_vulkan ? "enabled" : "disabled") << std::endl;
    }

#ifdef _WIN32
    SetConsoleCtrlHandler(ConsoleCtrlHandler, TRUE);
#else
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
#endif

    int ret = 0;
    switch (g_config.mode)
    {
    case RunMode::CLI:
        if (g_config.model_path.empty())
        {
            std::cerr << "Error: Image path required for CLI mode" << std::endl;
            ret = 1;
        }
        else
        {
            ret = run_cli_mode(g_config.model_path);
        }
        break;

    case RunMode::PIPE:
        ret = run_pipe_mode();
        break;

    case RunMode::TCP:
        ret = run_tcp_mode();
        break;
    }

    // Explicitly destroy the OCR engine and Vulkan gpu instance before
    // process exit to avoid teardown-order crashes in ncnn.
    g_ocr_engine.reset();

#ifdef ENABLE_VULKAN
    ncnn::destroy_gpu_instance();
#endif

    return ret;
}
