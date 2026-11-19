#include <onnxruntime_cxx_api.h>
#include <Accelerate/Accelerate.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <sstream>

namespace {
constexpr int kImageSize = 224;
constexpr int kChannels = 3;
constexpr int kFeatureDim = 512;
struct TextIndex {
    std::vector<float> features;
    std::vector<std::string> captions;
    std::vector<int> image_indices;
    int num_captions = 0;
    int feature_dim = 512;
};

TextIndex load_text_index(const std::string& index_dir) {
    TextIndex index;

    const std::string features_path =
        index_dir + "/text_features.bin";
    const std::string captions_path =
        index_dir + "/captions.txt";
    const std::string image_indices_path =
        index_dir + "/caption_image_indices.txt";

    std::ifstream features_file(
        features_path,
        std::ios::binary | std::ios::ate
    );

    if (!features_file) {
        throw std::runtime_error(
            "Cannot open text features: " + features_path
        );
    }

    const std::streamsize feature_bytes = features_file.tellg();
    const std::streamsize bytes_per_caption =
        index.feature_dim * sizeof(float);

    if (feature_bytes % bytes_per_caption != 0) {
        throw std::runtime_error(
            "Invalid text feature file size"
        );
    }

    index.num_captions = static_cast<int>(
        feature_bytes / bytes_per_caption
    );

    index.features.resize(
        static_cast<size_t>(index.num_captions)
        * index.feature_dim
    );

    features_file.seekg(0, std::ios::beg);
    features_file.read(
        reinterpret_cast<char*>(index.features.data()),
        feature_bytes
    );

    std::ifstream captions_file(captions_path);
    if (!captions_file) {
        throw std::runtime_error(
            "Cannot open captions: " + captions_path
        );
    }

    std::string line;
    while (std::getline(captions_file, line)) {
        index.captions.push_back(line);
    }

    std::ifstream image_indices_file(image_indices_path);
    if (!image_indices_file) {
        throw std::runtime_error(
            "Cannot open image indices: " + image_indices_path
        );
    }

    while (std::getline(image_indices_file, line)) {
        if (!line.empty()) {
            index.image_indices.push_back(std::stoi(line));
        }
    }

    if (index.captions.size() != index.num_captions) {
        throw std::runtime_error(
            "Caption count does not match feature count"
        );
    }

    if (index.image_indices.size() != index.num_captions) {
        throw std::runtime_error(
            "Image index count does not match feature count"
        );
    }

    return index;
}

std::vector<float> preprocess_image(const cv::Mat& bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    // OpenCLIP preprocess: resize the shorter side to 224, then center crop.
    const int h = rgb.rows;
    const int w = rgb.cols;
    const double scale = static_cast<double>(kImageSize) / std::min(h, w);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(), scale, scale, cv::INTER_CUBIC);

    const int crop_x = std::max(0, (resized.cols - kImageSize) / 2);
    const int crop_y = std::max(0, (resized.rows - kImageSize) / 2);
    cv::Mat cropped = resized(cv::Rect(
        crop_x,
        crop_y,
        std::min(kImageSize, resized.cols - crop_x),
        std::min(kImageSize, resized.rows - crop_y)
    )).clone();

    if (cropped.cols != kImageSize || cropped.rows != kImageSize) {
        cv::resize(cropped, cropped, cv::Size(kImageSize, kImageSize), 0, 0, cv::INTER_CUBIC);
    }

    static const float mean[3] = {
        0.48145466f, 0.4578275f, 0.40821073f
    };
    static const float stdv[3] = {
        0.26862954f, 0.26130258f, 0.27577711f
    };

    std::vector<float> tensor(
        static_cast<size_t>(kChannels) * kImageSize * kImageSize
    );

    for (int c = 0; c < kChannels; ++c) {
        for (int y = 0; y < kImageSize; ++y) {
            const cv::Vec3b* row = cropped.ptr<cv::Vec3b>(y);
            for (int x = 0; x < kImageSize; ++x) {
                const float pixel = static_cast<float>(row[x][c]) / 255.0f;
                const float normalized = (pixel - mean[c]) / stdv[c];
                tensor[
                    static_cast<size_t>(c) * kImageSize * kImageSize
                    + y * kImageSize
                    + x
                ] = normalized;
            }
        }
    }

    return tensor;
}

std::vector<float> run_model(
    Ort::Session& session,
    const std::vector<float>& input,
    const char* input_name,
    const char* output_name
) {
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator,
        OrtMemTypeDefault
    );

    const std::array<int64_t, 4> input_shape = {
        1, kChannels, kImageSize, kImageSize
    };

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(input.data()),
        input.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[] = {input_name};
    const char* output_names[] = {output_name};

    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    auto* output_data = outputs[0].GetTensorMutableData<float>();
    auto output_info = outputs[0].GetTensorTypeAndShapeInfo();

    if (output_info.GetElementCount() != kFeatureDim) {
        throw std::runtime_error(
            "Unexpected output size: " +
            std::to_string(output_info.GetElementCount())
        );
    }

    return std::vector<float>(output_data, output_data + kFeatureDim);
}

double vector_norm(const std::vector<float>& values) {
    double sum = 0.0;
    for (float value : values) {
        sum += static_cast<double>(value) * value;
    }
    return std::sqrt(sum);
}

void write_binary(const std::string& path, const std::vector<float>& values) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot write: " + path);
    }
    out.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float))
    );
}

std::vector<float> read_binary(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Cannot read: " + path);
    }

    const std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);

    if (size != static_cast<std::streamsize>(
        static_cast<size_t>(kChannels) * kImageSize * kImageSize * sizeof(float)
    )) {
        throw std::runtime_error("Unexpected input binary size");
    }

    std::vector<float> values(
        static_cast<size_t>(kChannels) * kImageSize * kImageSize
    );
    in.read(
        reinterpret_cast<char*>(values.data()),
        size
    );
    return values;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5 && argc != 6) {
    std::cerr
        << "Usage:\n"
        << "  " << argv[0]
        << " model.onnx image.jpg output.bin index_dir\n"
        << "  " << argv[0]
        << " --tensor model.onnx input.bin output.bin index_dir\n";
    return 1;
}

const bool tensor_mode = std::string(argv[1]) == "--tensor";

if (tensor_mode && argc != 6) {
    std::cerr << "Tensor mode requires model, input, output and index_dir.\n";
    return 1;
}

if (!tensor_mode && argc != 5) {
    std::cerr << "Image mode requires model, image, output and index_dir.\n";
    return 1;
}

const std::string model_path = tensor_mode ? argv[2] : argv[1];
const std::string input_path = tensor_mode ? argv[3] : argv[2];
const std::string output_path = tensor_mode ? argv[4] : argv[3];
const std::string index_dir = tensor_mode ? argv[5] : argv[4];

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "clip_infer");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(4);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        Ort::Session session(env, model_path.c_str(), options);
        Ort::AllocatorWithDefaultOptions allocator;

        auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
        auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);
        const char* input_name = input_name_alloc.get();
        const char* output_name = output_name_alloc.get();
        TextIndex text_index = load_text_index(index_dir);
        

std::cout << "num_captions: "
          << text_index.num_captions << "\n";
std::cout << "captions: "
          << text_index.captions.size() << "\n";
std::cout << "image_indices: "
          << text_index.image_indices.size() << "\n";

          double image_read_ms = 0.0;
          double preprocess_ms = 0.0;
        cv::Mat loaded_image;

        std::vector<float> input;

        if (tensor_mode) {
            input = read_binary(input_path);
        } else {
            const auto read_start =
    std::chrono::high_resolution_clock::now();

loaded_image = cv::imread(input_path, cv::IMREAD_COLOR);

const auto read_end =
    std::chrono::high_resolution_clock::now();

image_read_ms = std::chrono::duration<double, std::milli>(
    read_end - read_start
).count();

if (loaded_image.empty()) {
    throw std::runtime_error("Cannot read image: " + input_path);
}

const auto preprocess_start =
    std::chrono::high_resolution_clock::now();

input = preprocess_image(loaded_image);

const auto preprocess_end =
    std::chrono::high_resolution_clock::now();

preprocess_ms = std::chrono::duration<double, std::milli>(
    preprocess_end - preprocess_start
).count();
        }

        // Warm up.
        for (int i = 0; i < 3; ++i) {
            run_model(session, input, input_name, output_name);
        }

        const int runs = 10;
        double total_ms = 0.0;
        std::vector<float> features;

        for (int i = 0; i < runs; ++i) {
            const auto start = std::chrono::high_resolution_clock::now();
            features = run_model(session, input, input_name, output_name);
            const auto end = std::chrono::high_resolution_clock::now();
            total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        }

        write_binary(output_path, features);

        std::cout << "model: " << model_path << "\n";
        std::cout << "input: " << input_path << "\n";
        std::cout << "output: " << output_path << "\n";
        std::cout << "feature_dim: " << features.size() << "\n";
        std::cout << "feature_norm: " << vector_norm(features) << "\n";
        std::cout << "mean_ms: " << total_ms / runs << "\n";
        std::cout << "first_values: ";
        for (int i = 0; i < std::min(8, static_cast<int>(features.size())); ++i) {
            std::cout << features[i] << (i + 1 == 8 ? "\n" : " ");
        }
        const auto retrieval_start =
    std::chrono::high_resolution_clock::now();

        const int top_k = 5;

std::vector<float> score_values(
    text_index.num_captions,
    0.0f
);

// 25014 x 512 matrix-vector multiplication using Apple Accelerate.
cblas_sgemv(
    CblasRowMajor,
    CblasNoTrans,
    text_index.num_captions,
    kFeatureDim,
    1.0f,
    text_index.features.data(),
    kFeatureDim,
    features.data(),
    1,
    0.0f,
    score_values.data(),
    1
);

std::vector<std::pair<float, int>> scores;
scores.reserve(text_index.num_captions);

for (int i = 0; i < text_index.num_captions; ++i) {
    scores.emplace_back(score_values[i], i);
}

const int effective_top_k = std::min(
    top_k,
    static_cast<int>(scores.size())
);

std::partial_sort(
    scores.begin(),
    scores.begin() + effective_top_k,
    scores.end(),
    [](const auto& a, const auto& b) {
        return a.first > b.first;
    }
);

const auto retrieval_end =
    std::chrono::high_resolution_clock::now();

const double retrieval_ms =
    std::chrono::duration<double, std::milli>(
        retrieval_end - retrieval_start
    ).count();

std::cout << "\nTop-" << effective_top_k << " captions:\n";

for (int rank = 0; rank < effective_top_k; ++rank) {
    const int caption_index = scores[rank].second;
    const float score = scores[rank].first;

    std::cout
        << rank + 1
        << ". score=" << score
        << " | " << text_index.captions[caption_index]
        << "\n";
}

    const double model_ms = total_ms / runs;

const double end_to_end_ms =
    image_read_ms
    + preprocess_ms
    + model_ms
    + retrieval_ms;

std::cout << "\n========== End-to-end latency ==========\n";
std::cout << "image_read_ms: " << image_read_ms << "\n";
std::cout << "preprocess_ms: " << preprocess_ms << "\n";
std::cout << "model_ms: " << model_ms << "\n";
std::cout << "topk_ms: " << retrieval_ms << "\n";
std::cout << "end_to_end_ms: " << end_to_end_ms << "\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
