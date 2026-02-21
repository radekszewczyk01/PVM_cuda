#pragma once
// PVM CUDA C++ Implementation
// (C) 2024 - Research implementation based on
// "Unsupervised Learning from Continuous Video in a Scalable Predictive Recurrent Network"
// Piekniewski et al., 2016

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct PVMConfig {
    // Architecture
    std::vector<int> layer_shapes;
    int hidden_block_size    = 7;
    int input_block_size     = 5;
    int input_channels       = 3;
    int lateral_radius       = 2;
    int fan_in_square_size   = 2;
    float fan_in_radius      = 2.0f;
    bool context_exclude_self = false;
    bool send_context_two_layers_back = true;
    bool last_layer_context_to_all    = true;
    bool feed_context_in_complex_layer = false;
    bool polynomial = true;

    // Training
    float initial_learning_rate    = 0.002f;
    float final_learning_rate      = 0.00005f;
    float intermediate_learning_rate = 0.001f;
    float momentum                 = 0.1f;
    long long steps                = 100000000LL;
    int delay_each_layer_learning  = 1000;
    long long delay_final_learning_rate = 1000000LL;
    long long delay_intermediate_learning_rate = 1000000LL;

    // Batch training (new in C++ version)
    int batch_size = 1;  // number of independent sequences trained in parallel

    // Load from a file path
    static PVMConfig from_json(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot open spec file: " + path);
        json j;
        f >> j;
        return from_json(j);
    }

    static PVMConfig from_json(const json& j) {
        PVMConfig c;
        for (auto& s : j["layer_shapes"])
            c.layer_shapes.push_back(std::stoi(s.get<std::string>()));
        c.hidden_block_size    = std::stoi(j.value("hidden_block_size", "7"));
        c.input_block_size     = std::stoi(j.value("input_block_size", "5"));
        c.input_channels       = j.contains("input_channels") ? std::stoi(j["input_channels"].get<std::string>()) : 3;
        c.lateral_radius       = std::stoi(j.value("lateral_radius", "2"));
        c.fan_in_square_size   = std::stoi(j.value("fan_in_square_size", "2"));
        c.fan_in_radius        = std::stof(j.value("fan_in_radius", "2.0"));
        c.context_exclude_self = j.value("context_exclude_self", "0") == "1";
        c.send_context_two_layers_back = j.value("send_context_two_layers_back", "0") == "1";
        c.last_layer_context_to_all    = j.value("last_layer_context_to_all", "0") == "1";
        c.feed_context_in_complex_layer = j.value("feed_context_in_complex_layer", "0") == "1";
        c.polynomial           = j.value("polynomial", "0") == "1";
        c.initial_learning_rate = std::stof(j.value("initial_learning_rate", "0.002"));
        c.final_learning_rate   = std::stof(j.value("final_learning_rate", "0.00005"));
        if (j.contains("intermediate_learning_rate"))
            c.intermediate_learning_rate = (float)j["intermediate_learning_rate"].get<double>();
        c.momentum             = std::stof(j.value("momentum", "0.1"));
        c.steps                = std::stoll(j.value("steps", "100000000"));
        c.delay_each_layer_learning = std::stoi(j.value("delay_each_layer_learning", "1000"));
        c.delay_final_learning_rate = std::stoll(j.value("delay_final_learning_rate", "1000000"));
        c.delay_intermediate_learning_rate = std::stoll(j.value("delay_intermediate_learning_rate", "1000000"));
        return c;
    }

    int num_layers() const { return (int)layer_shapes.size(); }
    int input_size() const { return layer_shapes[0] * input_block_size; }
};
