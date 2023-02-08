// Copyright © 2022 <copyright holders>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the “Software”), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE

#include <gui_application/third_party/json/single_include/nlohmann/json.hpp>

namespace mpex { namespace environment {

std::unordered_map<std::string, std::vector<double>> read_track_from_json(const std::string &json_path)
{
    std::ifstream file(json_path.c_str());

    nlohmann::json json;
    file >> json;

    std::unordered_map<std::string, std::vector<double>> data;
    data["X"] = static_cast<std::vector<double>>(json["X"]);
    data["Y"] = static_cast<std::vector<double>>(json["Y"]);
    data["X_i"] = static_cast<std::vector<double>>(json["X_i"]);
    data["Y_i"] = static_cast<std::vector<double>>(json["Y_i"]);
    data["X_o"] = static_cast<std::vector<double>>(json["X_o"]);
    data["Y_o"] = static_cast<std::vector<double>>(json["Y_o"]);

    return data;
}

}} // namespace mpex::environment