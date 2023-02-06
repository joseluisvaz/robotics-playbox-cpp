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

namespace mpex { namespace environment {

/// Utility to read track file from csv.
///@param filename The filename where the track file resides, use the global path.
///@returns map with the contents of the track file.
std::unordered_map<std::string, std::vector<double>> read_track_from_csv(std::string filename)
{
    /// Gotten from stack overflow: https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c
    auto split_next_line_into_tokens = [](std::istream &str) {
        std::vector<std::string> result;
        std::string line;
        std::getline(str, line);

        std::stringstream lineStream(line);
        std::string cell;

        while (std::getline(lineStream, cell, ','))
        {
            result.push_back(cell);
        }
        return result;
    };

    std::ifstream file(filename.c_str());
    if (!file)
    {
        throw std::invalid_argument("Track file not found.");
        return {};
    }

    std::unordered_map<std::string, std::vector<double>> output;

    const auto header_tokens = split_next_line_into_tokens(file);
    for (const auto &header : header_tokens)
    {
        output[header] = std::vector<double>();
    }

    while (file)
    {
        auto value_tokens = split_next_line_into_tokens(file);
        if (value_tokens.empty())
        {
            // This if statement handles when there is a last empty line.
            continue;
        }

        for (size_t i{0}; i < header_tokens.size(); ++i)
        {
            output[header_tokens[i]].push_back(5.0 * std::stod(value_tokens[i]));
        }
    }
    return output;
}

}} // namespace mpex::environment