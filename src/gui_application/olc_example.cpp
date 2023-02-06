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

/****************************************************************************************
 An "Hello world" example of the DearImGui implementation for the olc::PixelGameEngine
 by Dandistine, available at https:github.com/dandistine/olcPGEDearImGui
 If you found this in a fork repository, please acknowledge the original author.
 This example requires the following sources (and is compatible with the versions)
       https:github.com/ocornut/imgui v. 1.86
       https:github.com/OneLoneCoder/olcPixelGameEngine v. 2.16
 and all requirements therein, such as an OpenGL implementation (like mesa).
 Clone DearImGui source in a folder (imgui) alongside this example.
 From the PGE, you just need olcPixelGameEngine.h
 Your folder should look like this:
 imgui
 compile.sh
 example.cpp
 imgui_impl_pge.h
 olcPixelGameEngine.h
 Under linux, you may compile it like this:
        g++ -o example \
                example.cpp \
                modules/imgui/imgui.cpp \
                modules/imgui/imgui_demo.cpp \
                modules/imgui/imgui_draw.cpp \
                modules/imgui/imgui_widgets.cpp \
                modules/imgui/imgui_tables.cpp \
                modules/imgui/backends/imgui_impl_opengl2.cpp \
                -Imodules/imgui -Imodules/imgui/backends \
                -Imodules/olcPGEDearImGui \
                -Imodules/olcPixelGameEngine \
                -lX11 -lGL -lpthread -lpng -lstdc++fs \
                -std=c++17 -O3
 Based on original example by Dandistine (author of imgui_impl_pge.h)
 If you choose to bind the drawing functions yourself, follow the in-code comments.
 ****************************************************************************************/

#include <Eigen/Dense>
#include <cmath>
#include <memory>

#include "imgui.h"
#include "imgui_impl_opengl2.h"
#include "imgui_internal.h"
#include "imstb_rectpack.h"
#include "imstb_textedit.h"
#include "imstb_truetype.h"

#define OLC_PGEX_DEAR_IMGUI_IMPLEMENTATION
#define PGE_GFX_OPENGL33
#include "imgui_impl_pge.h"

#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

int n_screen_width = 640;
int n_screen_height = 320;

namespace {

struct vec2d
{
    float x;
    float y;
};

struct SE2
{
    vec2d pos;   // Translation
    float angle; // Rotation
};

struct Polygon
{
    std::vector<vec2d> p; // Transformed Points
    vec2d pos;            // Position of shape
    float angle;          // Direction of shape
};

void transform_se2(const SE2 &se2, Polygon &r)
{
    for (int i{0}; i < r.p.size(); ++i)
    {
        r.p[i] =
            {(r.p[i].x * cosf(se2.angle)) - (r.p[i].y * sinf(se2.angle)) + se2.pos.x,
             (r.p[i].x * sinf(se2.angle)) + (r.p[i].y * cosf(se2.angle)) + se2.pos.y};
    }

    r.pos =
        {(r.pos.x * cosf(se2.angle)) - (r.pos.y * sinf(se2.angle)) + se2.pos.x,
         (r.pos.x * sinf(se2.angle)) + (r.pos.y * cosf(se2.angle)) + se2.pos.y};
}

void print_polygon(const Polygon &r)
{
    std::cout << "-------- Print polygon ----" << std::endl;
    for (int i{0}; i < r.p.size(); ++i)
    {
        std::cout << "x: " << r.p[i].x << " y: " << r.p[i].y << std::endl;
    }
    std::cout << "posx: " << r.pos.x << " posy: " << r.pos.y << " ang: " << r.angle << std::endl;
}

void print_SE2(const SE2 &e)
{
    std::cout << "-------- Print SE2 ----" << std::endl;
    std::cout << "posx: " << e.pos.x << " posy: " << e.pos.y << " ang: " << e.angle << std::endl;
}

SE2 operator*(const SE2 &e1, const SE2 &e2)
{
    auto c1 = cosf(e1.angle);
    auto c2 = cosf(e2.angle);
    auto s1 = sinf(e1.angle);
    auto s2 = sinf(e2.angle);
    auto x1 = e1.pos.x;
    auto y1 = e1.pos.y;
    auto x2 = e2.pos.x;
    auto y2 = e2.pos.y;

    // auto c3 = c1 * c2 - s1 * s2;
    // auto s3 = s1 * c2 + c1 * s2;
    // auto x3 = x2 * c1 - y2 * s1 + x1;
    // auto y3 = x2 * s1 + y2 * c1 + y1;

    // return SE2{{x3, y3}, atan2f(s3, c3)};

    auto t1 = e1.angle;
    auto t2 = e2.angle;

    Eigen::MatrixXf matrix_a = Eigen::MatrixXf::Zero(4, 4);
    Eigen::MatrixXf matrix_b = Eigen::MatrixXf::Zero(4, 4);
    matrix_a << c1, -s1, 0, x1, s1, c1, 0, y1, 0, 0, 1, 1, 0, 0, 0, 1;
    matrix_b << c2, -s2, 0, x2, s2, c2, 0, y2, 0, 0, 1, 1, 0, 0, 0, 1;
    Eigen::MatrixXf matrix_c = matrix_a * matrix_b;
    return SE2{{matrix_c(0, 3), matrix_c(1, 3)}, atan2f(matrix_c(1, 0), matrix_c(0, 0))};
}

SE2 inverse(const SE2 &e)
{
    auto c = cosf(e.angle);
    auto s = sinf(e.angle);

    auto x = e.pos.x;
    auto y = e.pos.y;

    // Eigen::MatrixXf matrix = Eigen::MatrixXf::Zero(4, 4);
    // matrix << c, -s, 0, x, s, c, 0, y, 0, 0, 1, 1, 0, 0, 0, 1;
    // matrix = matrix.inverse();
    // return SE2{{matrix(0, 3), matrix(1, 3)}, atan2f(matrix(1, 0), matrix(0, 0))};

    return SE2{{-x, -y}, atan2f(-s, c)};
}

}; // namespace

class Example : public olc::PixelGameEngine
{
    olc::imgui::PGE_ImGUI mImGui;
    std::uint8_t mGameLayer;

  public:
    // PGE_ImGui can automatically call the SetLayerCustomRenderFunction by passing true into the constructor.
    // false is the default value.
    Example() : mImGui(true)
    {

        sAppName = "Dear ImGui Demo";
    }

    // Vector of pointers to shapes used to update the shapes
    // of our on_screen objects
    std::vector<std::shared_ptr<Polygon>> vec_shapes_;

    // Shape of the ego_vehicle
    std::shared_ptr<Polygon> ego_shape_;

    SE2 get_origin_pose()
    {
        return SE2{{n_screen_width / 2, n_screen_height / 2}, 0.0};
    }

    void draw_polygon(const Polygon &r)
    {
        for (int i{0}; i < r.p.size(); ++i)
        {
            DrawLine(r.p[i].x, r.p[i].y, r.p[(i + 1) % r.p.size()].x, r.p[(i + 1) % r.p.size()].y, olc::WHITE);
        }

        DrawCircle(r.pos.x, r.pos.y, /* size */ 2, olc::WHITE);
    }

  public:
    bool OnUserCreate() override
    {
        // Create a new Layer which will be used for the game
        mGameLayer = CreateLayer();
        // The layer is not enabled by default,  so we need to enable it
        EnableLayer(mGameLayer, true);

        // Create the ego shape of our vehicle
        ego_shape_ = std::make_shared<Polygon>();
        ego_shape_->p = {{20.0f, 10.0f}, {20.0f, -10.0f}, {-20.0f, -10.0f}, {-20.0f, 10.0f}};
        ego_shape_->pos = {0, 0};
        ego_shape_->angle = {0.0};

        // Move car to origin pose
        transform_se2(get_origin_pose(), *ego_shape_);

        vec_shapes_.push_back(ego_shape_);

        return true;
    }

    bool OnUserUpdate(float fElapsedTime) override
    {
        // Change the Draw Target to not be Layer 0
        SetDrawTarget(mGameLayer);
        Clear(olc::BLACK);

        // Return car to origin position
        SE2 world_to_car_se2 = {{ego_shape_->pos.x, ego_shape_->pos.y}, ego_shape_->angle};
        SE2 world_to_origin_se2 = get_origin_pose();
        SE2 car_to_origin_se2 = world_to_origin_se2 * inverse(world_to_car_se2);
        SE2 car_to_next_se2 = {{0.0f, 0.0f}, 0.1f};
        transform_se2(car_to_next_se2, *ego_shape_);
        print_polygon(*ego_shape_);

        for (const auto &shape_ptr_ : vec_shapes_)
        {
            draw_polygon(*shape_ptr_);
        }

        // Create and react to your UI here, it will be drawn during the layer draw function
        ImGui::ShowDemoWindow();

        return true;
    }

    /*
            // If the mImGui was constructed with _register_handler = true, this method is not needed (see lines 61 and 80)
            void DrawUI(void) {
                    //This finishes the Dear ImGui and renders it to the screen
                    mImGui.ImGui_ImplPGE_Render();
            }
    */
};

// Much like any PGE example
int main()
{
    Example demo;
    if (demo.Construct(n_screen_width, n_screen_height, 4, 4))
        demo.Start();
    return 0;
}