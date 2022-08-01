

#include <Corrade/Containers/ArrayViewStl.h>
#include <Eigen/Dense>
#include <Magnum/GL/Buffer.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Vector.h>

#include "base_example.hpp"
#include "gui_application/base_example.hpp"
#include "gui_application/graphics/graphics_objects.hpp"

#include "gui_application/tsdf_example.hpp"

namespace RoboticsSandbox
{

namespace
{
double rastrigins_function(float x1, float x2)
{
  // https://en.wikipedia.org/wiki/Rastrigin_function#:~:text=In%20mathematical%20optimization%2C%20the%20Rastrigin,has%20been%20generalized%20by%20Rudolph.
  constexpr float A{10.0f};
  constexpr float n{2.0};
  return A * n + (x1 * x1 - A * std::cos(2.0 * 3.14f * x1)) + (x2 * x2 - A * std::cos(2.0 * 3.14f * x2));
};

MatrixXf make_matrix()
{
  constexpr size_t n_points{1000UL};
  VectorXf x_vals = VectorXf::LinSpaced(n_points, -5.0f, 5.0f);

  MatrixXf result = MatrixXf::Zero(x_vals.size(), x_vals.size());
  for (size_t i = 0UL; i < x_vals.size(); ++i)
  {
    for (size_t j = 0UL; j < x_vals.size(); ++j)
    {
      result(i, j) = rastrigins_function(x_vals[i], x_vals[j]) / 40.0f;
    }
  }
  return result;
}

} // namespace

TsdfExample::TsdfExample(const Arguments &arguments) : Magnum::Examples::BaseExample(arguments)
{
  std::cout << "constructing..." << std::endl;

  std::vector<Vertex> this_data;
  MatrixXf mat = make_matrix();
  auto min = mat.minCoeff();
  auto max = mat.maxCoeff();

  mat.array() = (mat.array() - min) / (max - min);

  const auto add_to = [&this_data, &mat](size_t i, size_t j)
  {
    float v = std::clamp(mat(i, j), 0.0f, 1.0f);
    Vertex vert{{j / 100.0f, 0.0f, i / 100.0f}, Magnum::Color3(v, 0.0f, 0.0f)};
    this_data.push_back(std::move(vert));
  };

  for (size_t i = 0UL; i + 1UL < mat.cols(); ++i)
  {
    for (size_t j = 0UL; j + 1UL < mat.rows(); ++j)
    {
      // add first triangle for the square
      add_to(i, j);
      add_to(i + 1UL, j);
      add_to(i, j + 1UL);

      // add second triangle for the square
      add_to(i + 1UL, j);
      add_to(i + 1UL, j + 1UL);
      add_to(i, j + 1UL);
    }
  }

  data_ = this_data;

  mesh_ = Magnum::GL::Mesh{};
  mesh_.setPrimitive(Magnum::GL::MeshPrimitive::Triangles)
      .setCount(data_.size())
      .addVertexBuffer(Magnum::GL::Buffer(data_), 0, Magnum::Shaders::VertexColorGL3D::Position{},
                       Magnum::Shaders::VertexColorGL3D::Color3{});
  // .setIndexBuffer(Magnum::GL::Buffer(indices), 0, Magnum::GL::MeshIndexType::UnsignedInt);

  /* Triangle object */
  auto triangle = new Object3D{&_scene};
  new Graphics::VertexColorDrawable{*triangle, _vertexColorShader, mesh_, _drawables};
}

void TsdfExample::execute() {}

void TsdfExample::show_menu()
{
  ImGui::ShowDemoWindow();
}

} // namespace RoboticsSandbox

MAGNUM_APPLICATION_MAIN(RoboticsSandbox::TsdfExample)