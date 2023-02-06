
#include "gui_application/graphics/graphics_objects.hpp"

#include "Magnum/GL/DefaultFramebuffer.h"
#include "graphics_objects.hpp"

namespace mpex::Graphics {

namespace {

constexpr double wheelbase_m = 3.0;
constexpr float avg_length_m{SCALE(4.5f)};
constexpr float avg_width_m{SCALE(1.75f)};
constexpr float avg_height_m{SCALE(1.75f)};

} // namespace

TrajectoryEntities::TrajectoryEntities(Scene3D &scene, const int horizon_points)
{
    vehicle_extent_ = Vector3{avg_width_m, avg_height_m, avg_length_m};

    for (size_t i{0UL}; i < horizon_points; ++i)
    {
        auto cube = std::make_shared<Object3D>(&scene);
        objects_.emplace_back(std::move(cube));
    }
}

const std::vector<std::shared_ptr<Object3D>> &TrajectoryEntities::get_objects() const
{
    return objects_;
}

void TrajectoryEntities::set_state_at(const size_t idx, const float x, const float y, const float yaw, const float z)
{
    get_objects()
        .at(idx)
        ->resetTransformation()
        .scale(get_vehicle_extent())
        .translate(Magnum::Vector3(0.0f, 0.0f, SCALE(static_cast<float>(wheelbase_m) / 2))) // move half wheelbase forward
        .rotateY(Magnum::Math::Rad(static_cast<float>(yaw)))
        .translate(Magnum::Vector3(SCALE(y), SCALE(z), SCALE(x)));
}

const Vector3 &TrajectoryEntities::get_vehicle_extent() const
{
    return vehicle_extent_;
}

VertexColorDrawable::VertexColorDrawable(
    Object3D &object, Shaders::VertexColorGL3D &shader, GL::Mesh &mesh, SceneGraph::DrawableGroup3D &drawables)
    : SceneGraph::Drawable3D{object, &drawables}, _shader(shader), _mesh(mesh)
{
}

void VertexColorDrawable::draw(const Magnum::Matrix4 &transformation, SceneGraph::Camera3D &camera)
{
    _shader.setTransformationProjectionMatrix(camera.projectionMatrix() * transformation).draw(_mesh);
}

WireframeDrawable::WireframeDrawable(
    Object3D &object, Shaders::MeshVisualizerGL3D &shader, GL::Mesh &mesh, SceneGraph::DrawableGroup3D &drawables, const Color3 &color)
    : SceneGraph::Drawable3D{object, &drawables}, _shader(shader), _mesh(mesh), _color(color)
{
}

void WireframeDrawable::draw(const Magnum::Matrix4 &transformation, SceneGraph::Camera3D &camera)
{
    _shader.setColor(_color)
        .setWireframeColor(_color)
        .setViewportSize(Vector2{GL::defaultFramebuffer.viewport().size()})
        .setTransformationProjectionMatrix(camera.projectionMatrix() * transformation)
        .draw(_mesh);
}

FlatDrawable::FlatDrawable(Object3D &object, Shaders::FlatGL3D &shader, GL::Mesh &mesh, SceneGraph::DrawableGroup3D &drawables)
    : SceneGraph::Drawable3D{object, &drawables}, _shader(shader), _mesh(mesh)
{
}

void FlatDrawable::draw(const Magnum::Matrix4 &transformation, SceneGraph::Camera3D &camera)
{
    _shader
        //.setColor(0x74747492_rgbaf) // gray color
        .setTransformationProjectionMatrix(camera.projectionMatrix() * transformation)
        .draw(_mesh);
}

LaneEntity::LaneEntity(
    environment::Corridor &lane, Scene3D &scene, Shaders::VertexColorGL3D &vertex_color_shader, SceneGraph::DrawableGroup3D &drawable_group)
    : lane_(lane)
{

    centerline_drawable_ = std::make_shared<Graphics::LineEntity>(scene);
    left_boundary_drawable_ = std::make_shared<Graphics::LineEntity>(scene);
    right_boundary_drawable_ = std::make_shared<Graphics::LineEntity>(scene);

    auto new_drawable = [&](auto &drawable) {
        new Graphics::VertexColorDrawable{*drawable->object_ptr_, vertex_color_shader, drawable->mesh_, drawable_group};
    };
    new_drawable(centerline_drawable_);
    new_drawable(left_boundary_drawable_);
    new_drawable(right_boundary_drawable_);

    const auto gray_color = Magnum::Math::Color3(0.63f, 0.63f, 0.63f);

    auto centerline_data = lane_.centerline_.get_data();
    auto lb_data = lane_.left_boundary_.get_data();
    auto rb_data = lane_.right_boundary_.get_data();

    std::vector<double> z_vals(centerline_data.rows(), 0.0);
    const std::size_t N = centerline_data.rows();
    centerline_drawable_->set_xy(N, centerline_data.data(), centerline_data.data() + N, z_vals.data(), gray_color);
    left_boundary_drawable_->set_xy(lb_data.rows(), lb_data.data(), lb_data.data() + lb_data.rows(), z_vals.data(), gray_color);
    right_boundary_drawable_->set_xy(rb_data.rows(), rb_data.data(), rb_data.data() + rb_data.rows(), z_vals.data(), gray_color);
}

} // namespace mpex::Graphics
