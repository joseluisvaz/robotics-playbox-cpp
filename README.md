
This repository is a sandbox to tryout planning and controls algorithms.


So far this project contains an implementation of:
* A simple sampling based model predictive planner using the cross entropy method (CEM). This planner also integrates the intelligent driver model to plan intelligent and interactive agents.
* An iLQR trajectory optimization implementation for a kinematic bicycle model 

For visualization we use the imgui and magnum library, therefore this repository started as a fork/copy of https://github.com/mosra/magnum-integration, which is a template for magnum/imgui projects.

## Dependencies

* [Corrade](https://doc.magnum.graphics/corrade/building-corrade.html#building-corrade-packages-deb)
* [Magnum](https://doc.magnum.graphics/magnum/building.html#building-packages-deb)
* [MagnumIntegration](https://doc.magnum.graphics/magnum/building.html#building-packages-deb)
* [Catch2](https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md#installing-catch2-from-git-repository)
* [easy_profiler](https://github.com/yse/easy_profiler)
* [autodiff](https://autodiff.github.io/installation/)
* [imgui]()
* [eigen]()
