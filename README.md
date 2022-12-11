
This repository is a sandbox to tryout planning and controls algorithms.


So far this project contains an implementation of:
* A simple sampling based model predictive planner using the cross entropy method (CEM). This planner also integrates the intelligent driver model to plan intelligent and interactive agents.
* An iLQR trajectory optimization implementation for a kinematic bicycle model 

For visualization we use the imgui and magnum library, therefore this repository started as a fork/copy of https://github.com/mosra/magnum-integration, which is a template for magnum/imgui projects.