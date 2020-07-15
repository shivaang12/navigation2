/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2020 Shivang Patel
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 *  * Neither the name of Willow Garage, Inc. nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Shivang Patel
 *********************************************************************/

#include <string>
#include <memory>
#include <cmath>
#include <fstream>
#include <chrono>

#include "ompl/base/spaces/RealVectorBounds.h"
#include "ompl/base/spaces/SE2StateSpace.h"
#include "ompl/geometric/planners/prm/PRMstar.h"
#include "ompl/geometric/planners/prm/LazyPRMstar.h"
#include "ompl/geometric/planners/rrt/RRTstar.h"
#include "ompl/geometric/planners/rrt/RRT.h"
#include "ompl/geometric/planners/rrt/RRTsharp.h"
#include "ompl/geometric/planners/rrt/RRTXstatic.h"
#include "ompl/geometric/planners/rrt/InformedRRTstar.h"
#include "ompl/geometric/planners/rrt/TRRT.h"
#include "ompl/geometric/planners/rrt/BiTRRT.h"
#include "ompl/base/terminationconditions/CostConvergenceTerminationCondition.h"
#include "tf2/utils.h"

#include "nav2_ompl_planner/ompl_planner.hpp"
#include "nav2_util/node_utils.hpp"

namespace nav2_ompl_planner
{

OMPLPlanner::OMPLPlanner()
: tf_(nullptr), node_(nullptr), costmap_(nullptr), collision_checker_(nullptr), is_initialized_(false) {}

void OMPLPlanner::configure(
  rclcpp_lifecycle::LifecycleNode::SharedPtr parent,
  std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent;
  tf_ = tf;
  name_ = name;
  costmap_ros_ = costmap_ros;
  costmap_.reset(costmap_ros->getCostmap());
  global_frame_ = costmap_ros->getGlobalFrameID();
  collision_checker_ = nav2_costmap_2d::FootprintCollisionChecker(costmap_);

  // Parameter initialization
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".solve_time",
    rclcpp::ParameterValue(1.0));
  node_->get_parameter(name_ + ".solve_time", solve_time_);
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".collision_checking_resolution", rclcpp::ParameterValue(
      0.0001));
  node_->get_parameter(name_ + ".collision_checking_resolution", collision_checking_resolution_);
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".allow_unknown", rclcpp::ParameterValue(
      true));
  node_->get_parameter(name_ + ".allow_unknown", allow_unknown_);
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".planner_name", rclcpp::ParameterValue(
      "RRTstar"));
  node_->get_parameter(name_ + ".planner_name", planner_name_);

  RCLCPP_INFO(
    node_->get_logger(), "Configuring plugin %s of type NavfnPlanner",
    name_.c_str());
}

void OMPLPlanner::initialize()
{
  auto bounds = ompl::base::RealVectorBounds(2);
  // bounds.setLow(0, 0.0);
  // bounds.setHigh(0, 1.0);
  // bounds.setLow(1, 0.0);
  // bounds.setHigh(1, 1.0);
  bounds.setLow(0, costmap_->getOriginX());
  bounds.setHigh(0, costmap_->getOriginX() + costmap_->getSizeInMetersX());
  bounds.setLow(1, costmap_->getOriginY());
  bounds.setHigh(1, costmap_->getOriginY() + costmap_->getSizeInMetersY());

  ompl_state_space_ = std::make_shared<ompl::base::SE2StateSpace>();
  ompl_state_space_->as<ompl::base::SE2StateSpace>()->setBounds(bounds);

  ss_.reset(new ompl::geometric::SimpleSetup(ompl_state_space_));
  ss_->setStateValidityChecker(
    [this](const ompl::base::State * state) -> bool {
      return this->isStateValid(state);
    });
  ss_->getSpaceInformation()->setStateValidityCheckingResolution(collision_checking_resolution_);
}

void OMPLPlanner::cleanup()
{
  RCLCPP_INFO(
    node_->get_logger(), "CleaningUp plugin %s of type OMPLPlanner",
    name_.c_str());
}

void OMPLPlanner::activate()
{
  RCLCPP_INFO(
    node_->get_logger(), "Activating plugin %s of type OMPLPlanner",
    name_.c_str());
}

void OMPLPlanner::deactivate()
{
  RCLCPP_INFO(
    node_->get_logger(), "Deactivating plugin %s of type OMPLPlanner",
    name_.c_str());
}

nav_msgs::msg::Path OMPLPlanner::createPlan(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal)
{
  auto start_time = std::chrono::high_resolution_clock::now();
  RCLCPP_INFO(node_->get_logger(), "Passed init of type OMPLPlanner");

  if(!is_initialized_) {
    initialize();
    is_initialized_ = true;
  }

  nav_msgs::msg::Path path;

  ompl::base::ScopedState<> ompl_start(ompl_state_space_);
  ompl::base::ScopedState<> ompl_goal(ompl_state_space_);

  ompl_start[0] = start.pose.position.x;
  ompl_start[1] = start.pose.position.y;
  ompl_start[2] = tf2::getYaw(start.pose.orientation);

  ompl_goal[0] = goal.pose.position.x;
  ompl_goal[1] = goal.pose.position.y;
  ompl_goal[2] = tf2::getYaw(goal.pose.orientation);



  // auto result = isStateValid(ompl_goal[0], ompl_goal[1], ompl_goal[2]);
  // RCLCPP_INFO(node_->get_logger(), "THE VALUE IS %f", result);

  ss_->setStartAndGoalStates(ompl_start, ompl_goal);

  // if (planner_name_ == "RRTstar") {
  //   setPlanner<ompl::geometric::RRTstar>();
  // } else if (planner_name_ == "LazyPRMstar") {
  //   setPlanner<ompl::geometric::LazyPRMstar>();
  // } else if (planner_name_ == "PRMstar") {
  //   setPlanner<ompl::geometric::PRMstar>();
  // } else if (planner_name_ == "RRTsharp") {
  //   setPlanner<ompl::geometric::RRTsharp>();
  // } else if (planner_name_ == "RRTXstatic") {
  //   setPlanner<ompl::geometric::RRTXstatic>();
  // } else if (planner_name_ == "InformedRRTstar") {
  //   setPlanner<ompl::geometric::InformedRRTstar>();
  // } else if (planner_name_ == "RRT") {
  //   setPlanner<ompl::geometric::RRT>();
  // } else {
  //   setPlanner<ompl::geometric::RRTstar>();
  // }

  setPlanner<ompl::geometric::RRTstar>();


  auto problemDef = ss_->getProblemDefinition();
  auto cct = ompl::base::CostConvergenceTerminationCondition(problemDef, 1, 1);
  ss_->setup();

  if (ss_->solve(cct)) {
    RCLCPP_INFO(node_->get_logger(), "Path found!");
    // ss_->simplifySolution(max_simplification_time_);
    auto solution_path = ss_->getSolutionPath();
    path.poses.clear();
    path.header.stamp = node_->now();
    path.header.frame_id = global_frame_;
    // Increasing number of path points
    int min_num_states = round(solution_path.length() / costmap_->getResolution());
    RCLCPP_INFO(node_->get_logger(), "MIN NUM OF STATE %d", min_num_states);
    RCLCPP_INFO(node_->get_logger(), "SOLUTION PATH LENGTH %f", solution_path.length());
    RCLCPP_INFO(node_->get_logger(), "COSTMAP RESOLUTION %f", costmap_->getResolution());
    solution_path.interpolate(min_num_states);

    path.poses.reserve(solution_path.getStates().size());
    for (const auto ptr : solution_path.getStates()) {
      path.poses.push_back(convertWaypoints(*ptr));
    }
    // path.poses[path.poses.size() - 1] = goal;
    RCLCPP_INFO(node_->get_logger(), "Path SIZE IS %d", path.poses.size());
  } else {
    RCLCPP_ERROR(node_->get_logger(), "Path not found!");
  }
  auto stop_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);
  std::ofstream file("time.txt", std::ios_base::app);
  file << std::to_string(duration.count()) << '\n';
  file.close();


  return path;
}

geometry_msgs::msg::PoseStamped OMPLPlanner::convertWaypoints(const ompl::base::State & state)
{
  ompl::base::ScopedState<> ss(ompl_state_space_);
  ss = state;

  geometry_msgs::msg::PoseStamped pose;
  pose.pose.position.x = ss[0];
  pose.pose.position.y = ss[1];
  pose.pose.position.z = 0.0;
  pose.pose.orientation.x = 0.0;
  pose.pose.orientation.y = 0.0;
  pose.pose.orientation.z = std::sin(0.5 * ss[2]);
  pose.pose.orientation.w = std::cos(0.5 * ss[2]);

  return pose;
}

bool OMPLPlanner::isStateValid(const ompl::base::State * state)
{
  if (!ss_->getSpaceInformation()->satisfiesBounds(state)) {
    return false;
  }

  ompl::base::ScopedState<> ss(ompl_state_space_);
  ss = state;
  // unsigned int mx, my;
  double x, y, th;
  x = ss[0];
  y = ss[1];
  th = ss[2];
  // costmap_->worldToMap(x, y, mx, my);

  // auto cost = costmap_->getCost(mx, my);
  auto cost = collision_checker_.footprintCostAtPose(x, y, th, costmap_ros_->getRobotFootprint());
  // RCLCPP_INFO(node_->get_logger(), "COST IS %d", cost);
  if (cost > 252)
  {
    return false;
  }

  return true;
}

double OMPLPlanner::isStateValid(double x, double y, double th)
{
  // x = ss[0];
  // y = ss[1];
  // th = ss[2];
  // costmap_->worldToMap(x, y, mx, my);

  // auto cost = costmap_->getCost(mx, my);
  auto cost = collision_checker_.footprintCostAtPose(x, y, th, costmap_ros_->getRobotFootprint());
  return cost;
  // RCLCPP_INFO(node_->get_logger(), "COST IS %d", cost);
  // if (cost > 253)
  // {
  //   return false;
  // }

  // return true;
}

}  // namespace nav2_ompl_planner

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(nav2_ompl_planner::OMPLPlanner, nav2_core::GlobalPlanner)
