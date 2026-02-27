#!/usr/bin/env python3
"""
Path Follower Node — Sends /mission_goals to Nav2 as NavigateToPose goals.

The directional planner generates semi-goals along the A* path.
This node takes each semi-goal and sends it to Nav2 as a NavigateToPose.
Nav2 does ALL the work: plans its own local path, follows it, moves motors.
When Nav2 reaches the goal, the planner publishes the next semi-goal.

Flow:
  RViz goal → goal_republisher → /my_goal_pose → planner (A*)
                                                      ↓
                                              /mission_goals (semi-goal 1)
                                                      ↓
                                          [this node] → NavigateToPose → Nav2
                                                                          ↓
                                                   Nav2 plans + follows → motors

  When Nav2 reaches semi-goal 1:
    Planner detects → publishes semi-goal 2 → this node → Nav2 → ...repeat
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose


class PathFollowerNode(Node):
    """Sends /mission_goals to Nav2 as NavigateToPose goals, one at a time."""

    def __init__(self):
        super().__init__('path_follower_node')

        # ── Parameters ───
        self.declare_parameter('mission_goals_topic', '/mission_goals')

        mission_topic = self.get_parameter('mission_goals_topic').value

        # ── State ───
        self.current_goal_handle = None
        self.navigating = False
        self.goals_sent = 0
        self.goals_reached = 0

        # ── Action client: Nav2 NavigateToPose ───
        self.nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose')

        # ── Subscriber: mission goals from planner ───
        self.create_subscription(
            PoseStamped, mission_topic, self._mission_goal_cb, 10)

        self.get_logger().info(
            f'PathFollowerNode started\n'
            f'  Subscribes: {mission_topic} (semi-goals from planner)\n'
            f'  Action:     navigate_to_pose (Nav2)\n'
            f'  Waiting for Nav2...'
        )

    # ──────────────────────────────────────────────────────────────────
    # Mission goal callback — send to Nav2
    # ──────────────────────────────────────────────────────────────────
    def _mission_goal_cb(self, msg: PoseStamped):
        """Receive semi-goal from planner, send to Nav2 as NavigateToPose."""
        x = msg.pose.position.x
        y = msg.pose.position.y

        self.get_logger().info(
            f'📍 Semi-goal received: ({x:.2f}, {y:.2f}) → sending to Nav2...'
        )

        # Cancel previous navigation if still active
        if self.navigating and self.current_goal_handle is not None:
            self.get_logger().info('Cancelling previous Nav2 goal...')
            self.current_goal_handle.cancel_goal_async()
            self.navigating = False

        # Wait for Nav2 action server
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(
                '❌ Nav2 NavigateToPose not available! Is Nav2 running?')
            return

        # Build NavigateToPose goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = msg
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Send goal
        self.goals_sent += 1
        self.get_logger().info(
            f'→ Sending NavigateToPose #{self.goals_sent}: ({x:.2f}, {y:.2f})')

        send_future = self.nav_client.send_goal_async(
            goal_msg, feedback_callback=self._feedback_cb)
        send_future.add_done_callback(self._goal_response_cb)

    # ──────────────────────────────────────────────────────────────────
    # Action callbacks
    # ──────────────────────────────────────────────────────────────────
    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('❌ Nav2 rejected the goal!')
            self.navigating = False
            return

        self.get_logger().info('✅ Nav2 accepted — navigating...')
        self.current_goal_handle = goal_handle
        self.navigating = True

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        status = future.result().status

        if status == 4:  # SUCCEEDED
            self.goals_reached += 1
            self.get_logger().info(
                f'★ Goal reached! ({self.goals_reached}/{self.goals_sent}) '
                f'— waiting for next semi-goal from planner...')
        elif status == 5:  # CANCELED
            self.get_logger().info('⚠ Navigation cancelled.')
        elif status == 6:  # ABORTED
            self.get_logger().warn('❌ Navigation aborted by Nav2.')
        else:
            self.get_logger().info(f'Navigation finished with status: {status}')

        self.navigating = False
        self.current_goal_handle = None

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        dist = fb.distance_remaining
        self.get_logger().info(
            f'  Navigating... dist_remaining={dist:.2f}m '
            f'(goal {self.goals_sent})',
            throttle_duration_sec=2.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = PathFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
