```bash 
ros2 run qcar2_laneseg_acc color_segmentation_node.py --ros-args -p input_image_topic:="/camera/csi_image_3" -p output_mask_topic:="lokita"

ros2 run qcar2_teleop yellow_line_position_node --ros-args -p mask_topic:="lokita"
```
