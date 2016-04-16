//num
#define JointNum 31
#define ParamNum 47
#define BoneNum 30
#define Num_Of_Const_Matr 30
#define num_of_bones_each_finger 5
#define num_of_dof_each_mcp 3
#define num_of_dof_each_finger 4
//end of num



//matr operation
#define rot_x 0
#define rot_y 1
#define rot_z 2
#define trans_x 3
#define trans_y 4
#define trans_z 5
#define Const_Matr 6
//end of matr operation

//DoF id
#define global_trans_x_id 0
#define global_trans_y_id 1
#define global_trans_z_id 2

#define global_rot_x_id 3
#define global_rot_y_id 4
#define global_rot_z_id 5

#define wrist_left_rot_x_id 6
#define wrist_left_rot_y_id 7
#define wrist_left_rot_z_id 8

#define wrist_middle_rot_x_id 9
#define wrist_middle_rot_y_id 10
#define wrist_middle_rot_z_id 11

#define thumb_mcp_rot_x_id 12
#define thumb_mcp_rot_y_id 13
#define thumb_mcp_rot_z_id 14

#define thumb_pip_rot_y_id 15
#define thumb_pip_rot_z_id 16
#define thumb_dip_rot_z_id 17
#define thumb_tip_rot_z_id 18

#define finger_mcp_rot_x_start_id 19
#define finger_mcp_rot_y_start_id 20
#define finger_mcp_rot_z_start_id 21

#define finger_base_rot_x_start_id 31
#define finger_base_rot_z_start_id 32

#define finger_pip_rot_x_start_id 33
#define finger_dip_rot_x_start_id 34
//end of DoF id

//const matrix id

#define wrist_left_id_in_const 0
#define wrist_middle_id_in_const 1
#define thumb_mcp_id_in_const 2
#define thumb_pip_id_in_const 3
#define thumb_dip_id_in_const 4
#define thumb_tip_id_in_const 5
#define finger_mcp_start_id_in_const 6
#define finger_base_start_id_in_const 10
#define finger_pip_first_start_id_in_const 11
#define finger_pip_second_start_id_in_const 12
#define finger_dip_start_id_in_const 13
#define finger_tip_start_id_in_const 14
//end of const matrix id

//bone id
#define bone_wrist_left 24
#define bone_wrist_middle 25
#define bone_thumb_mcp 26
#define bone_thumb_pip 27
#define bone_thumb_dip 28
#define bone_thumb_tip 29
#define bone_mcp_start 20
#define bone_base_start 4
#define bone_pip_first_start 3
#define bone_pip_second_start 2
#define bone_dip_start 1
#define bone_tip_start 0
//end of bone id

//keypoint (some are not real joint) id
#define palm_center 24
#define wrist_left 25
#define wrist_middle 26

#define thumb_mcp 27
#define thumb_pip 28
#define thumb_dip 29
#define thumb_tip 30

#define finger_mcp_start 20
#define finger_base_start 4
#define finger_pip_first_start 3
#define finger_pip_second_start 2
#define finger_dip_start 1
#define finger_tip_start 0

//End of keypoint

// sequence of forward

//

#define pb push_back
#define mp std::make_pair