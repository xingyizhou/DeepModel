#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/deep_hand_model_layers.hpp"
int forward_seq[31] = { 24, 25, 26, 27, 28, 29, 30, 20, 21, 22, 23, 4, 3, 2, 1, 0, 9, 8, 7, 6, 5, 14, 13, 12, 11, 10, 19, 18, 17, 16, 15 };
int prev_seq[31] = { -1, 24, 24, 24, 27, 28, 29, 24, 24, 24, 24, 20, 4, 3, 2, 1, 21, 9, 8, 7, 6, 22, 14, 13, 12, 11, 23, 19, 18, 17, 16 };
namespace caffe {

template <typename Dtype>
void DeepHandModelLayer<Dtype>::SetupConstantMatrices(){
  //finger 5: thumb
  const_matr[wrist_left_id_in_const] = Matr(trans_y, -bonelen[bone_wrist_left], 0);
  const_matr[wrist_middle_id_in_const] = Matr(trans_y, -bonelen[bone_wrist_middle], 0);
  const_matr[thumb_mcp_id_in_const] = Matr(trans_y, -bonelen[bone_thumb_mcp], 0);
  const_matr[thumb_pip_id_in_const] = Matr(trans_x, bonelen[bone_thumb_pip], 0);
  const_matr[thumb_dip_id_in_const] = Matr(trans_x, bonelen[bone_thumb_dip], 0);
  const_matr[thumb_tip_id_in_const] = Matr(trans_x, bonelen[bone_thumb_tip], 0);
  for (int k = 0; k < 4; k++) {//finger 1 - finger 4 (little, ring, middle, index)
    const_matr[finger_mcp_start_id_in_const + k] = Matr(trans_y, bonelen[bone_mcp_start + k], 0);
    const_matr[finger_base_start_id_in_const + num_of_bones_each_finger * k] = Matr(trans_y, bonelen[bone_base_start + num_of_bones_each_finger * k], 0);
    const_matr[finger_pip_first_start_id_in_const + num_of_bones_each_finger * k] = Matr(trans_y, bonelen[bone_pip_first_start + num_of_bones_each_finger * k], 0);
    const_matr[finger_pip_second_start_id_in_const + num_of_bones_each_finger * k] = Matr(trans_y, bonelen[bone_pip_second_start + num_of_bones_each_finger * k], 0);
    //Two points for DIP in each finger (in NYU dataset)
    const_matr[finger_dip_start_id_in_const + num_of_bones_each_finger * k] = Matr(trans_y, bonelen[bone_dip_start + num_of_bones_each_finger * k], 0);
    const_matr[finger_tip_start_id_in_const + num_of_bones_each_finger * k] = Matr(trans_y, bonelen[bone_tip_start + num_of_bones_each_finger * k], 0);
    //Two points for TIP in each finger
  }
}

template <typename Dtype>
void DeepHandModelLayer<Dtype>::SetupTransformation(){
  //palm center

  Homo_mat[palm_center].pb(mp(trans_x, global_trans_x_id));
  Homo_mat[palm_center].pb(mp(trans_y, global_trans_y_id));
  Homo_mat[palm_center].pb(mp(trans_z, global_trans_z_id)); //global translation DoF 0-2

  //wrist left
  for (int i = 0; i < Homo_mat[palm_center].size(); i++)
    Homo_mat[wrist_left].pb(Homo_mat[palm_center][i]);
  Homo_mat[wrist_left].pb(mp(rot_z, global_rot_z_id));
  Homo_mat[wrist_left].pb(mp(rot_x, global_rot_x_id));
  Homo_mat[wrist_left].pb(mp(rot_y, global_rot_y_id));

  Homo_mat[wrist_left].pb(mp(rot_z, wrist_left_rot_z_id));
  Homo_mat[wrist_left].pb(mp(rot_x, wrist_left_rot_x_id));
  Homo_mat[wrist_left].pb(mp(rot_y, wrist_left_rot_y_id));
  Homo_mat[wrist_left].pb(mp(Const_Matr, wrist_left_id_in_const));

  //wrist middle(carpals)
  for (int i = 0; i < Homo_mat[palm_center].size(); i++)
    Homo_mat[wrist_middle].pb(Homo_mat[palm_center][i]);
  Homo_mat[wrist_middle].pb(mp(rot_z, global_rot_z_id));
  Homo_mat[wrist_middle].pb(mp(rot_x, global_rot_x_id));
  Homo_mat[wrist_middle].pb(mp(rot_y, global_rot_y_id));

  Homo_mat[wrist_middle].pb(mp(rot_z, wrist_middle_rot_z_id));
  Homo_mat[wrist_middle].pb(mp(rot_x, wrist_middle_rot_x_id));
  Homo_mat[wrist_middle].pb(mp(rot_y, wrist_middle_rot_y_id));
  Homo_mat[wrist_middle].pb(mp(Const_Matr, wrist_middle_id_in_const));

  //thumb MCP (wrist right metacarpals)
  for (int i = 0; i < Homo_mat[palm_center].size(); i++)
    Homo_mat[thumb_mcp].pb(Homo_mat[palm_center][i]);
  Homo_mat[thumb_mcp].pb(mp(rot_z, global_rot_z_id));
  Homo_mat[thumb_mcp].pb(mp(rot_x, global_rot_x_id));
  Homo_mat[thumb_mcp].pb(mp(rot_y, global_rot_y_id));

  Homo_mat[thumb_mcp].pb(mp(rot_z, thumb_mcp_rot_z_id));
  Homo_mat[thumb_mcp].pb(mp(rot_x, thumb_mcp_rot_x_id));
  Homo_mat[thumb_mcp].pb(mp(rot_y, thumb_mcp_rot_y_id));
  Homo_mat[thumb_mcp].pb(mp(Const_Matr, thumb_mcp_id_in_const));

  //thumb PIP
  for (int i = 0; i < Homo_mat[thumb_mcp].size(); i++)
    Homo_mat[thumb_pip].pb(Homo_mat[thumb_mcp][i]);
  Homo_mat[thumb_pip].pb(mp(rot_z, thumb_pip_rot_z_id));
  Homo_mat[thumb_pip].pb(mp(rot_y, thumb_pip_rot_y_id));
  Homo_mat[thumb_pip].pb(mp(Const_Matr, thumb_pip_id_in_const));

  //thumb DIP
  for (int i = 0; i < Homo_mat[thumb_pip].size(); i++)
    Homo_mat[thumb_dip].pb(Homo_mat[thumb_pip][i]);
  Homo_mat[thumb_dip].pb(mp(rot_z, thumb_dip_rot_z_id));
  Homo_mat[thumb_dip].pb(mp(Const_Matr, thumb_dip_id_in_const));

  //thumb TIP
  for (int i = 0; i < Homo_mat[thumb_dip].size(); i++)
    Homo_mat[thumb_tip].pb(Homo_mat[thumb_dip][i]);
  Homo_mat[thumb_tip].pb(mp(rot_z, thumb_tip_rot_z_id));
  Homo_mat[thumb_tip].pb(mp(Const_Matr, thumb_tip_id_in_const));

  //Finger 1-4
  for (int k = 0; k < 4; k++)
  {
    //finger mcp
    for (int i = 0; i < Homo_mat[palm_center].size(); i++)
      Homo_mat[finger_mcp_start + k].pb(Homo_mat[palm_center][i]);
    Homo_mat[finger_mcp_start + k].pb(mp(rot_z, global_rot_z_id));
    Homo_mat[finger_mcp_start + k].pb(mp(rot_x, global_rot_x_id));
    Homo_mat[finger_mcp_start + k].pb(mp(rot_y, global_rot_y_id));

    Homo_mat[finger_mcp_start + k].pb(mp(rot_z, finger_mcp_rot_z_start_id + num_of_dof_each_mcp * k));
    Homo_mat[finger_mcp_start + k].pb(mp(rot_x, finger_mcp_rot_x_start_id + num_of_dof_each_mcp * k));
    Homo_mat[finger_mcp_start + k].pb(mp(rot_y, finger_mcp_rot_y_start_id + num_of_dof_each_mcp * k));
    Homo_mat[finger_mcp_start + k].pb(mp(Const_Matr, finger_mcp_start_id_in_const + k));

    //finger base
    for (int i = 0; i < Homo_mat[finger_mcp_start + k].size(); i++)
      Homo_mat[finger_base_start + num_of_bones_each_finger * k].pb(Homo_mat[finger_mcp_start + k][i]);
    Homo_mat[finger_base_start + num_of_bones_each_finger * k].pb(mp(rot_z, finger_base_rot_z_start_id + num_of_dof_each_finger * k));
    Homo_mat[finger_base_start + num_of_bones_each_finger * k].pb(mp(rot_x, finger_base_rot_x_start_id + num_of_dof_each_finger * k));
    Homo_mat[finger_base_start + num_of_bones_each_finger * k].pb(mp(Const_Matr, finger_base_start_id_in_const + num_of_bones_each_finger * k));

    //finger pip first
    for (int i = 0; i < Homo_mat[finger_base_start + num_of_bones_each_finger * k].size(); i++)
      Homo_mat[finger_pip_first_start + num_of_bones_each_finger * k].pb(Homo_mat[finger_base_start + num_of_bones_each_finger * k][i]);
    Homo_mat[finger_pip_first_start + num_of_bones_each_finger * k].pb(mp(rot_x, finger_pip_rot_x_start_id + num_of_dof_each_finger * k));
    Homo_mat[finger_pip_first_start + num_of_bones_each_finger * k].pb(mp(Const_Matr, finger_pip_first_start_id_in_const + num_of_bones_each_finger * k));

    //finger pip second
    for (int i = 0; i < Homo_mat[finger_pip_first_start + num_of_bones_each_finger * k].size(); i++)
      Homo_mat[finger_pip_second_start + num_of_bones_each_finger * k].pb(Homo_mat[finger_pip_first_start + num_of_bones_each_finger * k][i]);
    Homo_mat[finger_pip_second_start + num_of_bones_each_finger * k].pb(mp(Const_Matr, finger_pip_second_start_id_in_const + num_of_bones_each_finger * k));

    //finger dip
    for (int i = 0; i < Homo_mat[finger_pip_second_start + num_of_bones_each_finger * k].size(); i++)
      Homo_mat[finger_dip_start + num_of_bones_each_finger * k].pb(Homo_mat[finger_pip_second_start + num_of_bones_each_finger * k][i]);
    Homo_mat[finger_dip_start + num_of_bones_each_finger * k].pb(mp(rot_x, finger_dip_rot_x_start_id + num_of_dof_each_finger * k));
    Homo_mat[finger_dip_start + num_of_bones_each_finger * k].pb(mp(Const_Matr, finger_dip_start_id_in_const + num_of_bones_each_finger * k));

    //finger tip
    for (int i = 0; i < Homo_mat[finger_dip_start + num_of_bones_each_finger * k].size(); i++)
      Homo_mat[finger_tip_start + num_of_bones_each_finger * k].pb(Homo_mat[finger_dip_start + num_of_bones_each_finger * k][i]);
    Homo_mat[finger_tip_start + num_of_bones_each_finger * k].pb(mp(Const_Matr, finger_tip_start_id_in_const + num_of_bones_each_finger * k));
  }
}

template <typename Dtype>
void DeepHandModelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int n;
  FILE *fin = fopen("configuration/DofFixedId.txt", "r");
  fscanf(fin, "%d", &n);
  for (int i = 0; i < ParamNum; i++) 
      isFixed[i] = 0;
  for (int i = 0; i < n; i++) { 
    int id; 
    fscanf(fin, "%d", &id); 
    isFixed[id] = 1; 
  }
  fclose(fin);

  //load initial Homo_mat[i]ation matrices
  fin = fopen("configuration/InitialParameters.txt", "r");
  for (int i = 0; i < ParamNum; i++)
  {
    fscanf(fin, "%lf", &initparam[i]);    
  }
  fclose(fin);   

  //load initial bone length(fixed number)
  fin = fopen("configuration/BoneLength.txt", "r");
  for (int i = 0; i < BoneNum; i++)
  {
    fscanf(fin, "%lf", &bonelen[i]);
  }
  fclose(fin);

  SetupConstantMatrices();
  SetupTransformation();
}


template <typename Dtype>
void DeepHandModelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = JointNum * 3;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void DeepHandModelLayer<Dtype>::Forward(Matr mat, int i, int bottomId, int prevsize, const Dtype *bottom_data)
{  
  for (int r = prevsize; r < Homo_mat[i].size(); r++) {
    int opt = Homo_mat[i][r].first;
    int id = Homo_mat[i][r].second;
    if (opt == Const_Matr) {           //constant matrix
      mat = mat * const_matr[id];
    }
    else {
      if (isFixed[id]) mat = mat * Matr(opt, initparam[id], 0);
      else mat = mat * Matr(opt, bottom_data[bottomId + id] + initparam[id], 0);
    }
  }
  prevmat[i] = mat;
  x[i] = prevmat[i] * Vec(0.0, 0.0, 0.0, 1.0);  
}


template <typename Dtype>
void DeepHandModelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int batSize = (bottom[0]->shape())[0];  
  for (int t = 0; t < batSize; t++) {
    int bottomId = t * ParamNum;    
    int topId = t * JointNum * 3;
    for (int i = 0; i < JointNum; i++) {
      int id = forward_seq[i];
      Matr mat;
      if (prev_seq[i] != -1) mat = prevmat[prev_seq[i]];
      Forward(mat, id, bottomId, prev_seq[i] == -1 ? 0 : Homo_mat[prev_seq[i]].size(), bottom_data);        
    }
    for (int i = 0; i < JointNum; i++) {
      top_data[topId + i * 3] = x[i][0];
      top_data[topId + i * 3 + 1] = x[i][1];
      top_data[topId + i * 3 + 2] = x[i][2];
    }
  }
}

template <typename Dtype>
void DeepHandModelLayer<Dtype>::Backward(int bottomId, int i, 
    std::vector<std::pair<int, int> > mat, 
    const Dtype *bottom, Vec x) {
  Vec nowx(x[0], x[1], x[2], x[3]);
  for (int r = mat.size() - 1; r >= 0; r--) {
    int opt = mat[r].first;
    int id = mat[r].second;
    if (opt == Const_Matr) {//constant matrix trans
      for (int j = 0; j < ParamNum; j++) {
        f[i][j] = const_matr[id] * f[i][j]; //trans latter'      
      }
      nowx = const_matr[id] * nowx;
    }
    else { //normal w.r.t x y z  or global translation
      if (isFixed[id]) {
        for (int j = 0; j < ParamNum; j++) {
          f[i][j] = Matr(opt, initparam[id], 0) * f[i][j];
        }
        nowx = Matr(opt, initparam[id], 0) * nowx;
      }
      else
      {
        Matr derivative = Matr(opt, bottom[bottomId + id] + initparam[id], 1);  //\theta + {\theta}_0
        for (int j = 0; j < ParamNum; j++) {
          if (j == id) {
            f[i][j] = derivative * nowx + Matr(opt, bottom[bottomId + id] + initparam[id], 0) * f[i][j];
          }
          else {//irrelevant to the j-th dimension of DoF vector
            f[i][j] = Matr(opt, bottom[bottomId + id] + initparam[id], 0) * f[i][j];
          }
        }
        nowx = Matr(opt, bottom[bottomId + id] + initparam[id], 0) * nowx;
      }
    }
  }
}

//Key idea: (ABCD)'=A'(BCD)+A(BCD)'    (BCD)'=B'(CD)+B(CD)'   (CD)'=C'D+CD'
//f[i][j][0] : \frac{\partial x[i][0]}{\partial d[j]}  partial of x coordinate value of joint i with regard to the j-th DoF
//f[i][j][1] : \frac{\partial x[i][1]}{\partial d[j]}  partial of y coordinate value of joint i with regard to the j-th DoF
//f[i][j][2] : \frac{\partial x[i][2]}{\partial d[j]}  partial of z coordinate value of joint i with regard to the j-th DoF

template <typename Dtype>
void DeepHandModelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int batSize = (bottom[0]->shape())[0];

    for (int t = 0; t < batSize; t++) {
      int bottomId = t * ParamNum;
      for (int i = 0; i < JointNum; i++) {
        for (int j = 0; j < ParamNum; j++) {
          f[i][j].V[0] = f[i][j].V[1] = f[i][j].V[2] = f[i][j].V[3] = 0.0; //crucial
        }
      }      
      Backward(bottomId, palm_center, Homo_mat[palm_center], bottom_data, Vec(0.0, 0.0, 0.0, 1.0));               //BP palm center      
      Backward(bottomId, wrist_left, Homo_mat[wrist_left], bottom_data, Vec(0.0, 0.0, 0.0, 1.0));                 //BP wrist left      
      Backward(bottomId, wrist_middle, Homo_mat[wrist_middle], bottom_data, Vec(0.0, 0.0, 0.0, 1.0));             //BP wrist middle(carpals)      
      Backward(bottomId, thumb_mcp, Homo_mat[thumb_mcp], bottom_data, Vec(0.0, 0.0, 0.0, 1.0));                   //BP wrist right(thumb MCP metacarpals)      
      Backward(bottomId, thumb_pip, Homo_mat[thumb_pip], bottom_data, Vec(0.0, 0.0, 0.0, 1.0));                   //BP thumb PIP      
      Backward(bottomId, thumb_dip, Homo_mat[thumb_dip], bottom_data, Vec(0.0, 0.0, 0.0, 1.0));                   //BP thumb DIP      
      Backward(bottomId, thumb_tip, Homo_mat[thumb_tip], bottom_data, Vec(0.0, 0.0, 0.0, 1.0));                   //BP thumb TIP      
      
      //BP Finger 1-4      
      for (int k = 0; k < 4; k++) {
        //BP finger MCP finger PIP first finger PIP second finger DIP finger TIP    
        Backward(bottomId, finger_mcp_start + k, Homo_mat[finger_mcp_start + k], bottom_data, Vec(0.0, 0.0, 0.0, 1.0));
        Backward(bottomId, finger_base_start + num_of_bones_each_finger * k, Homo_mat[finger_base_start + num_of_bones_each_finger * k], bottom_data, Vec(0.0, 0.0, 0.0, 1.0));
        Backward(bottomId, finger_pip_first_start + num_of_bones_each_finger * k, Homo_mat[finger_pip_first_start + num_of_bones_each_finger * k], bottom_data, Vec(0.0, 0.0, 0.0, 1.0));
        Backward(bottomId, finger_pip_second_start + num_of_bones_each_finger * k, Homo_mat[finger_pip_second_start + num_of_bones_each_finger * k], bottom_data, Vec(0.0, 0.0, 0.0, 1.0));
        Backward(bottomId, finger_dip_start + num_of_bones_each_finger * k, Homo_mat[finger_dip_start + num_of_bones_each_finger * k], bottom_data, Vec(0.0, 0.0, 0.0, 1.0));
        Backward(bottomId, finger_tip_start + num_of_bones_each_finger * k, Homo_mat[finger_tip_start + num_of_bones_each_finger * k], bottom_data, Vec(0.0, 0.0, 0.0, 1.0));        
      }  

      //\frac{\partial loss}{\partial d[j]}= \sum_{i=1}^31 {\frac{\partial loss}{\partial x[i][0]} * \frac{\partial x[i][0]}{\partial d[j]}+
      //                                         \frac{\partial loss}{\partial x[i][1]} * \frac{\partial x[i][1]}{\partial d[j]}+
      //                                                    \frac{\partial loss}{\partial x[i][2]} * \frac{\partial x[i][2]}{\partial d[j]} }
      for (int j = 0; j < ParamNum; j++) {
        bottom_diff[bottomId + j] = 0;
        for (int i = 0; i < JointNum; i++) {
          int topId = t * JointNum * 3 + i * 3;
          bottom_diff[bottomId + j] += f[i][j][0] * top_diff[topId] + f[i][j][1] * top_diff[topId + 1] + f[i][j][2] * top_diff[topId + 2];
        }
      }
    }
  }
}

INSTANTIATE_CLASS(DeepHandModelLayer);
REGISTER_LAYER_CLASS(DeepHandModel);
}  // namespace caffe
