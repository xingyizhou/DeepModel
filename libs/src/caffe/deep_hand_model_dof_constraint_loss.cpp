#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_hand_model_layers.hpp"
#include "caffe/HandDefine.h"
namespace caffe {

template <typename Dtype>
void DeepHandModelDofConstraintLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(Dtype(1));
  }
  FILE *fin = fopen("configuration/DofConstraintLow.txt", "r");
  for (int i = 0; i < ParamNum; i++) {
    fscanf(fin, "%f", &dofConstraintLow[i]);
  }
  fclose(fin);
  fin = fopen("configuration/DofConstraintUp.txt", "r");
  for (int i = 0; i < ParamNum; i++) {
    fscanf(fin, "%f", &dofConstraintUp[i]);
  }
  fclose(fin);
  fin = fopen("configuration/DofConstraintId.txt", "r");
  for (int i = 0; i < ParamNum; i++) isIgnored[i] = 0;
  int n;
  fscanf(fin, "%d", &n);
  for (int i = 0; i < n; i++) {
    int id;
    fscanf(fin, "%d", &id);
    isIgnored[id] = 1;
  }
  fclose(fin);
}


template <typename Dtype>
void DeepHandModelDofConstraintLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> loss_shape(0);
  top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void DeepHandModelDofConstraintLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int batSize = (bottom[0]->shape())[0];
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype loss = 0;
  for (int t = 0; t < batSize; t++) {
    int Bid = t * ParamNum;
    for (int i = 0; i < ParamNum; i++) {
      if (isIgnored[i]) continue;
      if (bottom_data[Bid + i] > dofConstraintUp[i]) 
        loss += (bottom_data[Bid + i] - dofConstraintUp[i]) * (bottom_data[Bid + i] - dofConstraintUp[i]);
      else if (bottom_data[Bid + i] < dofConstraintLow[i]) 
        loss += (bottom_data[Bid + i] - dofConstraintLow[i]) * (bottom_data[Bid + i] - dofConstraintLow[i]);
  }      
  }
  top[0]->mutable_cpu_data()[0] = loss / batSize;
}

template <typename Dtype>
void DeepHandModelDofConstraintLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int batSize = (bottom[0]->shape())[0];
  const Dtype* bottom_data = bottom[0]->cpu_data();
  float top_diff = top[0]->cpu_diff()[0] / batSize;
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  
  if (propagate_down[0]) {
    for (int t = 0; t < batSize; t++) {
      int Bid = t * ParamNum;
    for (int i = 0; i < ParamNum; i++)
    {
      if (isIgnored[i]) continue;
      if (bottom_data[Bid + i] > dofConstraintUp[i]) 
        bottom_diff[Bid + i] = top_diff * 2 * (bottom_data[Bid + i] - dofConstraintUp[i]);
      else if (bottom_data[Bid + i] < dofConstraintLow[i]) 
        bottom_diff[Bid + i] = top_diff * 2 * (bottom_data[Bid + i] - dofConstraintLow[i]);
      else 
        bottom_diff[Bid + i] = 0;
    }        
    }
  }
}

INSTANTIATE_CLASS(DeepHandModelDofConstraintLossLayer);
REGISTER_LAYER_CLASS(DeepHandModelDofConstraintLoss);

}  // namespace caffe
