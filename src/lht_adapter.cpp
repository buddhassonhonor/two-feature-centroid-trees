// Local adapter around the unchanged official LHT.h (retrieved 2026-10-03).
// Input/output and deterministic C RNG only; no numerical corrections to LHT.
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include "../research/lht_source/LHT.h"
double forest_rate=1, beta=1, feature_train_size=0;
int numberLF_trees=0, minSamples2Split=16, constancyLimit=0;
size_t feature_dim=0,treeNo=0,class_num=0;
std::vector<std::vector<double>> train_features;
std::vector<uint8_t> train_labels;
double reinforcement(uint8_t action,uint8_t label){return action==label?1:0;}

int main(int argc,char** argv){
    if(argc!=6)throw std::runtime_error("train_file eval_file output_file min_split seed");
    minSamples2Split=std::stoi(argv[4]);srand(std::stoi(argv[5]));
    std::ifstream input(argv[1]);size_t n;input>>n>>feature_dim>>class_num;
    feature_train_size=n;numberLF_trees=(int)class_num;
    train_features.assign(n,std::vector<double>(feature_dim));train_labels.resize(n);
    for(size_t i=0;i<n;++i){for(auto& v:train_features[i])input>>v;int y;input>>y;train_labels[i]=(uint8_t)(y+1);}
    initCentroidSampleNumbers((int)feature_dim);
    std::vector<cLF_tree*> trees;
    for(treeNo=0;treeNo<class_num;++treeNo){
        auto t=create_LF_tree();t->setTreeNumber(treeNo);t->setSensorNumber((uint32_t)feature_dim);t->growTree();trees.push_back(t);
    }
    std::ifstream evaluation(argv[2]);size_t ne,pe; evaluation>>ne>>pe;
    if(pe!=feature_dim)throw std::runtime_error("feature dimensions differ");
    std::ofstream output(argv[3]);output<<std::setprecision(17);
    size_t nonfinite=0;
    for(size_t i=0;i<ne;++i){
        std::vector<double>x(pe);for(auto&v:x)evaluation>>v;
        std::vector<double> scores;
        for(auto t:trees){double weight=0;double v=t->evalBoundedWeightedSB(x,weight);if(!std::isfinite(v))++nonfinite;scores.push_back(v);}
        for(size_t c=0;c<class_num;++c){if(c)output<<',';output<<scores[c];}output<<'\n';
    }
    for(auto t:trees)delete t;
    std::cerr<<"nonfinite="<<nonfinite<<" leaves="<<leafCount<<"\n";
    return nonfinite?2:0;
}
