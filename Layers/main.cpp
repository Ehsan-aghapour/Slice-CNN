#include <iostream>
#include <vector>
#include "tasks.h"
#include "main_layer_checker.h"



int main(){
	for(auto pair:task_names){
		std::string model=pair.first;
		std::cerr<<"checking model "<<model<<std::endl;
		std::vector<std::string> ending;
		for(auto layer:pair.second){
			//std::cerr<<"checking layer: "<<layer<<std::endl;
			if(check(model,layer)){
				ending.push_back(layer);
			}
		}
		std::cerr<<"\n\n***********************\nEnding layers "<<model<<":\n";
		for(auto l : ending){
			std::cerr<<l<<std::endl;
		}
		std::cerr<<ending.size()<<"\n";
	}
	return 0;
}
