
#include <stdlib.h>
#include <sstream>
#include <map>

std::map<std::string,std::string> Models{{"Alex","bvlc_alexnet.caffemodel"},
										 {"Google","bvlc_googlenet.caffemodel"},
										 {"Squeeze","squeezenet_v1.0.caffemodel"},
										 {"Mobile","MobileNet.h5"},
										 {"Res50","ResNet50.h5"}};
std::map<std::string,std::string> Structures{{"Alex","deploy.prototxt"},
											{"Google","deploy.prototxt"},
											{"Squeeze","deploy.prototxt"},
											{"Mobile",""},
											{"Res50",""}};

std::string _dir="/home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/Sub_Model/";



void Slice_Model(std::string model, std::string structure, int start, int end){
	std::ostringstream command;
	command<<"conda run -n rock-kit3 python "<<_dir<<"Sub_Func.py --Start="<<start<<" --End="<<end<<" --Model="<<model<<" --Structure="<<structure;
	system(command.str().c_str());

	
}

void Convert(std::string model, std::string structure){
	//command.str("");
	//command.clear();
	std::ostringstream command;
	command<<"conda run -n rock-kit3 python "<<_dir<<"convert.py "<<structure<<" "<<model;
	system(command.str().c_str());
}
int main(int argc, char *argv[]){
	//Model
	
	std::string CNN=argv[1];
	int start=std::stoi(argv[2]);
	int end=std::stoi(argv[3]);
	
	
	std::string Model=_dir+CNN+"/"+Models[CNN];
	std::string Structure=_dir+CNN+"/"+Structures[CNN];
	//Start and End layers for slicing
	

	std::string Sliced=_dir+CNN+"/"+CNN+"_"+std::to_string(start)+"_"+std::to_string(end)+".prototxt";
	Slice_Model(Model,Structure,start,end);
	//Convert(Model,Sliced);
	return 0;
}
