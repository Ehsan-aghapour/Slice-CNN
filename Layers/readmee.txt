
bool IStreamPipeline::is_next_layer(std::string name){
		static int index=0;
		if (check_ending(graph_name, name)){
			std::string indent=(index%2)?"":"\t\t\t";
			std::cerr <<indent<< index<<" layer: "<<name << std::endl;
			return true;
		}
		else{
			std::string indent=(index%2)?"":"\t\t\t";
			std::cerr <<indent<<index<< " skipping layer: "<<name << std::endl;
			return false;
		}

		static int index=0;
		static bool ended=false;
		bool ret=false;
		static int conv_index=0;
		std::smatch match;
		std::regex expr;
		static std::string last_node_name="";
		if(graph_name=="yolov3"){
			goto yolov3;
		}
		/*else if(graph_name=="mobilenetv1"){
			goto mobilenetv1;
		}*/


		if(ended){
			index++;
			ended=false;
			ret=true;
			std::string indent=(index%2)?"":"\t\t\t";
			std::cerr <<indent<< index<<" layer: "<<name << std::endl;
		}
		else{
			std::string indent=(index%2)?"":"\t\t\t";
			std::cerr <<indent<<index<< " skipping layer: "<<name << std::endl;
			ret=false;
		}
		/*if(contains(ending_names,name)){
			ended=true;
		}
		else{
			ended=false;
		}*/
		if(ending_tasks.count(name)>0){
			ended=true;
		}
		else{
			ended=false;
		}
		return ret;

		/* patterns that are skipped:
		 * .*_g\\d* all names with _g then a digit like _g2 (for group convs)
		 * .*relu.* all names with relu because in python models relu layers all not separate layers
		 * .*BatchNorm.* names that have batchnorm
		 * .*Linear.* for the last operations in yolov3 (maybe this filter should be added to operation level granularity also [I think it is some kind of activation])
		 */

    	/*
		// ----> This does not work:
		//std::string formatPattern = ".*_g\\d*|.*relu.*|.*linear.*";
		//This works:
		std::string const formatPattern = ".*(_g\\d*|relu|linear).*";
		//However when I added the batchnorm, the later also not works!:
		//std::string const formatPattern = ".*(_g\\d*|relu|linear|batchnorm).*";
		std::regex pattern(formatPattern, std::regex_constants::icase);
		if (regex_search(name, pattern)) {
			std::cerr << "Skipping layer: "<<name << std::endl;
			return false;
		}
		else{
			return true;
		}*/


		//This is for conv layers that have multiple groups like alexnet graph
		// it will be devided into conv2_g0 conv2_g1 conv2 tasks
alexnet:
		//std::regex expr("conv\\D*(\\d+)");
		expr=("conv\\D*(\\d+)");
		if (std::regex_search(name, match, expr) && match.size() > 1) {
			// Convert the matched string to an integer
			int num = std::stoi(match.str(1));
			std::cerr<<"layer "<<name<<" find conv*d which d is: "<<num<<"\n";
			if (conv_index==num){
				std::cerr<<"this is same conv as previous\n";
				std::cerr <<index<< " skipping layer: "<<name << std::endl;
				return false;
			}
			conv_index=num;
		}
mobilenetv1:
		last_node_name=name;
		expr=("Conv2d_\\D*(\\d+)_");
		if (std::regex_search(name, match, expr) && match.size() > 1) {
			// Convert the matched string to an integer
			int num = std::stoi(match.str(1));
			std::cerr<<"layer "<<name<<" find conv2d_n which n is: "<<num<<"\n";
			if (conv_index==num){
				std::cerr<<"this is same conv as previous\n";
				std::string indent=(index%2)?"":"\t\t\t";
				std::cerr <<indent<<index<< " skipping layer: "<<name << std::endl;
				return false;
			}
			else{
				index++;
				std::string indent=(index%2)?"":"\t\t\t";
				std::cerr <<indent<< index<<" layer: "<<name << std::endl;
				ending_tasks.insert(last_node_name);
				last_node_name=name;
				conv_index=num;
				return true;
			}
		}
		expr=("Logits/Conv2d");
		if(regex_search(name,expr)){
			index++;
			std::string indent=(index%2)?"":"\t\t\t";
			std::cerr <<indent<< index<<" layer: "<<name << std::endl;
			ending_tasks.insert(last_node_name);
			last_node_name=name;
			return true;
		}
		else{
			std::string indent=(index%2)?"":"\t\t\t";
			std::cerr <<indent<<index<< " skipping layer: "<<name << std::endl;
			return false;
		}
		return false;
yolov3:


		//Method 1:
		//const std::string formatPattern = "^(?!.*_g\\d*)(?!.*relu)(?!.*batchnorm)(?!.*linear)(.*conv.*|.*fc.*)";
		const std::string formatPattern = "^(?!.*relu)(?!.*batchnorm)(?!.*linear)(.*conv.*|.*fc.*)";
		//If you do not want to skip input and output layers :
		//std::string formatPattern = "^(?!.*_g\\d*)(?!.*relu)(?!.*batchnorm)(?!.*linear)(.*conv.*|.*fc.*|$)";



		std::regex pattern(formatPattern, std::regex_constants::icase);
		if(regex_search(name,pattern)){
			index++;
			std::cerr << index<<" layer: "<<name << std::endl;
			ending_tasks.insert(last_node_name);
			last_node_name=name;
			return true;
		}
		else{
			std::cerr <<index<< " skipping layer: "<<name << std::endl;
			last_node_name=name;
			return false;
		}

		//Method 2
		if (name==""){
					std::cerr << "Skipping layer: "<<name << std::endl;
					return false;
				}
		// patterns that are skipped:
		std::vector<std::string> formats={
				".*_g\\d*",	//all names with _g then a digit like _g2 (for group convs)
				".*relu.*",	//all names with relu because in python models relu layers all not separate layers
				".*linear.*",	//some activation layers in yolo are linear
				".*batchnorm.*"	//names that have batchnorm
		};
		std::vector<std::regex> patterns;
		for(auto format:formats){
			patterns.push_back(std::regex(format, std::regex_constants::icase));
		}
		bool skip = false;
		for(auto pattern:patterns){
			skip=skip || (regex_search(name,pattern));
		}
		if (skip){
    		std::cerr << "Skipping layer: "<<name << std::endl;
    		return false;
		}



		// patterns that are accepted:
		std::vector<std::string> accepted_formats={
				".*conv.*",	//all names with conv (conv/leaky and conv/linear layers already skipped
				".*fc.*",	//all fully connected layers
		};
		std::vector<std::regex> accepted_patterns;
		for(auto format:accepted_formats){
			accepted_patterns.push_back(std::regex(format, std::regex_constants::icase));
		}
		bool accept = false;
		for(auto pattern:accepted_patterns){
			accept=accept || (regex_search(name,pattern));
		}
		if (accept){
			std::cerr << "layer: "<<name << std::endl;
			return true;
		}
		else{
			std::cerr << "Skipping layer: "<<name << std::endl;
			return false;
		}


		//Method 3:
		if (name==""){
					std::cerr << "Skipping layer: "<<name << std::endl;
					return false;
				}
    	/* pattern that are considered as start of a layer
    	 * .*conv.* all names with conv
    	 * .*fc.* all names with fc
    	 * ^$ all names wit empty string (for input layer) //for now we did not count it separately becaue make wrong output (test with alexnet ./Run_CO-UP model=Alex --order=BBGGBBBBL push=1 compile=1)
    	 */
    	//std::string formatPattern_conv = ".*conv.*|.*fc.*|^$";
    	/*std::string formatPattern_conv = ".*conv.*|.*fc.*";
    	std::regex pattern_conv(formatPattern_conv, std::regex_constants::icase);

    	if (regex_search(name, pattern)) {
    		std::cerr << "Skipping layer: "<<name << std::endl;
    		return false;
    	}
    	else{
    		if (regex_search(name, pattern_conv)){
    			std::cerr<<"layer name: "<<name<<std::endl;
    			return true;
    		}
    		std::cerr << "2-Skipping layer: "<<name << std::endl;
    		return false;
    	}*/

    	/*
    	 * or you can just write this:
    	std::string formatPattern = "^(?!.*_g\\d*)(?!.*relu).*conv.*";
		std::regex pattern(formatPattern, std::regex_constants::icase);
    	 */

    }
#endif
