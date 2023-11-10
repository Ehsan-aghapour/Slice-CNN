# with mode you select the check function comes from main_layer_checker.h in which dir for example:
#make mode=1_Simple/

# Compiler to use
CXX = g++ 
# Compiler flags
CXXFLAGS = -Wall -std=c++17 -I$(mode)
# Name of the output executable
TARGET = myprogram
# Source files
SOURCES = main.cpp
# Header files
HEADERS = tasks.h $(mode)/main_layer_checker.h
 
# Object files
OBJECTS = $(SOURCES:.cpp=.o)
# Default target
all: $(TARGET)

# Link the target executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET)
	chmod +x $(TARGET)
	./$(TARGET)

run:
	./$(TARGET)

# Compile the source files into object files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean target for removing compiled files
clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean