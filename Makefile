

# Define the compiler
CXX := g++ 

# Define the source and target files
SRC := C_call_CO_UP.cpp
TARGET := CO-UP.out

# Define the build rule for the target
$(TARGET): $(SRC)
	$(CXX) $(SRC) -o $(TARGET)
	chmod +x $(TARGET)

# Define the .PHONY rule to declare 'clean' as a non-file target
.PHONY: clean

# Define the clean rule to remove the target file
clean:
	rm -f $(TARGET)

