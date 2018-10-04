#ROSVER = indigo
ROSVER = groovy
CC = g++
CXX = g++
CXXFLAGS = -Iinclude -I/opt/ros/$(ROSVER)/include

LDFLAGS = -L/opt/ros/$(ROSVER)/lib -lrosconsole -lroscpp_serialization 
LDFLAGS +=  -lroscpp -ltf -lrostime

all: getObstacles getMonkeys

getObstacles: getObstacles.o
	$(CXX) $^ -o $@ $(LDFLAGS)

getObstacles.o: getObstacles.cc
	$(CXX) $(CXXFLAGS) -o $@ -c $< 

getMonkeys: getMonkeys.o
	$(CXX) $^ -o $@ $(LDFLAGS)

getMonkeys.o: getMonkeys.cc
	$(CXX) $(CXXFLAGS) -o $@ -c $< 

clean:
	rm *.o *~
