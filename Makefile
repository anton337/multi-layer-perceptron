all:
	g++ -g0 -O3 main.cpp -lglut -lGL -lGLU -lboost_thread -lboost_system;#./a.out
 
debug:
	g++ -g3 -O0 main.cpp -lglut -lGL -lGLU -lboost_thread -lboost_system;#gdb ./a.out

