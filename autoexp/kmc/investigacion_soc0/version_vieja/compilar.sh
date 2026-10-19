g++ -std=c++11 -O3 -march=native -mtune=native -fopenmp -pthread -funroll-loops -ffast-math -flto -DTIMING -mavx2 -mfma -o ./video/kmc_prueba KMC-Galvanostatic-interactions-terminal.cpp -fopenmp -lm
