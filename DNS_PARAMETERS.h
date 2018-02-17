// Define constants for CUDA
#define TX 8
#define TY 8
#define TZ 8

// Define constants for problem
#define NX 256
#define NY 256
#define NZ 256
#define NZ2 (NZ/2 + 1)
#define NN (NX*NY*NZ)
#define PI (M_PI)
#define LX (2*PI)
#define LY (2*PI)
#define LZ (2*PI)
#define dx (LX/NX)
#define n_save 20		// Number of steps to take between saving data
#define dt .000817653	// Timestep
#define nt 3000		// Total number of timesteps to take in the simulation
#define Re 100
#define Sc 0.7
#define k_max (2.0/3.0*(double)NX/2.0)			// De-alias using the 2/3 truncation rule
// #define k_max ( 15.0/32.0*(double)NX )		// De-alias using 15/32 truncation (from Weirong's thesis)
#define location "/home/bblakeley/Documents/Research/DNS_Data/Isotropic/multiGPU/R2/%c.%i"
#define DataLocation "/home/bblakeley/Documents/Research/DNS_Data/Flamelet_Data/R2/%s.0"

/*
// Define constants for problem
#define NX 512
#define NY 512
#define NZ 512
#define NZ2 (NZ/2 + 1)
#define NN (NX*NY*NZ)
#define PI (M_PI)
#define LX (2*PI)
#define LY (2*PI)
#define LZ (2*PI)
#define dx (LX/NX)
#define n_save 260		// Number of steps to take between saving data
#define dt .0004717653	// Timestep
#define nt 4940		// Total number of timesteps to take in the simulation
#define Re 400
#define Sc 0.7
#define k_max (2.0/3.0*(double)NX/2.0)			// De-alias using the 2/3 truncation rule
// #define k_max ( 15.0/32.0*(double)NX )		// De-alias using 15/32 truncation (from Weirong's thesis)
#define location "/home/bblakeley/Documents/Research/DNS_Data/Isotropic/Test/R4_cuda_customworksize/%c.%i"
#define DataLocation "/home/bblakeley/Documents/Research/DNS_Data/Flamelet_Data/R4/%s.0"
*/
/*
// Define constants for problem
#define NX 1024
#define NY 1024
#define NZ 1024
#define NZ2 (NZ/2 + 1)
#define NN (NX*NY*NZ)
#define PI (M_PI)
#define LX (2*PI)
#define LY (2*PI)
#define LZ (2*PI)
#define dx (LX/NX)
#define n_save 260		// Number of steps to take between saving data
#define dt .0002924483	// Timestep
#define nt 520		// Total number of timesteps to take in the simulation
// #define nt 6760		// Total number of timesteps to take in the simulation
#define Re 1600
#define Sc 0.7
// #define k_max (2.0/3.0*(double)NX/2.0)			// De-alias using the 2/3 truncation rule
#define k_max ( 15.0/32.0*(double)NX )		// De-alias using 15/32 truncation (from Weirong's thesis)
#define location "/home/bblakeley/Documents/Research/DNS_Data/Isotropic/Test/R4_cuda_customworksize/%c.%i"
#define DataLocation "/home/bblakeley/Documents/Research/DNS_Data/Flamelet_Data/R4/%s.0"
*/
/*
// Define constants for problem
#define NX 128
#define NY 128
#define NZ 128
#define NZ2 (NZ/2 + 1)
#define NN (NX*NY*NZ)
#define LX (2*M_PI)
#define LY (2*M_PI)
#define LZ (2*M_PI)
#define PI ((double)M_PI)
#define n_save 20		// Number of steps to take between saving data
#define dt 0.01 	// Timestep
#define nt 1000		// Total number of timesteps to take in the simulation
#define Re 100
#define Sc 0.7
#define k_max ( 2.0/3.0*(double)NX/2.0 )	// De-alias using the 2/3rd truncation rule
// #define k_max ( 15.0/32.0*(double)NX )		// De-alias using 15/32 truncation (from Weirong's thesis)
#define location "/home/bblakeley/Documents/Research/DNS_Data/Taylor_Green/CUDA/mGPU_Test/N%i_Re%i_Sc07_dt01/%c.%i"
#define DataLocation "/home/bblakeley/Documents/Research/DNS_Data/Initialization/TG_256/%s.0"
*/
