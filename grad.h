#ifndef GRAD_H
#define GRAD_H


#include "qpu.h"

#define NUM_QPUS        12
#define MAX_CODE_SIZE   4096
#define NUM_UNIS        17
#define MEM_PARAMS      27
#define WIDTH           360
#define HEIGHT          240
#define WINDOW_DIM      23
#define GRAD_DIM        (WINDOW_DIM-2)
#define GRAD_ELMTS      GRAD_DIM*GRAD_DIM
#define NUM_FEATURES    30
#define SIZE            8192*8192*2



/*###########################*/
/*###########################*/
/*                           */
/*__GRAD__GLOBAL__VARIABLES__*/
/*                           */
/*###########################*/
/*###########################*/
static unsigned int qpu_code[MAX_CODE_SIZE];
static unsigned int grad_VCptrsArray[MEM_PARAMS];


static int mb, code_words;
static unsigned vc_ptr, handle;
static GPU_FFT_HOST host;
static void *arm_ptr;
static struct grad_MemLayout *arm_map;
/*################################*/
/*################################*/
/*                                */
/*__END__GRAD__GLOBAL__VARIABLES__*/
/*                                */
/*################################*/
/*################################*/



/*########################*/
/*########################*/
/*                        */
/*__GRAD__MEMORY__LAYOUT__*/
/*                        */
/*########################*/
/*########################*/
struct grad_MemLayout
{

    unsigned int gradXcode[MAX_CODE_SIZE];
    unsigned int gradYcode[MAX_CODE_SIZE];
    unsigned int gradXYcode[MAX_CODE_SIZE];
    unsigned int xtrBIP2code[MAX_CODE_SIZE];
    unsigned int computeBIP2code[MAX_CODE_SIZE];
    unsigned int dPosXYcode[MAX_CODE_SIZE];
    unsigned int convX3code[MAX_CODE_SIZE];
    unsigned int convX6code[MAX_CODE_SIZE];
    unsigned int xtrX4code[MAX_CODE_SIZE];


    unsigned int uniforms[NUM_QPUS][NUM_UNIS];  /*14 uniforms per QPU :
			                        --1st is the address of the FIRST FRAME data
			                        --2nd is the address of the FEATUREs array
			                        --3rd is the address of the GRADX result array
			                        --4th is the address of the GXX result array
			                        --5rd is the address of the GRADY result array
			                        --6th is the address of the GYY result array
			                        --7th is the address of the GXY result array
			                        --8th is the QPU NUMBER
			                        --9th is the address of the SECOND FRAME data
			                        --10th is the address of the xtrBIP2 result array
			                        --11th is the address of the dPOSXY result array
			                        --12th is the address of the computeBIP2 result array
			                        --13th is the address of the detXY result array
			                        --14th is the address of the frameX3 result array
			                        --15th is the address of the frameX6 result array
			                        --16th is the address of the firstFrameX4 result array
			                        --17th is the address of the secondFrameX4 result array*/


    unsigned int msg[NUM_QPUS][2];


    float firstFrameShared[WIDTH*HEIGHT];                 // FIRST FRAME for the QPUs to read into
    float secondFrameShared[WIDTH*HEIGHT];                // SECOND FRAME for the QPUs to read into
    unsigned int featureXYshared[NUM_FEATURES*2];         // array containing each FEATURE (x/y) position



    /*__GRADX__*/
    float gradXshared[NUM_FEATURES*GRAD_ELMTS];           // result array containing 30*441 GRADX values
    float gXXshared[NUM_FEATURES];                        // result array containing 30*GXX values
    /*__END__GRADX__*/


    /*__GRADY__*/
    float gradYshared[NUM_FEATURES*GRAD_ELMTS];           // result array containing 30*441 GRADY values
    float gYYshared[NUM_FEATURES];                        // result array containing 30*GYY values
    /*__END__GRADY__*/


    /*__GRADXY__*/
    float gXYshared[NUM_FEATURES];                        // result array containing 30*GXY values
    /*__END__GRADXY__*/


    /*__BIP2__*/
    float xtrBIP2shared[NUM_FEATURES*GRAD_ELMTS*4];       // result array containing 4*30*441 eXTRacted weighted values from FRAME 2
    float computeBIP2shared[NUM_FEATURES*GRAD_ELMTS];     // result array containing BILINEAR INTERPOLATION values from FRAME 2
    /*__END__BIP2__*/


    /*__dPOSXY__*/
    float dPosXYshared[NUM_FEATURES*2];                   // result array containing 30*2 dPosXY deplacement values for each FEATURE
    /*__END__dPOSXY__*/


    /*__detXY__*/
    float detXYshared[NUM_FEATURES];                      // result array containing 30 DET values for each FEATURE
    /*__END__detXY__*/


    /*__frameX3__*/
    float frameX3shared[(WIDTH-2)*(HEIGHT-2)];            // result array containing firstFrameX3/secondFrameX3 values from first convolution
    /*__END__frameX3__*/


    /*__frameX6__*/
    float frameX6shared[(WIDTH-4)*(HEIGHT-4)];            // result array containing firstFrameX6/secondFrameX6 values from second convolution
    /*__END__frameX6__*/


    /*__firstFrameX4__*/
    float firstFrameX4shared[(WIDTH-4)*(HEIGHT-4)/4];     // result array containing firstFrameX4/secondFrameX4  values from frameX6 extraction
    /*__END__firstFrameX4*__*/


    /*__secondFrameX4__*/
    float secondFrameX4shared[(WIDTH-4)*(HEIGHT-4)/4];    // result array containing firstFrameX4/secondFrameX4  values from frameX6 extraction
    /*__END__secondFrameX4*__*/
};
/*#############################*/
/*#############################*/
/*                             */
/*__END__GRAD__MEMORY__LAYOUT__*/
/*                             */
/*#############################*/
/*#############################*/



/*##############################################*/
/*##############################################*/
/*                                              */
/*__FUNCTIONS__INDEPENDENT__OF__MEMORY__LAYOUT__*/
/*                                              */
/*##############################################*/
/*##############################################*/
int loadShaderCode(const char *fname, unsigned int* buffer, int len);
int getVCptr(void);
void freeMemory(void);
/*###################################################*/
/*###################################################*/
/*                                                   */
/*__END__FUNCTIONS__INDEPENDENT__OF__MEMORY__LAYOUT__*/
/*                                                   */
/*###################################################*/
/*###################################################*/




/*############################################*/
/*############################################*/
/*                                            */
/*__FUNCTIONS__DEPENDENT__OF__MEMORY__LAYOUT__*/
/*                                            */
/*############################################*/
/*############################################*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*__INIT_&_COPY__FUNCTIONS__*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~*/
// initialize VC/CPU SHARED memory
void grad_setVCptrs(void);
void grad_init(void);

// pull INPUT FRAME data into SHARED memory
void grad_firstFrameSharedCpy(float *firstFrameMain);
void grad_secondFrameSharedCpy(float *secondFrameMain);
void grad_featureXYsharedCpy(unsigned int *featureXYmain);
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*__END__INIT_&_COPY__FUNCTIONS__*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/



/*~~~~~~~~~~~~~~~~~~~~*/
/*__GRADX__FUNCTIONS__*/
/*~~~~~~~~~~~~~~~~~~~~*/
// invoke qpu to execute GRADX code
void grad_gradXqpu(int toggle);

// display GRADX results from SHARED memory
void grad_gradXsharedDisplay(void);
void grad_gXXsharedDisplay(void);

// store GRADX results into cpu RAM memory
void grad_gradXsharedCpy(float *gradXmain);
void grad_gXXsharedCpy(float *gXXmain);
/*~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*__END__GRADX__FUNCTIONS__*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~*/



/*~~~~~~~~~~~~~~~~~~~~*/
/*__GRADY__FUNCTIONS__*/
/*~~~~~~~~~~~~~~~~~~~~*/
// invoke qpu to execute GRADY code
void grad_gradYqpu(int toggle);

// display GRADY results from SHARED memory
void grad_gradYsharedDisplay(void);
void grad_gYYsharedDisplay(void);

// store GRADY results into cpu RAM memory
void grad_gradYsharedCpy(float *gradYmain);
void grad_gYYsharedCpy(float *gYYmain);
/*~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*__END__GRADY__FUNCTIONS__*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~*/



/*~~~~~~~~~~~~~~~~~~~~~*/
/*__GRADXY__FUNCTIONS__*/
/*~~~~~~~~~~~~~~~~~~~~~*/
// invoke qpu to execute GRADXY code
void grad_gradXYqpu(void);

// display GRADXY results from SHARED memory
void grad_gXYsharedDisplay(void);

// store GRADXY results into cpu RAM memory
void grad_gXYsharedCpy(float *gXYmain);
/*~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*__END__GRADXY__FUNCTIONS__*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~*/



/*~~~~~~~~~~~~~~~~~~~~~~*/
/*__xtrBIP2__FUNCTIONS__*/
/*~~~~~~~~~~~~~~~~~~~~~~*/
// invoke qpu to execute xtrBIP2 code
void grad_xtrBIP2qpu(int toggle);

// display xtrBIP2 results from SHARED memory
void grad_xtrBIP2sharedDisplay(void);
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*__END__xtrBIP2__FUNCTIONS__*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~*/



/*~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*__computeBIP2__FUNCTIONS__*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~*/
// invoke qpu to execute computeBIP2 code
void grad_computeBIP2qpu(int toggle);

// display computeBIP2 results from SHARED memory
void grad_computeBIP2sharedDisplay(void);
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*__END__computeBIP2__FUNCTIONS__*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/



/*~~~~~~~~~~~~~~~~~~~~~*/
/*__dPOSXY__FUNCTIONS__*/
/*~~~~~~~~~~~~~~~~~~~~~*/
// initialize dPosXYshared array
void grad_dPosXYsharedInit(void);

// invoke qpu to execute dPOSXY code
void grad_dPosXYqpu(int toggle);

// display dPOSXY results from SHARED memory
void grad_dPosXYsharedDisplay(void);

// store dPOSXY results into cpu RAM memory
void grad_dPosXYsharedCpy(float *dPosXYmain);
/*~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*__END__dPOSXY__FUNCTIONS__*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~*/



/*~~~~~~~~~~~~~~~~~~~~*/
/*__detXY__FUNCTIONS__*/
/*~~~~~~~~~~~~~~~~~~~~*/
// compute detXYarray
void grad_detXYsharedCompute(void);

// display detXY results from SHARED memory
void grad_detXYsharedDisplay(void);

// store dPOSXY results into cpu RAM memory
void grad_detXYsharedCpy(float *detXYmain);
/*~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*__END__detXY__FUNCTIONS__*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~*/



/*~~~~~~~~~~~~~~~~~~~~~~*/
/*__frameX3__FUNCTIONS__*/
/*~~~~~~~~~~~~~~~~~~~~~~*/
// compute frameX3array
void grad_frameX3sharedCheck(void);

// display frameX3 results from SHARED memory
void grad_frameX3sharedDisplay(void);
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*__END__frameX3__FUNCTIONS__*/


/*#################################################*/
/*#################################################*/
/*                                                 */
/*__END__FUNCTIONS__DEPENDENT__OF__MEMORY__LAYOUT__*/
/*                                                 */
/*#################################################*/
/*#################################################*/


#endif
