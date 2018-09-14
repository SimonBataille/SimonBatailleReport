#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <sys/time.h>
#include <time.h>

#include "mailbox.h"
#include "grad.h"



/*################################################*/
/*################################################*/
/*                                                */
/*__FUNCTIONS__INDEPENDENT__FROM__MEMORY__LAYOUT__*/
/*                                                */
/*################################################*/
/*################################################*/

/*~~~~~~~~~~~~~~~~~~*/
/* loadShaderCode() */
/*~~~~~~~~~~~~~~~~~~*/
int loadShaderCode(const char *fname, unsigned int *buffer, int len)
{
    FILE *in = fopen(fname, "r");
    if (!in)
    {
        fprintf(stderr, "Failed to open %s.\n", fname);
        exit(0);
    }

    size_t items = fread(buffer, sizeof(unsigned int), len, in);
    fclose(in);

    return items;
}



/*~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*        getVCptr()         */
/* using mailbox's functions */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
int getVCptr()
{

    mb = mbox_open();

    if (gpu_fft_get_host_info(&host))
    {
        fprintf(stderr, "QPU fetch of host information (Rpi version, etc.) failed.\n");
        return -5;
    }

    if (qpu_enable(mb, 1))
    {
        fprintf(stderr, "QPU enable failed.\n");
        return -1;
    }
    printf("QPU enabled.\n");


    handle = mem_alloc(mb, SIZE, 4096, host.mem_flg);
    if (!handle)
    {
        fprintf(stderr, "Unable to allocate %d bytes of GPU memory", SIZE);
        exit(0);
        return -2;
    }


    /* Lock vc_ptr inside VC/CPU BUS Addresses */
    vc_ptr = mem_lock(mb, handle);

    return 0;
}



/*~~~~~~~~~~~~*/
/* freeMemory */
/*~~~~~~~~~~~~*/
void freeMemory()
{

    printf("Cleaning up.\n");
    unmapmem(arm_ptr, SIZE);
    mem_unlock(mb, handle);
    mem_free(mb, handle);
    qpu_enable(mb, 0);
    printf("Done.\n");
}

/*#####################################################*/
/*#####################################################*/
/*                                                     */
/*__END__FUNCTIONS__INDEPENDENT__FROM__MEMORY__LAYOUT__*/
/*                                                     */
/*#####################################################*/
/*#####################################################*/



/*##############################################*/
/*##############################################*/
/*                                              */
/*__FUNCTIONS__DEPENDENT__FROM__MEMORY__LAYOUT__*/
/*                                              */
/*##############################################*/
/*##############################################*/

/*##########################*/
/*__INIT_&_COPY__FUNCTIONS__*/
/*##########################*/

/*~~~~~~~~~~~~~~~~~~*/
/* grad_setVCptrs() */
/*~~~~~~~~~~~~~~~~~~*/
void grad_setVCptrs()
{
    grad_VCptrsArray[0] = vc_ptr + offsetof(struct grad_MemLayout,
                                            gradXcode); //vc_gradXcode
    grad_VCptrsArray[1] = vc_ptr + offsetof(struct grad_MemLayout,
                                            gradYcode); //vc_gradYcode
    grad_VCptrsArray[2] = vc_ptr + offsetof(struct grad_MemLayout,
                                            gradXYcode); //vc_gradXYcode
    grad_VCptrsArray[3] = vc_ptr + offsetof(struct grad_MemLayout,
                                            xtrBIP2code); //vc_xtrBIP2code
    grad_VCptrsArray[4] = vc_ptr + offsetof(struct grad_MemLayout,
                                            computeBIP2code); //vc_computeBIP2code
    grad_VCptrsArray[5] = vc_ptr + offsetof(struct grad_MemLayout,
                                            dPosXYcode); //vc_dPosXYcode
    grad_VCptrsArray[6] = vc_ptr + offsetof(struct grad_MemLayout,
                                            convX3code); //vc_convX3code
    grad_VCptrsArray[7] = vc_ptr + offsetof(struct grad_MemLayout,
                                            convX6code); //vc_convX6code
    grad_VCptrsArray[8] = vc_ptr + offsetof(struct grad_MemLayout,
                                            xtrX4code); //vc_xtrX4code


    grad_VCptrsArray[9] = vc_ptr + offsetof(struct grad_MemLayout,
                                            uniforms); //vc_uniforms


    grad_VCptrsArray[10] = vc_ptr + offsetof(struct grad_MemLayout, msg); //vc_msg


    grad_VCptrsArray[11] = vc_ptr + offsetof(struct grad_MemLayout,
                                            firstFrameShared); //vc_firstFrameShared
    grad_VCptrsArray[12] = vc_ptr + offsetof(struct grad_MemLayout,
                                            secondFrameShared); //vc_secondFrameShared
    grad_VCptrsArray[13] = vc_ptr + offsetof(struct grad_MemLayout,
                           featureXYshared); //vc_featureXYshared


    grad_VCptrsArray[14] = vc_ptr + offsetof(struct grad_MemLayout,
                           gradXshared); //vc_gradXshared
    grad_VCptrsArray[15] = vc_ptr + offsetof(struct grad_MemLayout,
                           gXXshared); //vc_gXXshared


    grad_VCptrsArray[16] = vc_ptr + offsetof(struct grad_MemLayout,
                           gradYshared); //vc_gradYshared
    grad_VCptrsArray[17] = vc_ptr + offsetof(struct grad_MemLayout,
                           gYYshared); //vc_gYYshared


    grad_VCptrsArray[18] = vc_ptr + offsetof(struct grad_MemLayout,
                           gXYshared); //vc_gXYshared


    grad_VCptrsArray[19] = vc_ptr + offsetof(struct grad_MemLayout,
                           xtrBIP2shared); //vc_xtrBIP2shared
    grad_VCptrsArray[20] = vc_ptr + offsetof(struct grad_MemLayout,
                           computeBIP2shared); //vc_computeBIP2shared


    grad_VCptrsArray[21] = vc_ptr + offsetof(struct grad_MemLayout,
                           dPosXYshared); //vc_dPosXYshared


    grad_VCptrsArray[22] = vc_ptr + offsetof(struct grad_MemLayout,
                           detXYshared); //vc_detXYshared


    grad_VCptrsArray[23] = vc_ptr + offsetof(struct grad_MemLayout,
                           frameX3shared); //vc_frameX3shared


    grad_VCptrsArray[24] = vc_ptr + offsetof(struct grad_MemLayout,
                           frameX6shared); //vc_frameX6shared


    grad_VCptrsArray[25] = vc_ptr + offsetof(struct grad_MemLayout,
                           firstFrameX4shared); //vc_firstFrameX4shared


    grad_VCptrsArray[26] = vc_ptr + offsetof(struct grad_MemLayout,
                           secondFrameX4shared); //vc_secondFrameX4shared
}


/*~~~~~~~~~~~~~*/
/* grad_init() */
/*~~~~~~~~~~~~~*/
void grad_init()
{

    /* Get vc_ptr inside VC/CPU BUS Addresses */
    getVCptr();


    /* Assert vc_ptr inside VC/CPU BUS Addresses */
    grad_setVCptrs();


    /* Get "arm_ptr" pointer inside ARM VIRTUAL Addresses */
    arm_ptr = mapmem(BUS_TO_PHYS(vc_ptr + host.mem_map), SIZE);

    arm_map = (struct grad_MemLayout *)arm_ptr;
    memset(arm_map, 0x0, sizeof(struct grad_MemLayout));



    /*~~~~~~~~~~~~~~~~~~*/
    /*__GET__GRADXCODE__*/
    /*~~~~~~~~~~~~~~~~~~*/
    // Load gradXcode into ARM VIRTUAL Addresses
    code_words = loadShaderCode("../src/gradX.bin", qpu_code, MAX_CODE_SIZE);

    if (code_words == 0)
    {
        exit(0);
    }
    else
    {
        printf("Loaded %d bytes of gradXcode from %s ...\n",
               code_words * sizeof(unsigned), "gradX.bin");
    }

    // Copy gradXcode into SHARED Memory
    memcpy(arm_map->gradXcode, qpu_code, code_words * sizeof(unsigned int));
    /*~~~~~~~~~~~~~~~~~~~~~~~*/
    /*__END__GET__GRADXCODE__*/
    /*~~~~~~~~~~~~~~~~~~~~~~~*/



    /*~~~~~~~~~~~~~~~~~~*/
    /*__GET__GRADYCODE__*/
    /*~~~~~~~~~~~~~~~~~~*/
    // Load gradYcode into ARM VIRTUAL Addresses
    code_words = loadShaderCode("../src/gradY.bin", qpu_code, MAX_CODE_SIZE);

    if (code_words == 0)
    {
        exit(0);
    }
    else
    {
        printf("Loaded %d bytes of gradYcode from %s ...\n",
               code_words * sizeof(unsigned), "gradY.bin");
    }

    // Copy gradYcode into SHARED Memory
    memcpy(arm_map->gradYcode, qpu_code, code_words * sizeof(unsigned int));
    /*~~~~~~~~~~~~~~~~~~~~~~~*/
    /*__END__GET__GRADYCODE__*/
    /*~~~~~~~~~~~~~~~~~~~~~~~*/



    /*~~~~~~~~~~~~~~~~~~~*/
    /*__GET__GRADXYCODE__*/
    /*~~~~~~~~~~~~~~~~~~~*/
    // Load gradXYcode into ARM VIRTUAL Addresses
    code_words = loadShaderCode("../src/gradXY.bin", qpu_code, MAX_CODE_SIZE);

    if (code_words == 0)
    {
        exit(0);
    }
    else
    {
        printf("Loaded %d bytes of gradXYcode from %s ...\n",
               code_words * sizeof(unsigned), "gradXY.bin");
    }

    // Copy gradXYcode into SHARED Memory
    memcpy(arm_map->gradXYcode, qpu_code, code_words * sizeof(unsigned int));
    /*~~~~~~~~~~~~~~~~~~~~~~~~*/
    /*__END__GET__GRADXYCODE__*/
    /*~~~~~~~~~~~~~~~~~~~~~~~~*/



    /*~~~~~~~~~~~~~~~~~~~~*/
    /*__GET__xtrBIP2CODE__*/
    /*~~~~~~~~~~~~~~~~~~~~*/
    // Load xtrBIP2code into ARM VIRTUAL Addresses
    code_words = loadShaderCode("../src/xtrBIP2.bin", qpu_code, MAX_CODE_SIZE);

    if (code_words == 0)
    {
        exit(0);
    }
    else
    {
        printf("Loaded %d bytes of xtrBIP2code from %s ...\n",
               code_words * sizeof(unsigned), "xtrBIP2.bin");
    }

    // Copy xtrBIP2code into SHARED Memory
    memcpy(arm_map->xtrBIP2code, qpu_code, code_words * sizeof(unsigned int));
    /*~~~~~~~~~~~~~~~~~~~~~~~~~*/
    /*__END__GET__xtrBIP2CODE__*/
    /*~~~~~~~~~~~~~~~~~~~~~~~~~*/



    /*~~~~~~~~~~~~~~~~~~~~~~~~*/
    /*__GET__computeBIP2CODE__*/
    /*~~~~~~~~~~~~~~~~~~~~~~~~*/
    // Load computeBIP2code into ARM VIRTUAL Addresses
    code_words = loadShaderCode("../src/computeBIP2.bin", qpu_code, MAX_CODE_SIZE);

    if (code_words == 0)
    {
        exit(0);
    }
    else
    {
        printf("Loaded %d bytes of computeBIP2code from %s ...\n",
               code_words * sizeof(unsigned), "computeBIP2.bin");
    }

    // Copy computeBIP2code into SHARED Memory
    memcpy(arm_map->computeBIP2code, qpu_code, code_words * sizeof(unsigned int));
    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    /*__END__GET__computeBIP2CODE__*/
    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/



    /*~~~~~~~~~~~~~~~~~~~*/
    /*__GET__dPosXYCODE__*/
    /*~~~~~~~~~~~~~~~~~~~*/
    // Load dPosXYcode into ARM VIRTUAL Addresses
    code_words = loadShaderCode("../src/dPosXY.bin", qpu_code, MAX_CODE_SIZE);

    if (code_words == 0)
    {
        exit(0);
    }
    else
    {
        printf("Loaded %d bytes of dPosXYcode from %s ...\n",
               code_words * sizeof(unsigned), "dPosXY.bin");
    }

    // Copy dPosXYcode into SHARED Memory
    memcpy(arm_map->dPosXYcode, qpu_code, code_words * sizeof(unsigned int));
    /*~~~~~~~~~~~~~~~~~~~~~~~~*/
    /*__END__GET__dPosXYCODE__*/
    /*~~~~~~~~~~~~~~~~~~~~~~~~*/



    /* Copy UNIFORMS & first MSG into SHARED Memory */
    for (int i = 0; i < NUM_QPUS; i++)
    {

        arm_map->uniforms[i][0] = grad_VCptrsArray[11]; //vc_firstFrameShared
        arm_map->uniforms[i][1] = grad_VCptrsArray[13]; //vc_featureXYshared
        arm_map->uniforms[i][2] = grad_VCptrsArray[14]; //vc_gradXshared
        arm_map->uniforms[i][3] = grad_VCptrsArray[15]; //vc_gXXshared
        arm_map->uniforms[i][4] = grad_VCptrsArray[16]; //vc_gradYshared
        arm_map->uniforms[i][5] = grad_VCptrsArray[17]; //vc_gYYshared
        arm_map->uniforms[i][6] = grad_VCptrsArray[18]; //vc_gXYshared
        arm_map->uniforms[i][7] = i; //QPU number
        arm_map->uniforms[i][8] = grad_VCptrsArray[12]; //vc_secondFrameShared
        arm_map->uniforms[i][9] = grad_VCptrsArray[19]; //vc_xtrBIP2shared
        arm_map->uniforms[i][10] = grad_VCptrsArray[21]; //vc_dPosXYshared
        arm_map->uniforms[i][11] = grad_VCptrsArray[20]; //vc_computeBIP2shared
        arm_map->uniforms[i][12] = grad_VCptrsArray[22]; //vc_detXYshared
        arm_map->uniforms[i][13] = grad_VCptrsArray[23]; //vc_frameX3shared
        arm_map->uniforms[i][14] = grad_VCptrsArray[24]; //vc_frameX6shared
        arm_map->uniforms[i][15] = grad_VCptrsArray[25]; //vc_firstFrameX4shared
        arm_map->uniforms[i][16] = grad_VCptrsArray[26]; //vc_secondFrameX4shared

        arm_map->msg[i][0] = grad_VCptrsArray[9] + i * sizeof(unsigned) *
                             NUM_UNIS; //vc_uniforms + i*NUM_UNIS
        //arm_map->msg[i][1] = grad_VCptrsArray[0]; //vc_gradXcode
    }
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/* grad_firstFrameSharedCpy() */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void grad_firstFrameSharedCpy(float *firstFrameMain)
{

    /* Copy INPUT image into shared RAM memory */
    memcpy(arm_map->firstFrameShared, firstFrameMain,
           sizeof(float) * WIDTH * HEIGHT);

    /*__DEBUG__*/
    /* Check pointers values */
    //printf("\nDRIVER memcpy firstFrameMain : &arm_map->firstFrameShared = %p ----- &firstFrameMain = %p\n", arm_map->firstFrameShared, firstFrameMain);


    /* Check firstFrameShared values */
    /*for (int i = 0; i < WIDTH*HEIGHT; i++){
        printf("firstFrameShared[%02u] = %06f\n", i, *(arm_map->firstFrameShared + i));
    }*/
    /*__END__DEBUG__*/
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/* grad_secondFrameSharedCpy() */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void grad_secondFrameSharedCpy(float *secondFrameMain)
{

    /* Copy INPUT image into shared RAM memory */
    memcpy(arm_map->secondFrameShared, secondFrameMain,
           sizeof(float) * WIDTH * HEIGHT);

    /*__DEBUG__*/
    /* Check pointers values */
    //printf("\nDRIVER memcpy secondFrameMain : &arm_map->secondFrameShared = %p ----- &secondFrameMain = %p\n", arm_map->secondFrameShared, secondFrameMain);


    /* Check secondFrameShared values */
    /*for (int i = 0; i < WIDTH*HEIGHT; i++){
        printf("secondFrameShared[%02u] = %06f\n", i, *(arm_map->secondFrameShared + i));
    }*/
    /*__END__DEBUG__*/
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~*/
/* grad_featureXYsharedCpy() */
/*~~~~~~~~~~~~~~~~~~~~~~~~~*/
void grad_featureXYsharedCpy(unsigned int *featureXYmain)
{

    /* Copy FEATUREs array into shared RAM memory */
    memcpy(arm_map->featureXYshared, featureXYmain,
           sizeof(unsigned int) * 2 * NUM_FEATURES);


    /*__DEBUG__*/
    /* Check pointers values */
    //printf("\nDRIVER memcpy featureXYmain : &arm_map->featureXYshared = %p ----- &featureXYmain = %p\n", arm_map->featureXYshared, featureXYmain);


    /* Check featureXYshared values */
    /*for (int i = 0; i < 2*NUM_FEATURES; i++){
        printf("featureXYmain[%02u] = %03u ----- ", i, featureXYmain[i]);
        printf("featureXYshared[%02u] = %03u\n", i, *(arm_map->featureXYshared + i));
    }*/
    /*__END__DEBUG__*/
}

/*###############################*/
/*__END__INIT_&_COPY__FUNCTIONS__*/
/*###############################*/



/*####################*/
/*__GRADX__FUNCTIONS__*/
/*####################*/

/*~~~~~~~~~~~~~~~~~*/
/* grad_gradXqpu() */
/*~~~~~~~~~~~~~~~~~*/
void grad_gradXqpu(int toggle)
{

    // Toggle Frame Address
    if (toggle == 1)
    {
        for (int i = 0; i < NUM_QPUS; i++)
        {
            arm_map->uniforms[i][0] = grad_VCptrsArray[12]; //vc_secondFrameShared
            arm_map->uniforms[i][8] = grad_VCptrsArray[11]; //vc_firstFrameShared
        }
    }
    else if (toggle == 0)
    {
        for (int i = 0; i < NUM_QPUS; i++)
        {
            arm_map->uniforms[i][0] = grad_VCptrsArray[11]; //vc_firstFrameShared
            arm_map->uniforms[i][8] = grad_VCptrsArray[12]; //vc_secondFrameShared
        }
    }


    /* Pass vc_gradXcode pointer to QPUs */
    for (int i = 0; i < NUM_QPUS; i++)
    {
        arm_map->msg[i][1] = grad_VCptrsArray[0]; //vc_gradXcode
    }


    /* Send insructions through mailbox */
    execute_qpu(mb, NUM_QPUS, grad_VCptrsArray[10], GPU_FFT_NO_FLUSH,
                GPU_FFT_TIMEOUT); //vc_msg
}
