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



/*~~~~~~~~~~~~~~~~~~~~~~~~~*/
/* grad_gradXsharedDisplay */
/*~~~~~~~~~~~~~~~~~~~~~~~~~*/
void grad_gradXsharedDisplay(void)
{

    /* Check pointers values */
    printf("\nDRIVER Check arm_map->gradXshared = %p\n", arm_map->gradXshared);

    /* Display gradXshared verbose */
    for (int i = 0; i < NUM_FEATURES * GRAD_ELMTS; i++)
    {
        if ((i % (GRAD_ELMTS)) == 0) printf("\nQPU %u, FEATURE %u\n",
                                                i / (GRAD_ELMTS) % (NUM_QPUS), i / (GRAD_ELMTS));
        if ((i % (GRAD_DIM)) == 0)
        {
            printf("\nRow %u\n", i / (GRAD_DIM) % (GRAD_DIM));
        }
        //printf("DRIVER &*(arm_map->gradXshared + %04d) = &%p = &%04dd ----- ", i, (arm_map->gradXshared + i), (arm_map->gradXshared + i));
        printf(" DRIVER *(arm_map->gradXshared + %04d) = %.8ff\n", i,
               *(arm_map->gradXshared + i));
    }


    /* Display only values */
    /*for (int i=0; i < NUM_FEATURES*GRAD_ELMTS; i++) {
        printf("%f\n", *(arm_map->gradXshared + i));
    }*/
}



/*~~~~~~~~~~~~~~~~~~~~~~~~~*/
/* grad_gradXsharedDisplay */
/*~~~~~~~~~~~~~~~~~~~~~~~~~*/
void grad_gXXsharedDisplay(void)
{

    /* Check pointers values */
    printf("\nDRIVER Check arm_map->gXXshared = %p\n", arm_map->gXXshared);

    /* Display gXXshared verbose */
    for (int i = 0; i < NUM_FEATURES; i++)
    {
        //printf("DRIVER &*(arm_map->gXXshared + %04d) = &%p = &%04dd ----- ", i, (arm_map->gXXshared + i), (arm_map->gXXshared + i));
        printf(" DRIVER *(arm_map->gXXshared + %04d) = %.08ff\n", i,
               *(arm_map->gXXshared + i));
    }


    /* Display only values */
    /*for (int i=0; i < NUM_FEATURES; i++) {
        printf("%f\n", *(arm_map->gXXshared + i));
    }*/
}



/*~~~~~~~~~~~~~~~~~~~~~*/
/* grad_gradXsharedCpy */
/*~~~~~~~~~~~~~~~~~~~~~*/
void grad_gradXsharedCpy(float *gradXmain)
{

    /* Copy gradXshared in non-shared CPU RAM memory */
    memcpy(gradXmain, arm_map->gradXshared,
           NUM_FEATURES * GRAD_ELMTS * sizeof(float));


    /*__DEBUG__*/
    /*for (int i=0; i < NUM_FEATURES*GRAD_ELMTS; i++) {
        printf("DRIVER memcpy *(arm_map->gradXshared + %03d) = 0x%08x = %03dd", i, *(arm_map->gradXshared + i), *(arm_map->gradXshared + i));
        printf("-----DRIVER memcpy *(gradXmain+ %03d) = 0x%08x = %03dd\n", i, *(gradXmain + i), *(gradXmain + i));
    }*/
    /*__END__DEBUG__*/
}



/*~~~~~~~~~~~~~~~~~~~*/
/* grad_gXXsharedCpy */
/*~~~~~~~~~~~~~~~~~~~*/
void grad_gXXsharedCpy(float *gXXmain)
{

    /* Copy gXXshared in non-shared CPU RAM memory */
    memcpy(gXXmain, arm_map->gXXshared, NUM_FEATURES * sizeof(float));


    /*__DEBUG__*/
    /*for (int i=0; i < NUM_FEATURES; i++) {
        printf("DRIVER memcpy *(arm_map->gXXshared + %03d) = 0x%08x = %03dd", i, *(arm_map->gXXshared + i), *(arm_map->gXXshared + i));
        printf("-----DRIVER memcpy *(gXXmain+ %03d) = 0x%08x = %03dd\n", i, *(gXXmain + i), *(gXXmain + i));
    }*/
    /*__DEBUG__*/
}

/*#########################*/
/*__END__GRADX__FUNCTIONS__*/
/*#########################*/



/*####################*/
/*__GRADY__FUNCTIONS__*/
/*####################*/

/*~~~~~~~~~~~~~~~~~*/
/* grad_gradYqpu() */
/*~~~~~~~~~~~~~~~~~*/
void grad_gradYqpu(int toggle)
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

    /* Pass vc_gradYcode pointer to QPUs */
    for (int i = 0; i < NUM_QPUS; i++)
    {
        arm_map->msg[i][1] = grad_VCptrsArray[1]; //vc_gradYcode
    }

    /* Send insructions through mailbox */
    execute_qpu(mb, NUM_QPUS, grad_VCptrsArray[10], GPU_FFT_NO_FLUSH,
                GPU_FFT_TIMEOUT); //vc_msg
}



/*~~~~~~~~~~~~~~~~~~~~~~~~~*/
/* grad_gradYsharedDisplay */
/*~~~~~~~~~~~~~~~~~~~~~~~~~*/
void grad_gradYsharedDisplay(void)
{

    /* Check pointers values */
    printf("\nDRIVER Check arm_map->gradYshared = %p\n", arm_map->gradYshared);

    /* Display gradYshared verbose */
    for (int i = 0; i < NUM_FEATURES * GRAD_ELMTS; i++)
    {
        if ((i % (GRAD_ELMTS)) == 0) printf("\nQPU %u, FEATURE %u\n",
                                                i / (GRAD_ELMTS) % (NUM_QPUS), i / (GRAD_ELMTS));
        if ((i % (GRAD_DIM)) == 0)
        {
            printf("\nRow %u\n", i / (GRAD_DIM) % (GRAD_DIM));
        }
        //printf("DRIVER &*(arm_map->gradYshared + %04d) = &%p = &%04dd ----- ", i, (arm_map->gradYshared + i), (arm_map->gradYshared + i));
        printf(" DRIVER *(arm_map->gradYshared + %04d) = %.8ff\n", i,
               *(arm_map->gradYshared + i));
    }


    /* Display only values */
    /*for (int i=0; i < NUM_FEATURES*GRAD_ELMTS; i++) {
        printf("%f\n", *(arm_map->gradYshared + i));
    }*/
}



/*~~~~~~~~~~~~~~~~~~~~~~~~~*/
/* grad_gradYsharedDisplay */
/*~~~~~~~~~~~~~~~~~~~~~~~~~*/
void grad_gYYsharedDisplay(void)
{

    /* Check pointers values */
    printf("\nDRIVER Check arm_map->gYYshared = %p\n", arm_map->gYYshared);

    /* Display gYYshared verbose */
    for (int i = 0; i < NUM_FEATURES; i++)
    {
        //printf("DRIVER &*(arm_map->gYYshared + %04d) = &%p = &%04dd ----- ", i, (arm_map->gYYshared + i), (arm_map->gYYshared + i));
        printf(" DRIVER *(arm_map->gYYshared + %04d) = %.08ff\n", i,
               *(arm_map->gYYshared + i));
    }


    /* Display only values */
    /*for (int i=0; i < NUM_FEATURES; i++) {
        printf("%f\n", *(arm_map->gYYshared + i));
    }*/
}



/*~~~~~~~~~~~~~~~~~~~~~*/
/* grad_gradYsharedCpy */
/*~~~~~~~~~~~~~~~~~~~~~*/
void grad_gradYsharedCpy(float *gradYmain)
{

    /* Copy gradYshared in non-shared CPU RAM memory */
    memcpy(gradYmain, arm_map->gradYshared,
           NUM_FEATURES * GRAD_ELMTS * sizeof(float));


    /*__DEBUG__*/
    /*for (int i=0; i < NUM_FEATURES*GRAD_ELMTS; i++) {
        printf("DRIVER memcpy *(arm_map->gradYshared + %03d) = 0x%08x = %03dd", i, *(arm_map->gradYshared + i), *(arm_map->gradYshared + i));
        printf("-----DRIVER memcpy *(gradYmain+ %03d) = 0x%08x = %03dd\n", i, *(gradYmain + i), *(gradYmain + i));
    }*/
    /*__END__DEBUG__*/
}



/*~~~~~~~~~~~~~~~~~~~*/
/* grad_gYYsharedCpy */
/*~~~~~~~~~~~~~~~~~~~*/
void grad_gYYsharedCpy(float *gYYmain)
{

    /* Copy gYYshared in non-shared CPU RAM memory */
    memcpy(gYYmain, arm_map->gYYshared, NUM_FEATURES * sizeof(float));


    /*__DEBUG__*/
    /*for (int i=0; i < NUM_FEATURES; i++) {
        printf("DRIVER memcpy *(arm_map->gYYshared + %03d) = 0x%08x = %03dd", i, *(arm_map->gYYshared + i), *(arm_map->gYYshared + i));
        printf("-----DRIVER memcpy *(gYYmain+ %03d) = 0x%08x = %03dd\n", i, *(gYYmain + i), *(gYYmain + i));
    }*/
    /*__DEBUG__*/
}

/*#########################*/
/*__END__GRADY__FUNCTIONS__*/
/*#########################*/



/*#####################*/
/*__GRADXY__FUNCTIONS__*/
/*#####################*/

/*~~~~~~~~~~~~~~~~~~*/
/* grad_gradXYqpu() */
/*~~~~~~~~~~~~~~~~~~*/
void grad_gradXYqpu(void)
{

    /* Pass vc_gradXYcode pointer to QPUs */
    for (int i = 0; i < NUM_QPUS; i++)
    {
        arm_map->msg[i][1] = grad_VCptrsArray[2]; //vc_gradXYcode
    }

    /* Send insructions through mailbox */
    execute_qpu(mb, NUM_QPUS, grad_VCptrsArray[10], GPU_FFT_NO_FLUSH,
                GPU_FFT_TIMEOUT); //vc_msg
}



/*~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/* grad_gradXYsharedDisplay */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void grad_gXYsharedDisplay(void)
{

    /* Check pointers values */
    printf("\nDRIVER Check arm_map->gXYshared = %p\n", arm_map->gXYshared);

    /* Display gXYshared verbose */
    for (int i = 0; i < NUM_FEATURES; i++)
    {
        //printf("DRIVER &*(arm_map->gXYshared + %04d) = &%p = &%04dd ----- ", i, (arm_map->gXYshared + i), (arm_map->gXYshared + i));
        printf(" DRIVER *(arm_map->gXYshared + %04d) = %.08ff\n", i,
               *(arm_map->gXYshared + i));
    }


    /* Display only values */
    /*for (int i=0; i < NUM_FEATURES; i++) {
        printf("%f\n", *(arm_map->gXYshared + i));
    }*/
}




/*~~~~~~~~~~~~~~~~~~~*/
/* grad_gXYsharedCpy */
/*~~~~~~~~~~~~~~~~~~~*/
void grad_gXYsharedCpy(float *gXYmain)
{

    /* Copy gXYshared in non-shared CPU RAM memory */
    memcpy(gXYmain, arm_map->gXYshared, NUM_FEATURES * sizeof(float));


    /*__DEBUG__*/
    /*for (int i=0; i < NUM_FEATURES; i++) {
        printf("DRIVER memcpy *(arm_map->gXYshared + %03d) = 0x%08x = %03dd", i, *(arm_map->gXYshared + i), *(arm_map->gXYshared + i));
        printf("-----DRIVER memcpy *(gXYmain+ %03d) = 0x%08x = %03dd\n", i, *(gXYmain + i), *(gXYmain + i));
    }*/
    /*__DEBUG__*/
}

/*##########################*/
/*__END__GRADXY__FUNCTIONS__*/
/*##########################*/



/*######################*/
/*__xtrBIP2__FUNCTIONS__*/
/*######################*/

/*~~~~~~~~~~~~~~~~~~~*/
/* grad_xtrBIP2qpu() */
/*~~~~~~~~~~~~~~~~~~~*/
void grad_xtrBIP2qpu(int toggle)
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

    /* Pass vc_xtrBIP2code pointer to QPUs */
    for (int i = 0; i < NUM_QPUS; i++)
    {
        arm_map->msg[i][1] = grad_VCptrsArray[3]; //vc_xtrBIP2code
    }

    /* Send insructions through mailbox */
    execute_qpu(mb, NUM_QPUS, grad_VCptrsArray[10], GPU_FFT_NO_FLUSH,
                GPU_FFT_TIMEOUT); //vc_msg
}


/*~~~~~~~~~~~~~~~~~~~~~~~*/
/* grad_xtrBIP2Display() */
/*~~~~~~~~~~~~~~~~~~~~~~~*/
void grad_xtrBIP2sharedDisplay(void)
{

    /* Check pointers values */
    printf("\nDRIVER Check arm_map->xtrBIP2shared = %p\n", arm_map->xtrBIP2shared);

    /* Display xtrBIP2shared verbose */
    for (int i = 0; i < NUM_FEATURES * GRAD_ELMTS * 4; i++)
    {
        if ((i % (GRAD_ELMTS * 4)) == 0) printf("\nQPU %u, FEATURE %u\n",
                                                    i / (GRAD_ELMTS) % (NUM_QPUS), i / (GRAD_ELMTS));
        if ((i % (GRAD_DIM * 4)) == 0) printf("\nRow %u\n",
                                                  i / (GRAD_DIM) % (GRAD_DIM));
        //printf("DRIVER &*(arm_map->xtrBIP2shared + %04d) = &%p = &%04dd ----- ", i, (arm_map->xtrBIP2shared + i), (arm_map->xtrBIP2shared + i));
        printf(" DRIVER *(arm_map->xtrBIP2shared + %04d) = %.8ff\n", i,
               *(arm_map->xtrBIP2shared + i));
    }


    /* Display only values */
    /*for (int i=0; i < NUM_FEATURES*GRAD_ELMTS*4; i++) {
        printf("%f\n", *(arm_map->xtrBIP2shared + i));
    }*/
}

/*###########################*/
/*__END__xtrBIP2__FUNCTIONS__*/
/*###########################*/



/*##########################*/
/*__computeBIP2__FUNCTIONS__*/
/*##########################*/

/*~~~~~~~~~~~~~~~~~~~~~~~*/
/* grad_computeBIP2qpu() */
/*~~~~~~~~~~~~~~~~~~~~~~~*/
void grad_computeBIP2qpu(int toggle)
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

    /* Pass vc_computeBIP2code pointer to QPUs */
    for (int i = 0; i < NUM_QPUS; i++)
    {
        arm_map->msg[i][1] = grad_VCptrsArray[4]; //vc_computeBIP2code
    }

    /* Send insructions through mailbox */
    execute_qpu(mb, NUM_QPUS, grad_VCptrsArray[10], GPU_FFT_NO_FLUSH,
                GPU_FFT_TIMEOUT); //vc_msg
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/* grad_computeBIP2Display() */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void grad_computeBIP2sharedDisplay(void)
{

    /* Check pointers values */
    printf("\nDRIVER Check arm_map->computeBIP2shared = %p\n",
           arm_map->computeBIP2shared);

    /* Display computeBIP2shared verbose */
    for (int i = 0; i < NUM_FEATURES * GRAD_ELMTS; i++)
    {
        if ((i % (GRAD_ELMTS)) == 0) printf("\nQPU %u, FEATURE %u\n",
                                                i / (GRAD_ELMTS) % (NUM_QPUS), i / (GRAD_ELMTS));
        if ((i % (GRAD_DIM)) == 0)
        {
            printf("\nRow %u\n", i / (GRAD_DIM) % (GRAD_DIM));
        }
        //printf("DRIVER &*(arm_map->computeBIP2shared + %04d) = &%p = &%04dd ----- ", i, (arm_map->computeBIP2shared + i), (arm_map->computeBIP2shared + i));
        printf(" DRIVER *(arm_map->computeBIP2shared + %04d) = %.8ff\n", i,
               *(arm_map->computeBIP2shared + i));
    }


    /* Display only values */
    /*for (int i=0; i < NUM_FEATURES*GRAD_ELMTS*4; i++) {
        printf("%f\n", *(arm_map->computeBIP2shared + i));
    }*/
}

/*###############################*/
/*__END__computeBIP2__FUNCTIONS__*/
/*###############################*/



/*#####################*/
/*__dPOSXY__FUNCTIONS__*/
/*#####################*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~*/
/* grad_dPosXYsharedInit() */
/*~~~~~~~~~~~~~~~~~~~~~~~~~*/
void grad_dPosXYsharedInit()
{

    /**(arm_map->dPosXYshared) = 1.2;
    *(arm_map->dPosXYshared + 30) = 0.9;

    for(int i = 1; i < 30; i++)
    {
        *(arm_map->dPosXYshared + i) = *(arm_map->dPosXYshared) + i*0.05;
    }

    for(int i = 31; i < 60; i++)
    {
        *(arm_map->dPosXYshared + i) = *(arm_map->dPosXYshared + 30) + i*0.05;
    }*/

    //*(arm_map->dPosXYshared) = 0.258435;
    //*(arm_map->dPosXYshared + 30) = 1.29545;

    *(arm_map->dPosXYshared) = 0.0;
    *(arm_map->dPosXYshared + 30) = 0.0;

    for (int i = 1; i < 30; i++)
    {
        *(arm_map->dPosXYshared + i) = *(arm_map->dPosXYshared);
    }

    for (int i = 31; i < 60; i++)
    {
        *(arm_map->dPosXYshared + i) = *(arm_map->dPosXYshared + 30);
    }
}



/*~~~~~~~~~~~~~~~~~~*/
/* grad_dPosXYqpu() */
/*~~~~~~~~~~~~~~~~~~*/
void grad_dPosXYqpu(int toggle)
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

    /* Pass vc_dPosXYcode pointer to QPUs */
    for (int i = 0; i < NUM_QPUS; i++)
    {
        arm_map->msg[i][1] = grad_VCptrsArray[5]; //vc_dPosXYcode
    }

    /* Send insructions through mailbox */
    execute_qpu(mb, NUM_QPUS, grad_VCptrsArray[10], GPU_FFT_NO_FLUSH,
                GPU_FFT_TIMEOUT); //vc_msg
}



/*~~~~~~~~~~~~~~~~~~~~~~*/
/* grad_dPosXYDisplay() */
/*~~~~~~~~~~~~~~~~~~~~~~*/
void grad_dPosXYsharedDisplay(void)
{

    /* Check pointers values */
    printf("\nDRIVER Check arm_map->dPosXYshared = %p\n", arm_map->dPosXYshared);

    /* Display dPosXYshared verbose */
    for (int i = 0; i < NUM_FEATURES * 2; i++)
    {
        printf(" DRIVER *(arm_map->dPosXYshared + %04d) = %.8ff\n", i,
               *(arm_map->dPosXYshared + i));
    }


    /* Display only values */
    /*for (int i=0; i < NUM_FEATURES; i++) {
        printf("%f\n", *(arm_map->dPosXYshared + i));
    }*/
}



/*~~~~~~~~~~~~~~~~~~~~~~*/
/* grad_dPosXYsharedCpy */
/*~~~~~~~~~~~~~~~~~~~~~~*/
void grad_dPosXYsharedCpy(float *dPosXYmain)
{

    /* Copy dPosXYshared in non-shared CPU RAM memory */
    memcpy(dPosXYmain, arm_map->dPosXYshared, 2 * NUM_FEATURES * sizeof(float));


    /*__DEBUG__*/
    /*for (int i=0; i < 2*NUM_FEATURES; i++) {
        printf("DRIVER memcpy *(arm_map->dPosXYshared + %03d) = %.8ff", i, *(arm_map->dPosXYshared + i), *(arm_map->dPosXYshared + i));
        printf("-----DRIVER memcpy *(dPosXYmain+ %03d) = %.8ff\n", i, *(dPosXYmain + i), *(dPosXYmain + i));
    }*/
    /*__DEBUG__*/
}

/*##########################*/
/*__END__dPOSXY__FUNCTIONS__*/
/*##########################*/



/*####################*/
/*__detXY__FUNCTIONS__*/
/*####################*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/* grad_detXYsharedCompute() */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void grad_detXYsharedCompute()
{
    for (int i = 0; i < NUM_FEATURES; i++)
    {
        *(arm_map->detXYshared + i) = 1 / ((*(arm_map->gXXshared + i)) * (*
                                           (arm_map->gYYshared + i)) - (*(arm_map->gXYshared + i)) * (*
                                                   (arm_map->gXYshared + i)));
    }
}



/*~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/* grad_detXYsharedDisplay() */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void grad_detXYsharedDisplay(void)
{

    /* Check pointers values */
    printf("\nDRIVER Check arm_map->detXYshared = %p\n", arm_map->detXYshared);

    /* Display detXYshared verbose */
    for (int i = 0; i < NUM_FEATURES; i++)
    {
        printf(" DRIVER *(arm_map->detXYshared + %04d) = %.8ff\n", i,
               *(arm_map->detXYshared + i));
    }


    /* Display only values */
    /*for (int i=0; i < NUM_FEATURES; i++) {
        printf("%f\n", *(arm_map->detXYshared + i));
    }*/

}


/*~~~~~~~~~~~~~~~~~~~~~*/
/* grad_detXYsharedCpy */
/*~~~~~~~~~~~~~~~~~~~~~*/
void grad_detXYsharedCpy(float *detXYmain)
{

    /* Copy detXYshared in non-shared CPU RAM memory */
    memcpy(detXYmain, arm_map->detXYshared, 2 * NUM_FEATURES * sizeof(float));


    /*__DEBUG__*/
    /*for (int i=0; i < 2*NUM_FEATURES; i++) {
        printf("DRIVER memcpy *(arm_map->detXYshared + %03d) = %.8ff", i, *(arm_map->detXYshared + i), *(arm_map->detXYshared + i));
        printf("-----DRIVER memcpy *(detXYmain+ %03d) = %.8ff\n", i, *(detXYmain + i), *(detXYmain + i));
    }*/
    /*__DEBUG__*/
}
/*#########################*/
/*__END__detXY__FUNCTIONS__*/
/*#########################*/



/*######################*/
/*__frameX3__FUNCTIONS__*/
/*######################*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/* grad_frameX3sharedDisplay() */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void grad_frameX3sharedDisplay(void)
{

    /* Check pointers values */
    printf("\nDRIVER Check arm_map->frameX3shared = %p\n", arm_map->frameX3shared);

    /* Display frameX3shared verbose */
    for (int i = 0; i < NUM_FEATURES * 2; i++)
    {
        printf(" DRIVER *(arm_map->frameX3shared + %04d) = %.8ff\n", i,
               *(arm_map->frameX3shared + i));
    }


    /* Display only values */
    /*for (int i=0; i < NUM_FEATURES; i++) {
        printf("%f\n", *(arm_map->frameX3shared + i));
    }*/

}



/*~~~~~~~~~~~~~~~~~~~~~~~*/
/* grad_frameX3sharedCpy */
/*~~~~~~~~~~~~~~~~~~~~~~~*/
void grad_frameX3sharedCpy(float *frameX3main)
{

    /* Copy frameX3shared in non-shared CPU RAM memory */
    memcpy(frameX3main, arm_map->frameX3shared, 2 * NUM_FEATURES * sizeof(float));


    /*__DEBUG__*/
    /*for (int i=0; i < 2*NUM_FEATURES; i++) {
        printf("DRIVER memcpy *(arm_map->frameX3shared + %03d) = %.8ff", i, *(arm_map->frameX3shared + i), *(arm_map->frameX3shared + i));
        printf("-----DRIVER memcpy *(frameX3main+ %03d) = %.8ff\n", i, *(frameX3main + i), *(frameX3main + i));
    }*/
    /*__DEBUG__*/
}
/*###########################*/
/*__END__frameX3__FUNCTIONS__*/
/*###########################*/


/*#################################################*/
/*#################################################*/
/*                                                 */
/*__END__FUNCTIONS__DEPENDENT__OF__MEMORY__LAYOUT__*/
/*                                                 */
/*#################################################*/
/*#################################################*/
