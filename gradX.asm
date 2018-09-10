include(helpers.asm)



# Constants we'll use later on
define(`VECTORS_PER_PASS', 1)
define(`ELEMENTS_PER_PASS', `eval(VECTORS_PER_PASS * 16)')
define(`ELEMENTS_PER_PASS_MINUS_ONE', `eval(ELEMENTS_PER_PASS - 1)')
define(`A_BYTES_PER_PASS', `eval(ELEMENTS_PER_PASS * 4)')
define(`B_BYTES_PER_PASS', `eval(ELEMENTS_PER_PASS * 4)')
define(`ELEMENTS_PER_FINISH_PASS', 16)
define(`ELEMENTS_PER_FINISH_PASS_MINUS_ONE', `eval(ELEMENTS_PER_FINISH_PASS - 1)')
define(`A_BYTES_PER_FINISH_PASS', `eval(ELEMENTS_PER_FINISH_PASS * 4)')
define(`B_BYTES_PER_FINISH_PASS', `eval(ELEMENTS_PER_FINISH_PASS * 4)')
define(`VPM_ROWS_PER_PASS', 1)
define(`NUM_QPUS', 12)
define(`ALL_DONE_SEMA', 0)
define(`SHOULD_DISABLE_TMU_SWAPPING', 1)
define(`WIDTH', 360)
define(`HEIGHT', 240)
define(`WINDOW_DIM', 23)
define(`GRAD_DIM', 21)
define(`GRAD_ELMTS', 441)
define(`NUM_FEATURES', 30)



# Registers used to hold uniforms
define(`rFirstFrameAddress', ra0)
define(`rFeaturesAddress', ra1)
define(`rGradXAddress', ra2)
define(`rGXXAddress', ra3)
define(`rGradYAddress', ra4)
define(`rGYYAddress', ra5)
define(`rGXYAddress', ra6)
define(`rWhichQPU', ra7)
define(`rSecondFrameAddress', ra8)
define(`rXtrBIP2Address', ra9)
define(`rDPosXYAddress', ra10)
define(`rComputeBIP2Address', ra11)
define(`rDetXYAddress', ra12)
define(`rFrameX3Address', ra13)



# Registers used to hold working values
define(`rCurrentFeatureNumber', ra14)
define(`rCurrentFeatureRow', ra15)
define(`rCurrentFeatureCol', ra16)
define(`rCurrentWindowAddress', ra17)
define(`rCurrentWindowPass', ra18)
define(`rCurrentWindowRow', ra19)
define(`rGrad0to15', ra20)
define(`rGrad16to20', ra21)
define(`rCurrentGradAddress', ra22)
define(`rVPMWriteAddr', ra30)
define(`rDMAStoreAddrX', ra31)
#define(`rDMAStoreAddrY', ra31) #VPM HORIZONTAL ACCESS
#define(`rDMALoadAddrY', ra30)
#define(`rVPMReadAddr', ra31)

define(`rLinearRampLow', rb0)
define(`rLinearRampUp', rb1)
define(`rSelectFiveElements', rb2)
define(`rGXX', rb31)



# The special accumulator registers, heavily reused so generally not named
define(`rAccum0', r0)
define(`rAccum1', r1)
define(`rAccum2', r2)
define(`rTotal', r3)



# LOAD__UNIFORMS__ARGUMENTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~
or rFirstFrameAddress, raReadUniform, 0; nop
or rFeaturesAddress, raReadUniform, 0; nop
or rGradXAddress, raReadUniform, 0; nop
or rGXXAddress, raReadUniform, 0; nop
or rGradYAddress, raReadUniform, 0; nop
or rGYYAddress, raReadUniform, 0; nop
or rGXYAddress, raReadUniform, 0; nop
or rWhichQPU, raReadUniform, 0; nop
or rSecondFrameAddress, raReadUniform, 0; nop
or rXtrBIP2Address, raReadUniform, 0; nop
or rDPosXYAddress, raReadUniform, 0; nop
or rComputeBIP2Address, raReadUniform, 0; nop
or rDetXYAddress, raReadUniform, 0; nop
or rFrameX3Address, raReadUniform, 0; nop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##########__END__LOAD__UNIFORMS__ARGUMENTS__##########



# TURN__OFF__TMU__SWITCHING
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Turn off the automatic switching of TMU0/1 behind the scenes since we're
# going to explicitly control calling each TMU unit.
ldi raTmuNoSwap, SHOULD_DISABLE_TMU_SWAPPING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##########__END__TURN__OFF__TMU__SWITCHING__##########



# SET__UP__VERTICAL__VPM_DMA
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up our working area of memory in the shared VPM space, based on the
# QPU number we've been given. The VPM can be viewed as a 2d table, 16 floats
# wide and 64 rows high. In our case, we use 12 QPUs, and give each one two
# 16-word vertical rows in the VPM table.

nop rb39, r0, r0; mul24 rTotal, rWhichQPU, VPM_ROWS_PER_PASS
##rTotal = rWhichQPU * VPM_ROWS_PER_PASS = rWhichQPU * 1

#ldi rAccum0, VPM_DMA_LOAD_SETUP_ADDRY_SHIFT
###rAccum0 = VPM_DMA_LOAD_SETUP_ADDRY_SHIFT = 4
#shl rDMALoadAddrY, rTotal, rAccum0; nop
###rDMALoadAddrY = (rWhichQPU * 2)<<4 : DMAtoVPM - datasheet Table 36

#ldi rAccum0, VPM_BLOCK_READ_SETUP_ADDR_SHIFT
###rAccum0 = VPM_BLOCK_READ_SETUP_ADDR_SHIFT = 0
#shl rVPMReadAddr, rTotal, rAccum0; nop
###rVPMReadAddr = (rWhichQPU * 2)<<0 : VPMtoQPU - datasheet Table 33

ldi rAccum0, VPM_BLOCK_WRITE_SETUP_ADDR_SHIFT
##rAccum0 = VPM_BLOCK_WRITE_SETUP_ADDR_SHIFT = 0
shl rVPMWriteAddr, rTotal, rAccum0; nop
##rVPMWriteAddr = (rWhichQPU * 1)<<0 : QPUtoVPM - datasheet Table 32

ldi rAccum0, VPM_DMA_STORE_SETUP_ADDRX_SHIFT
##rAccum0 = VPM_DMA_STORE_SETUP_ADDRX_SHIFT = 3
shl rDMAStoreAddrX, rTotal, rAccum0; nop
##rDMAStoreAddrX = (rWhichQPU * 1)<<3 : VPMtoDMA - datasheet Table 34

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##########__END__SET__UP__VERTICAL__VPM_DMA__##########



# COMPUTE__LINEAR__RAMP__LOW
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create a special vector that we'll use to select wanted pixels
# on 23-elements row. The result should be:
# rLinearRampLow = [0, 1, 2, ..., 14, 15]*4
ldi rAccum0, [0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
ldi rAccum1, [0, 0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
add rAccum0, rAccum0, rAccum1; nop
ldi rAccum1, [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3]
add rAccum0, rAccum0, rAccum1; nop
ldi rAccum1, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 3, 3]
add rAccum0, rAccum0, rAccum1; nop
ldi rAccum1, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3]
add rAccum0, rAccum0, rAccum1; nop
shl rAccum0, rAccum0, 2; nop
add rLinearRampLow, rAccum0, 0; nop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#########__END__COMPUTE__LINEAR__RAMP__LOW__##########



# COMPUTE__LINEAR__RAMP__UP
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Create a special vector that we'll use to select wanted pixels
# on 23-elements row. The result should be:
# rLinearRampUp = [2, 3, 4, ..., 16, 17]*4
ldi rAccum1, 8
add rLinearRampUp, rAccum1, rLinearRampLow; nop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#########__END__COMPUTE__LINEAR__RAMP__UP__##########



# COMPUTE__SELECT__FIVE__ELEMENTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create a special vector that we'll use to select five first
# elements of rGrad16to20 register.
# The result should be:
# rSelectFiveElements = [1, 1, 1, 1, 1, 0, .., 0]
ldi rAccum1, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
add rSelectFiveElements, rAccum1, 0; nop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#########__END__COMPUTE__SELECT__FIVE__ELEMENTS__##########



# GRADX__LOOP__ON__30__FEATURES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Initialize loop on 30 features
or rCurrentFeatureNumber, rWhichQPU, 0; nop
##rCurrentFeatureNumber = rWhichQPU

# This is the section where we compute
# (21*21)GradX array
# on a (21*23)-pixel window
# for a SINGLE feature
loop_feature:
ldi rAccum0, NUM_FEATURES
##rAccum0 = NUM_FEATURES = 30
sub ra39, rCurrentFeatureNumber, rAccum0; nop
brr.ne ra39, loop_feature_break
##branch if result is NOT negative (ne = negative flag empty)
NOP
NOP
NOP



# Initialze rGXX
ldi rGXX, 0
##rGXX = 0



# GET__CURRENT__FEATURE__ROW/COL
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Set up the reading address for the featureShared array
shl rAccum0, rCurrentFeatureNumber, 2; nop
##rAccum0 = rCurrentFeatureNumber * 4 = (rWhichQPU + i*NUM_FEATURES)*4
ldi rAccum1, NUM_FEATURES
##rAccum1 = NUM_FEATURES = 30
add rAccum1, rCurrentFeatureNumber, rAccum1; nop
##rAccum1 = rCurrentFeatureNumber + NUM_FEATURES
shl rAccum1, rAccum1, 2; nop
##rAccum1 = (rCurrentFeatureNumber + NUM_FEATURES)*4

# Kick off 2 vectors fetches through the TMUS
# These 2 vectors contain :
# CURRENT_FEATURE_ROW_ADDRESS inside featureShared array
# CURRENT_FEATURE_COL_ADDRESS inside featureShared array
add raTmu0S, rFeaturesAddress, rAccum0; nop
##raTmu0S = rFeaturesAddress + rCurrentFeatureNumber*4
add raTmu1S, rFeaturesAddress, rAccum1; nop
##raTmu1S = rFeaturesAddress + (rCurrentFeatureNumber + NUM_FEATURES)*4

# Read the pending ROW result from the queue
# and store it into rCurrentFeatureRow register
or.ldtmu0 ra39, ra39, ra39; nop
or rCurrentFeatureRow, r4, 0; nop

# Read the pending COL result from the queue
# and store it into rCurrentFeatureCol register
or.ldtmu1 ra39, ra39, ra39; nop
or rCurrentFeatureCol, r4, 0; nop

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##########__END__GET__CURRENT__FEATURE__ROW/COL__##########



# COMPUTE__CURRENT__WINDOW__ADDRESS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sub rAccum0, rCurrentFeatureRow, 10; nop
##rAccum0 = rCurrentFeatureRow - 10
sub rAccum1, rCurrentFeatureCol, 11; nop
##rAccum1 = rCurrentFeatureCol - 11
ldi rAccum2, WIDTH
##rAccum2 = WIDTH = 360
shl rAccum1, rAccum1, 2; nop
##rAccum1 = (rCurrentFeatureCol - 11)*4
nop rb39, rb39, rb39; mul24 rAccum0, rAccum0, rAccum2
##rAccum0 = (rCurrentFeatureRow - 10)*360
shl rAccum0, rAccum0, 2; nop
##rAccum0 = (rCurrentFeatureRow - 10)*360*4
add rTotal, rAccum0, rAccum1; nop
##rTotal = (rCurrentFeatureRow - 10)*360*4 + (rCurrentFeatureCol - 11)*4
add rCurrentWindowAddress, rFirstFrameAddress, rTotal; nop
##rCurrentWindowAddress = rFirstFrameAddress + (rCurrentFeatureRow - 10)*360*4 + (rCurrentFeatureCol - 11)*4

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##########__END__COMPUTE__CURRENT__WINDOW__ADDRESS__##########



# COMPUTE__CURRENT__FEATURE__21*21__GRADX
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Initialize loop on (21-row)*(23-col) window
ldi rCurrentWindowPass, 0
##rCurrentWindowPass = 0

# This is the section where we compute
# a single 21-element row of GradX
# on a single 23-pixel row of current window
loop_window:
ldi rAccum0, GRAD_DIM
##rAccum0 = GRAD_DIM = 21
sub ra39, rCurrentWindowPass, rAccum0; nop
brr.ne ra39, loop_window_break
##branch if result is NOT negative (ne = negative flag empty)
NOP
NOP
NOP



# FETCH__2*21__PIXELS__THROUGH__TMUS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Set up the reading address for the current window ROW
ldi rAccum0, WIDTH
##rAccum0 = WIDTH = 360
shl rAccum0, rAccum0, 2; nop
##rAccum0 = WIDTH*4 = 360*4
nop rb39, rb39, rb39; mul24 rAccum0, rCurrentWindowPass, rAccum0
##rAccum0 = rCurrentWindowPass*360*4
add rCurrentWindowRow, rCurrentWindowAddress, rAccum0; nop
##rCurrentWindowRow = rCurrentWindowAddress + rCurrentWindowPass*360*4

# Constants we use to select wanted elements on 23-pixel single row
or rAccum1, rLinearRampLow, rLinearRampLow; nop
or rAccum2, rLinearRampUp, rLinearRampUp; nop


# Kick off 2*16*Pixels fetches through the TMUs
add raTmu0S, rCurrentWindowRow, rAccum1; nop
##raTmu0S = rCurrentWindowRow + [0, 1, 2, ..., 15]*4
add raTmu1S, rCurrentWindowRow, rAccum2; nop
##raTmu1S = rCurrentWindowRow + [2, 3, 4, ..., 17]*4


ldi rAccum0, 64
##rAccum0 = 64 = 16*4
add rAccum1, rAccum1, rAccum0; nop
##rAccum1 = rAccum1 + 64 = rLinearRampLow + 16*4
add rAccum2, rAccum2, rAccum0; nop
##rAccum2 = rAccum2 + 64 = rLinearRampUp + 16*4


# Kick off 2*16*Pixels fetches through the TMUs
add raTmu0S, rCurrentWindowRow, rAccum1; nop
##raTmu0S = rCurrentWindowRow + [0+16, 1+16, 2+16, ..., 15+16]*4 = rCurrentWindowRow + [16, 17, ..., 31]*4
add raTmu1S, rCurrentWindowRow, rAccum2; nop
##raTmu1S = rCurrentWindowRow + [2+16, 3+16, 4+16, ..., 17+16]*4 = rCurrentWindowRow + [18, 19, ..., 33]*4

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##########__END__FETCH__2*21__PIXELS__THROUGH__TMUS__##########



# COMPUTE__21__GRADX__ON__CURRENT__WINDOW__ROW
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Constant to divide by 2
ldi rAccum2, 0x3f000000
##rAccum2 = 1/2

# Read 2*16*pixels from TMUS
or.ldtmu0 ra39, ra39, ra39; nop
or rAccum0,  r4, 0; nop

or.ldtmu1 ra39, ra39, ra39; nop
or rAccum1,  r4, 0; nop

# Compute 16 gradx
fsub rTotal, rAccum1, rAccum0; nop
nop rb39, rb39, rb39; fmul rGrad0to15, rAccum2, rTotal


# Read 2*16*pixels from TMUS
or.ldtmu0 ra39, ra39, ra39; nop
or rAccum0,  r4, 0; nop

or.ldtmu1 ra39, ra39, ra39; nop
or rAccum1,  r4, 0; nop

# Compute last 5 gradx
fsub rTotal, rAccum1, rAccum0; nop
nop rb39, rb39, rb39; fmul rGrad16to20, rAccum2, rTotal

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##########__END__COMPUTE__21__GRADX__ON__CURRENT__WINDOW__ROW__##########



# COMPUTE__AND__UPDATE__GXX__ON__CURRENT__WINDOW__ROW
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Multiply first 16 gradx
or rTotal, rGrad0to15, 0; nop
##rTotal = rGrad0to15
or rAccum2, rSelectFiveElements, rSelectFiveElements; nop
##rAccum2 = rSelectFiveElements
nop rb39, rb39, rb39; fmul rTotal, rGrad0to15, rTotal
##rTotal = rGrad0to15*rGrad0to15

# Take the 16-component-wide result vector and sum it into a single value
or r0, rTotal, 0; nop
or r3, rTotal, 0; nop
nop rb39, r0, <<1; v8max r0, r0, r0
fadd rTotal, rTotal, r0; nop
or r0, rTotal, 0; nop
or r3, rTotal, 0; nop
nop rb39, r0, <<2; v8max r0, r0, r0
fadd rTotal, rTotal, r0; nop
or r0, rTotal, 0; nop
or r3, rTotal, 0; nop
nop rb39, r0, <<4; v8max r0, r0, r0
fadd rTotal, rTotal, r0; nop
or r0, rTotal, 0; nop
or r3, rTotal, 0; nop
nop rb39, r0, <<8; v8max r0, r0, r0
fadd rTotal, rTotal, r0; nop


# Update rGXX
fadd rGXX, rTotal, rGXX; nop


# Select and Multiply last 5 gradx
or rTotal, rGrad16to20, 0; nop
##rTotal = rGrad16to20
itof rAccum2, rAccum2, rAccum2; nop
##rAccum2 = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, ..., 0.0]
nop rb39, rb39, rb39; fmul rTotal, rGrad16to20, rTotal
##rTotal = rGrad16to20*rGrad16to20
nop rb39, rb39, rb39; fmul rTotal, rAccum2, rTotal
##rTotal = rGrad16to20*rGrad16to20*[1, 1, 1, 1, 1, 0, ..., 0]

# Take the 16-component-wide result vector and sum it into a single value
or r0, rTotal, 0; nop
or r3, rTotal, 0; nop
nop rb39, r0, <<1; v8max r0, r0, r0
fadd rTotal, rTotal, r0; nop
or r0, rTotal, 0; nop
or r3, rTotal, 0; nop
nop rb39, r0, <<2; v8max r0, r0, r0
fadd rTotal, rTotal, r0; nop
or r0, rTotal, 0; nop
or r3, rTotal, 0; nop
nop rb39, r0, <<4; v8max r0, r0, r0
fadd rTotal, rTotal, r0; nop
or r0, rTotal, 0; nop
or r3, rTotal, 0; nop
nop rb39, r0, <<8; v8max r0, r0, r0
fadd rTotal, rTotal, r0; nop


# Update rGXX
fadd rGXX, rTotal, rGXX; nop

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##########__END__COMPUTE__AND__UPDATE__GXX__ON__CURRENT__WINDOW__ROW__##########



# 21__PIXELS__FROM__QPU__TO__VPM
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# VPM__VERTICAL__ACCES
# ~~~~~~~~~~~~~~~~~~~~

# Configure VPM memory storage
define(`STRIDE', 16)
define(`ADDR', 0)
ldi rAccum0, VPM_BLOCK_WRITE_SETUP_VALUE(STRIDE, NOT_HORIZ, NOT_LANED, SIZE_32_BIT, ADDR)
or rb49, rVPMWriteAddr, rAccum0; nop


# Store result into the VPM Fifo.
or rVpmWriteFifo, rGrad0to15, 0; nop

# Store result into the VPM Fifo.
or rVpmWriteFifo, rGrad16to20, 0; nop

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#####__END__VPM__VERTICAL__ACCES__#####

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##########__END__21__PIXELS__FROM__QPU__TO__VPM__##########



# 21__PIXELS__FROM__VPM__TO__DMA
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Compute DMA address
ldi rAccum1, GRAD_DIM
##rAccum1 = 21
shl rAccum1, rAccum1, 2; nop
##rAccum1 = 21*4
nop rb39, rb39, rb39; mul24 rAccum1, rCurrentWindowPass, rAccum1
##rAccum1 = rCurrentWindowPass*GRAD_DIM*4

ldi rAccum0, GRAD_ELMTS
##rAccum0 = GRAD_DIM*GRAD_DIM = GRAD_ELMTS = 441
shl rAccum0, rAccum0, 2; nop
##rAccum0 = GRAD_ELMTS*4 = 441*4
nop rb39, rb39, rb39; mul24 rAccum0, rCurrentFeatureNumber, rAccum0
##rAccum0 = rCurrentFeatureNumber*GRAD_ELMTS*4

add rAccum0, rAccum0, rAccum1; nop
##rAccum0 = rCurrentFeatureNumber*GRAD_DIM*GRAD_DIM*4 + rCurrentWindowPass*GRAD_DIM*4
add rCurrentGradAddress, rGradXAddress, rAccum0; nop
##rCurrentGradAddress = rGradXAddress + rCurrentFeatureNumber*GRAD_DIM*GRAD_DIM*4 + rCurrentWindowPass*GRAD_DIM*4


# DMA__VERTICAL__ACCES
# ~~~~~~~~~~~~~~~~~~~~
# DMA the result into main memory from the VPM
define(`UNITS', 21)
define(`DEPTH', 1)
define(`ADDRY', 0)
define(`ADDRX', 0)

ldi rAccum0, VPM_DMA_STORE_SETUP_VALUE(UNITS, DEPTH, IS_HORIZ, ADDRY, ADDRX, MODEW_32_BIT)
or rb49, rAccum0, rDMAStoreAddrX; nop

MUTEX_ACQUIRE()
VPM_DMA_STORE_START(rCurrentGradAddress)
VPM_DMA_STORE_WAIT_FOR_COMPLETION()
MUTEX_RELEASE()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#####__END__DMA__VERTICAL__ACCES__######

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##########__END__21__PIXELS__FROM__VPM__TO__DMA__##########



add rCurrentWindowPass, rCurrentWindowPass, 1; nop
brr ra39, loop_window
NOP
NOP
NOP

loop_window_break:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##########__END__COMPUTE__CURRENT__FEATURE__21*21__GRADX__##########



# GXX__FROM__QPU__TO__VPM
# ~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~

# VPM__VERTICAL__ACCES
# ~~~~~~~~~~~~~~~~~~~~

# Configure VPM memory storage
define(`STRIDE', 0)
define(`ADDR', 32)
ldi rAccum0, VPM_BLOCK_WRITE_SETUP_VALUE(STRIDE, NOT_HORIZ, NOT_LANED, SIZE_32_BIT, ADDR)
or rb49, rVPMWriteAddr, rAccum0; nop

# Store rGXX result into the VPM Fifo.
or rVpmWriteFifo, rGXX, rGXX; nop

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#####__END__VPM__VERTICAL__ACCES__#####

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##########__END__GXX__FROM__QPU__TO__VPM__##########



# GXX__FROM__VPM__TO__DMA
# ~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~

shl rAccum0, rCurrentFeatureNumber, 2; nop
##rAccum0 = rCurrentFeatureNumber*4
add rCurrentGradAddress, rGXXAddress, rAccum0; nop
##rCurrentGradAddress = rGXXAddress + rCurrentFeatureNumber*4

# DMA__VERTICAL__ACCES
# ~~~~~~~~~~~~~~~~~~~~
# DMA the result into main memory from the VPM
define(`UNITS', 1)
define(`DEPTH', 1)
define(`ADDRY', 32)
define(`ADDRX', 0)

ldi rAccum0, VPM_DMA_STORE_SETUP_VALUE(UNITS, DEPTH, NOT_HORIZ, ADDRY, ADDRX, MODEW_32_BIT)
or rb49, rAccum0, rDMAStoreAddrX; nop

MUTEX_ACQUIRE()
VPM_DMA_STORE_START(rCurrentGradAddress)
VPM_DMA_STORE_WAIT_FOR_COMPLETION()
MUTEX_RELEASE()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#####__END__DMA__VERTICAL__ACCES__######

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##########__END__GXX__FROM__VPM__TO__DMA__##########



add rCurrentFeatureNumber, rCurrentFeatureNumber, NUM_QPUS; nop
brr ra39, loop_feature
NOP
NOP
NOP

loop_feature_break:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##########__END__GRADX__LOOP__ON__30__FEATURES__##########



# COORDINATE__QPUS
# ~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~
# We need to coordinate the execution of all the QPUs, so that the program end
# isn't signaled before they're all done. To handle this, first each program
# raises a semaphore to say that it's done, and then the master QPU (given the
# number 9 in its rWhichQPU uniform) pulls the semaphore down eight times to
# ensure all the others are done, before signaling back to the main CPU.
sema up, ALL_DONE_SEMA

xor rb39, rWhichQPU, 9; nop
brr.zc rb39, non_master_finish
##branch if results is NOT zero (e.g. not QPU9)
NOP
NOP
NOP

# The number of 'down's must match the number of QPUs being run
sema down, ALL_DONE_SEMA
sema down, ALL_DONE_SEMA
sema down, ALL_DONE_SEMA
sema down, ALL_DONE_SEMA
sema down, ALL_DONE_SEMA
sema down, ALL_DONE_SEMA
sema down, ALL_DONE_SEMA
sema down, ALL_DONE_SEMA
sema down, ALL_DONE_SEMA
sema down, ALL_DONE_SEMA
sema down, ALL_DONE_SEMA
sema down, ALL_DONE_SEMA

END_PROGRAM_HARD()

non_master_finish:

END_PROGRAM_SOFT()
