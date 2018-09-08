#!/bin/bash
# Generate .bin file for each .asm file
for f in *.asm; do \
	echo ${f%%.*}; \
	m4 ${f} | ../qpu-asm/qpu-asm -o ${f%%.*}.bin; \
	mv ${f%%.*}.bin ../src;
done;
rm ../src/helpers.bin;
