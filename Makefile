CC=mpicc
CFLAGS=-I.

EBGM:
	$(CC) -lm -I/usr/include/opencv -lcv -lhighgui -o EBGM EBGM.c EBGM_FaceComparison.c EBGM_FeatureVectors.c GaborFilterResponse.c -I. -O3

.PHONY: clean

clean:
	rm -f EBGM EBGM.o