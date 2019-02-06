#!/usr/bin/env bash

g++ -g -o test low_pass.cc \
    -I /home/wspeirs/src/gnuradio-3.7.9.3/gr-filter/include/ \
    -L /home/wspeirs/src/gnuradio-3.7.9.3/build/gr-filter/lib/ \
    -lgnuradio-filter \
    -L /home/wspeirs/src/gnuradio-3.7.9.3/build/volk/lib/ \
    -l volk \
    -l boost_system \
    -L /home/wspeirs/src/gnuradio-3.7.9.3/build/gnuradio-runtime/lib/ \
    -lgnuradio-runtime

#VOLK_GENERIC=1 LD_LIBRARY_PATH=/home/wspeirs/src/gnuradio-3.7.9.3/build/gr-filter/lib/:/home/wspeirs/src/gnuradio-3.7.9.3/build/volk/lib/:/home/wspeirs/src/gnuradio-3.7.9.3/build/gnuradio-runtime/lib valgrind ./test