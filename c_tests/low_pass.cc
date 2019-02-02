#include <stdio.h>

#include <vector>
#include <gnuradio/filter/firdes.h>
#include <gnuradio/filter/fir_filter.h>

using namespace gr::filter;



using namespace std;

//
// g++ -I /home/wspeirs/src/gnuradio-3.7.9.3/gr-filter/include/ low_pass.cc -L /home/wspeirs/src/gnuradio-3.7.9.3/build/gr-filter/lib/ -lgnuradio-filter -o test
// LD_LIBRARY_PATH=/home/wspeirs/src/gnuradio-3.7.9.3/build/gr-filter/lib/ ./test
//

int main(int argc, char **argv) {
    vector<float> taps = firdes::low_pass(1.0, 10000000, 100000, 10000, (gr::filter::firdes::win_type)0, 0);

    kernel::fir_filter_ccf filter = kernel::fir_filter_ccf(20, taps);

    printf("%0.16f\n", taps[0]);

    return 0;
}
