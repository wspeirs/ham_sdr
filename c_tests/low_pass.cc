#include <stdio.h>

#include <vector>
#include <gnuradio/filter/firdes.h>
//#include <gnuradio/filter/fir_filter.h>
#include <gnuradio/filter/fir_filter_with_buffer.h>
#include <gnuradio/random.h>

#include <volk/volk.h>

#include "/home/wspeirs/src/gnuradio-3.7.9.3/build-debug/gr-filter/lib/fir_filter_ccf_impl.h"

using namespace gr::filter;

using namespace std;

//
// g++ -g -I /home/wspeirs/src/gnuradio-3.7.9.3/gr-filter/include/ low_pass.cc -L /home/wspeirs/src/gnuradio-3.7.9.3/build-debug/gr-filter/lib/ -lgnuradio-filter -o test
// LD_LIBRARY_PATH=/home/wspeirs/src/gnuradio-3.7.9.3/build-debug/gr-filter/lib/ ./test
//

#define MAX_DATA        (16383)
#define ERR_DELTA       (1e-5)

typedef gr_complex i_type;
typedef gr_complex o_type;
typedef float      tap_type;
typedef gr_complex acc_type;

static gr::random rndm;

static float
uniform()
{
  return 2.0 * (rndm.ran1() - 0.5); // uniformly (-1, 1)
}

static void
random_floats(float *buf, unsigned n)
{
  for(unsigned i = 0; i < n; i++)
    buf[i] = (float)rint(uniform() * 32767);
}

static void
random_complex(gr_complex *buf, unsigned n)
{
  for(unsigned i = 0; i < n; i++) {
    float re = rint(uniform() * MAX_DATA);
    float im = rint(uniform() * MAX_DATA);
    buf[i] = gr_complex(re, im);
  }
}

static o_type
ref_dotprod(const i_type input[], const tap_type taps[], int ntaps)
{
    acc_type sum = 0;
    for(int i = 0; i < ntaps; i++) {
      sum += input[i] * taps[i];
    }

    //return gr_complex(121,9)*sum;
    return sum;
}


int main(int argc, char **argv) {
    vector<float> taps = firdes::low_pass(1.0, 10000000, 100000, 10000, (gr::filter::firdes::win_type)0, 0);

//    for(int i=0; i < taps.size(); ++i) {
//        printf("%0.016f\n", taps[i]);
//    }

//    printf("%0.016f\n", taps.front()+1);
//    printf("%0.016f\n", taps.back()-1);
//
//    printf("TAPS: %d\n", taps.size());
//
//    return 0;

    vector<gr_complex> input = vector<gr_complex>(100, gr_complex(1.0, 1.0));
    vector<gr_complex> output = vector<gr_complex>(5);

//    while(input.size() < taps.size() + output.size() + 1) {
//        input.push_back(gr_complex(0.0, 0.0));
//    }

//    kernel::fir_filter_with_buffer_ccf *filter = new kernel::fir_filter_with_buffer_ccf(taps);
    kernel::fir_filter_ccf *filter = new kernel::fir_filter_ccf(20, taps);

    filter->filterNdec(output.data(), input.data(), output.size(), 20);

    for(int i=0; i < output.size(); ++i) {
        printf("R: %0.16f I: %0.16f\n", output[i].real(), output[i].imag());
    }

/*
    unsigned int decimate = 3;

    const int MAX_TAPS   = 29;
    const int OUTPUT_LEN = 37;
    const int INPUT_LEN  = MAX_TAPS + OUTPUT_LEN;
    size_t align = volk_get_alignment();

    // Mem aligned buffer not really necessary, but why not?
//    i_type   *input = (gr_complex*)volk_malloc(INPUT_LEN*sizeof(gr_complex), align);
    i_type   *input = (gr_complex*)malloc(INPUT_LEN*sizeof(gr_complex));
    i_type   *dline = (gr_complex*)volk_malloc(INPUT_LEN*sizeof(gr_complex), align);
    o_type   *expected_output = (gr_complex*)volk_malloc(OUTPUT_LEN*sizeof(gr_complex), align);
//    o_type   *actual_output = (gr_complex*)volk_malloc(OUTPUT_LEN*sizeof(gr_complex), align);
    o_type   *actual_output = (gr_complex*)malloc(OUTPUT_LEN*sizeof(gr_complex));
    tap_type *taps = (float*)volk_malloc(MAX_TAPS*sizeof(float), align);

    srandom(0);     // we want reproducibility
    memset(dline, 0, INPUT_LEN*sizeof(i_type));

//    for(int n = 0; n <= MAX_TAPS; n++) {
//      for(int ol = 0; ol <= OUTPUT_LEN; ol++) {

        int n = 15;
        int ol = 3;

        // build random test case
        random_complex(input, INPUT_LEN);
        random_floats(taps, MAX_TAPS);

        // compute expected output values
        memset(dline, 0, INPUT_LEN*sizeof(i_type));

        for(int o = 0; o < (int)(ol/decimate); o++) {
            // use an actual delay line for this test
            for(int dd = 0; dd < (int)decimate; dd++) {
                for(int oo = INPUT_LEN-1; oo > 0; oo--)
                    dline[oo] = dline[oo-1];

                dline[0] = input[decimate*o+dd];
            }

            expected_output[o] = ref_dotprod(dline, taps, n);
        }

        // build filter
        vector<tap_type> f1_taps(&taps[0], &taps[n]);
        kernel::fir_filter_with_buffer_ccf *f1 = new kernel::fir_filter_with_buffer_ccf(f1_taps);

        printf("OUT LEN: %d\n", ol/decimate);

        // zero the output, then do the filtering
        memset(actual_output, 0, OUTPUT_LEN*sizeof(gr_complex));
        f1->filterNdec(actual_output, input, ol/decimate, decimate);

        // check results
        //
        // we use a sloppy error margin because on the x86 architecture,
        // our reference implementation is using 80 bit floating point
        // arithmetic, while the SSE version is using 32 bit float point
        // arithmetic.

        for(int o = 0; o < (int)(ol/decimate); o++) {
//          CPPUNIT_ASSERT_COMPLEXES_EQUAL(expected_output[o], actual_output[o], sqrt((float)n)*0.25*MAX_DATA*MAX_DATA * ERR_DELTA);
          printf("EXP R: %0.04f I: %0.04f ACT R: %0.04f I: %0.04f\n", expected_output[o].real(), expected_output[0].imag(), actual_output[o].real(), actual_output[o].imag());
        }

        delete f1;
//      }
//    }

//    volk_free(input);
    free(input);
    volk_free(dline);
    volk_free(expected_output);
//    volk_free(actual_output);
    free(actual_output);
    volk_free(taps);

*/

    printf("\n");
    return 0;
}
