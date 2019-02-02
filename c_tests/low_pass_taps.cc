#include <stdio.h>
#include <vector>
#include <math.h>

using std::vector;

# define M_PI		3.14159265358979323846

int
compute_ntaps(double sampling_freq, double transition_width)
{
  double a = 53.0;
  int ntaps = (int)(a*sampling_freq/(22.0*transition_width));
  if((ntaps & 1) == 0)	// if even...
ntaps++;		// ...make odd

  return ntaps;
}

vector<float>
hamming(int ntaps)
{
  std::vector<float> taps(ntaps);
  float M = static_cast<float>(ntaps - 1);

  for(int n = 0; n < ntaps; n++)
    taps[n] = 0.54 - 0.46 * cos((2 * M_PI * n) / M);
  return taps;
}


vector<float>
low_pass(double gain,
         double sampling_freq,
         double cutoff_freq,	// Hz center of transition band
         double transition_width)		// used only with Kaiser
{
//  sanity_check_1f(sampling_freq, cutoff_freq, transition_width);

  int ntaps = compute_ntaps(sampling_freq, transition_width);

  printf("NTAPS: %d\n", ntaps);

  // construct the truncated ideal impulse response
  // [sin(x)/x for the low pass case]

  vector<float> taps(ntaps);
  vector<float> w = hamming(ntaps);

//    for (auto tap : w) {
//        printf("%0.09f ", tap);
//    }

  int M = (ntaps - 1) / 2;
  double fwT0 = 2 * M_PI * cutoff_freq / sampling_freq;

  for(int n = -M; n <= M; n++) {
    if(n == 0)
      taps[n + M] = fwT0 / M_PI * w[n + M];
    else {
      // a little algebra gets this into the more familiar sin(x)/x form
      taps[n + M] =  sin(n * fwT0) / (n * M_PI) * w[n + M];
    }
  }

  // find the factor to normalize the gain, fmax.
  // For low-pass, gain @ zero freq = 1.0

  double fmax = taps[0 + M];

  for(int n = 1; n <= M; n++)
    fmax += 2 * taps[n + M];

  gain /= fmax;	// normalize

  for(int i = 0; i < ntaps; i++)
    taps[i] *= gain;

  return taps;
}


int main(int argc, char **argv)
{
    vector<float> taps = low_pass(1, 10e6, 100e3, 10e3);

    printf("LEN: %u\n", taps.size());

    for (auto tap : taps) {
        printf("%0.09f ", tap);
    }

    printf("\n");

    return 0;
}