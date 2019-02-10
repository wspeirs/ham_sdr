use std::f64::consts::PI;
use std::mem::transmute;
use std::ptr::null_mut;
use std::alloc::{System, GlobalAlloc, Layout};
use std::time::{Duration, Instant};

use core::arch::x86_64::{__m256, _mm256_setzero_ps, _mm256_loadu_ps, _mm256_unpacklo_ps, _mm256_unpackhi_ps, _mm256_permute2f128_ps, _mm256_mul_ps, _mm256_add_ps, _mm256_storeu_ps};

use rayon::prelude::*;


pub struct Filter {
    taps: Vec<f32>,
    output: Vec<f32>,  // we keep this around so we don't have to re-allocate every time
    use_avx: bool
}

impl Filter {
    /// Generates the taps for a low-pass filter
    /// Translated from https://github.com/gnuradio/gnuradio/blob/v3.7.9.3/gr-filter/lib/firdes.cc#L92
    pub fn generate_low_pass_taps(gain :f64, sampling_freq :f64, cutoff_freq :f64, transition_width :f64) -> Vec<f32> {
        // perform some sanity checks
        assert!(sampling_freq > 0.0, format!("sampling_freq ({}) < 0", sampling_freq));
        assert!(cutoff_freq > 0.0, format!("cutoff_freq ({}) <= 0", cutoff_freq));
        assert!(cutoff_freq <= sampling_freq / 2.0, format!("cutoff_freq ({}) > {}", cutoff_freq, sampling_freq/2.0));
        assert!(transition_width > 0.0, format!("transition_width <= 0"));

        // we're using a Hamming window
        const MAX_ATTENUATION :f64 = 53.0;

        let ntaps = (MAX_ATTENUATION * sampling_freq / (22.0 * transition_width)).floor() as isize;
        let ntaps  = if (ntaps & 1) == 0 { ntaps + 1 } else { ntaps };

        // construct the truncated ideal impulse response
        // [sin(x)/x for the low pass case]

        let mut window :Vec<f32> = Vec::with_capacity(ntaps as usize);
        let M = (ntaps - 1) as f64;

        // compute the window values
        for n in 0..ntaps {
//            println!("{:0.20} - {:0.20} * (({:0.20}) / {:0.20}).cos()", 0.54, 0.46, (2.0 * PI * n as f64), M);
//            println!("{:0.20}", 2.0_f64 * PI_64 * n as f64);
//            println!("({:0.20}).cos() = {:0.20}", (2.0_f64 * PI * n as f64) / M as f64, ((2.0_f64 * PI * n as f64) / M as f64).cos());
//            println!("{:0.20}", 0.54 - 0.46 * ((2.0_f64 * PI * n as f64) / M as f64).cos());
            window.push((0.54 - 0.46 * ((2.0 * PI * n as f64) / M).cos()) as f32);
        }

        let M :isize = (ntaps - 1) / 2;
        let fw_t0 = 2.0 * PI * cutoff_freq / sampling_freq;

        let mut taps :Vec<f32> = vec![0.0; ntaps as usize];

        // compute the tap values
        for n in -M..=M {
            if n == 0 {
                taps[(n+M) as usize] = (fw_t0 / PI * window[(n+M) as usize] as f64) as f32;
            } else {
                taps[(n+M) as usize] = ((n as f64 * fw_t0).sin() / (n as f64 * PI) * window[(n+M) as usize] as f64) as f32;
            }
        }

        // find the factor to normalize the gain, fmax.
        // For low-pass, gain @ zero freq = 1.0

        let mut fmax = taps[0 + M as usize] as f64;

        for n in 1..= M {
            fmax += 2.0 * taps[(n + M) as usize] as f64;
        }

        let gain = gain / fmax;	// normalize

        for i in 0..ntaps as usize {
            taps[i] = (taps[i] as f64 * gain) as f32;
        }

        return taps;
    }

    /// Construct a new filter using taps.len() as the size for the output buffer
    pub fn new(taps: &[f32]) -> Filter {
        Filter::with_capacity(taps, taps.len())
    }

    /// Construct a new filter using the given capacity for the output buffer
    pub fn with_capacity(taps: &[f32], capacity: usize) -> Filter {
        let mut r_taps = Vec::<f32>::with_capacity(taps.len());

        // we want to reverse the taps so it's easier to manage in the filter code
        taps.iter().rev().for_each(|t| r_taps.push(*t));

        let use_avx = is_x86_feature_detected!("avx");

        debug!("USING {}", if use_avx {"AVX"} else {"GENERIC"});

        Filter {
            taps: r_taps,
            output: vec![0.0_f32; capacity],
            use_avx: use_avx
        }
    }

    /// Computes the dot product: (i*b, q*b) where a[even] = i and a[odd] = q
    #[inline]
    fn dot_product(a: &[f32], b: &[f32]) -> [f32; 2] {
        assert_eq!(a.len(), b.len()*2);

        let f = |acc :(f32,f32), (i, t) :(&[f32], &f32)| (i[0].mul_add(*t, acc.0), i[1].mul_add(*t, acc.1));

        let (r,i) = a.chunks(2)
            .zip(b.iter())
            .fold((0.0, 0.0), f );

        [r,i]
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_product_avx(a: &[f32], b: &[f32]) -> [f32; 2] {
        assert_eq!(a.len(), b.len()*2, "Input (a) and taps (b) are not the correct lengths: {} != {}*2", a.len(), b.len());

        let mut real :f32 = 0.0;
        let mut imag :f32 = 0.0;

        let mut real_ptr :*mut f32 = &mut real as *mut f32;
        let mut imag_ptr :*mut f32 = &mut imag as *mut f32;

        let a_ptr :*const f32 = a.as_ptr();
        let b_ptr :*const f32 = b.as_ptr();

        let (mut a0_val, mut a1_val, mut a2_val, mut a3_val) :(__m256, __m256, __m256, __m256);
        let (mut b0_val, mut b1_val, mut b2_val, mut b3_val) :(__m256, __m256, __m256, __m256);
        let (mut x0_val, mut x1_val, mut x0_low_val, mut x0_high_val, mut x1_low_val, mut x1_high_val) :(__m256, __m256, __m256, __m256, __m256, __m256);
        let (mut c0_val, mut c1_val, mut c2_val, mut c3_val) :(__m256, __m256, __m256, __m256);

        let mut dot_prod0_val :__m256 = _mm256_setzero_ps();
        let mut dot_prod1_val :__m256 = _mm256_setzero_ps();
        let mut dot_prod2_val :__m256 = _mm256_setzero_ps();
        let mut dot_prod3_val :__m256 = _mm256_setzero_ps();

        let mut a_offset :isize = 0;
        let mut b_offset :isize = 0;

        let a_len_mul32 = (a.len() / 32) * 32;

        while a_offset < a_len_mul32 as isize {
            trace!("1st loop a_offset: {}\tb_offset: {}", a_offset, b_offset);

            a0_val = _mm256_loadu_ps(a_ptr.offset(a_offset));
            a1_val = _mm256_loadu_ps(a_ptr.offset(a_offset + 8));
            a2_val = _mm256_loadu_ps(a_ptr.offset(a_offset + 16));
            a3_val = _mm256_loadu_ps(a_ptr.offset(a_offset + 24));

            x0_val = _mm256_loadu_ps(b_ptr.offset(b_offset)); // t0|t1|t2|t3|t4|t5|t6|t7
            x1_val = _mm256_loadu_ps(b_ptr.offset(b_offset + 8));

            x0_low_val = _mm256_unpacklo_ps(x0_val, x0_val); // t0|t0|t1|t1|t4|t4|t5|t5
            x0_high_val = _mm256_unpackhi_ps(x0_val, x0_val); // t2|t2|t3|t3|t6|t6|t7|t7
            x1_low_val = _mm256_unpacklo_ps(x1_val, x1_val);
            x1_high_val = _mm256_unpackhi_ps(x1_val, x1_val);

            // TODO: it may be possible to rearrange swizzling to better pipeline data
            b0_val = _mm256_permute2f128_ps(x0_low_val, x0_high_val, 0x20); // t0|t0|t1|t1|t2|t2|t3|t3
            b1_val = _mm256_permute2f128_ps(x0_low_val, x0_high_val, 0x31); // t4|t4|t5|t5|t6|t6|t7|t7
            b2_val = _mm256_permute2f128_ps(x1_low_val, x1_high_val, 0x20);
            b3_val = _mm256_permute2f128_ps(x1_low_val, x1_high_val, 0x31);

            c0_val = _mm256_mul_ps(a0_val, b0_val);
            c1_val = _mm256_mul_ps(a1_val, b1_val);
            c2_val = _mm256_mul_ps(a2_val, b2_val);
            c3_val = _mm256_mul_ps(a3_val, b3_val);

            dot_prod0_val = _mm256_add_ps(c0_val, dot_prod0_val);
            dot_prod1_val = _mm256_add_ps(c1_val, dot_prod1_val);
            dot_prod2_val = _mm256_add_ps(c2_val, dot_prod2_val);
            dot_prod3_val = _mm256_add_ps(c3_val, dot_prod3_val);

            a_offset += 32;
            b_offset += 16;
        }

        dot_prod0_val = _mm256_add_ps(dot_prod0_val, dot_prod1_val);
        dot_prod0_val = _mm256_add_ps(dot_prod0_val, dot_prod2_val);
        dot_prod0_val = _mm256_add_ps(dot_prod0_val, dot_prod3_val);

        trace!("DP VAL: {:?}", dot_prod0_val);

        // create a layout for our vec![0.0_f32; 8] properly aligned
        let mut dp = vec![0_f32; 8];
        let dp_ptr = dp.as_mut_ptr();

        // Store the results back into the dot product vector
        _mm256_storeu_ps(dp_ptr,dot_prod0_val);

        *real_ptr = *dp_ptr.offset(0);
        *imag_ptr = *dp_ptr.offset(1);
        *real_ptr += *dp_ptr.offset(2);
        *imag_ptr += *dp_ptr.offset(3);
        *real_ptr += *dp_ptr.offset(4);
        *imag_ptr += *dp_ptr.offset(5);
        *real_ptr += *dp_ptr.offset(6);
        *imag_ptr += *dp_ptr.offset(7);

        trace!("re: {} im: {}", *real_ptr, *imag_ptr);

        while a_offset < a.len() as isize {
            trace!("2nd loop a_offset: {}\tb_offset: {}", a_offset, b_offset);

            *real_ptr += (*a_ptr.offset(a_offset)) * (*b_ptr.offset(b_offset));
            *imag_ptr += (*a_ptr.offset(a_offset+1)) * (*b_ptr.offset(b_offset));

            trace!("re: {}\tim: {}", *real_ptr, *imag_ptr);

            a_offset += 2;
            b_offset += 1;
        }

        [*real_ptr, *imag_ptr]
    }

    pub fn filter(&mut self, decimation: usize, input: &[f32]) -> &[f32] {
        assert_eq!(input.len() % 2, 0, "Input length is not even");

        let start_time = Instant::now();

        let stop = (input.len() / (decimation*2)) * (decimation*2);

        // ensure we have enough space in our internal buffer for the result
        if self.output.capacity() < stop {
            debug!("REALLOC: {} -> {}", self.output.capacity(), stop);
            let additional_capacity = stop - self.output.capacity() + 1;
            self.output.reserve(additional_capacity);
        }

        // reset the length of the buffer so we can push elements into it
        unsafe { self.output.set_len(0); }

        let n_vals = (0..stop).step_by(decimation*2).collect::<Vec<_>>();

        dbg!(&n_vals);

        // get around immutable & mutable self borrows
        let taps = &self.taps;
        let use_avx = self.use_avx;

        self.output.extend(n_vals.into_iter().flat_map(|n| {
            let i_start = if (n/2) >= taps.len() { n - (2 * (taps.len()-1)) } else { 0 };
            let t_start = if (n/2) >= taps.len()-1 { 0 } else { (taps.len()-1) - (n/2)};

            debug!("{}: input[{}..={}] taps[{}..{}]", n, i_start, n+1, t_start, taps.len());

            let ret = if use_avx {
                unsafe { Filter::dot_product_avx(&input[i_start..=n+1], &taps[t_start..]) }
            } else {
                Filter::dot_product(&input[i_start..=n+1], &taps[t_start..])
            };

            vec![ret[0], ret[1]]
        }));

        let end_time = start_time.elapsed();

        debug!("FILTER TOOK: {}ms", end_time.as_secs()*1_000 + end_time.subsec_nanos() as u64 / 1_000_000);

        self.output.as_slice()
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use num::complex::Complex32;
    use std::io::{Read, Write, BufReader, BufWriter};
    use std::fs::{File, OpenOptions};
    use byteorder::{LE, BE, WriteBytesExt, ReadBytesExt};
    use std::sync::{Once, ONCE_INIT};

    static LOGGER_INIT: Once = ONCE_INIT;

    fn dot_product(len :usize) {
        let a = vec![1.9_f32, 2.8_f32].into_iter().cycle().take(len*2).collect::<Vec<_>>();
        let b = vec![4.6; len];

        let res_generic = Filter::dot_product(&a, &b);
        let res_avx = unsafe { Filter::dot_product_avx(&a, &b) };

        debug!("{:02}: Generic: {:?}\tAVX: {:?}", len, res_generic, res_avx);

        assert!((res_generic[0] - res_avx[0]).abs() < 0.1, "Real value difference too high for {}", len);
        assert!((res_generic[1] - res_avx[1]).abs() < 0.1, "Imag value difference too high for {}", len);
    }

    #[test]
    fn dot_product_avx() {
        LOGGER_INIT.call_once(|| simple_logger::init_with_level(log::Level::Debug).unwrap()); // this will panic on error

        // test a bunch of different sizes
        for i in 1..65 {
            dot_product(i);
        }
    }

    #[test]
    fn generate_low_pass_taps() {
        let taps = Filter::generate_low_pass_taps(1.0, 10e6, 100e3, 10e3);
        let mut taps_file = BufReader::new(OpenOptions::new().read(true).create(false).open("taps.dat").expect("Cannot open taps.dat file"));
        let mut my_file = BufWriter::new(OpenOptions::new().write(true).create(true).truncate(true).open("my_taps.dat").expect("Cannot open phase file"));

        taps.iter().for_each(|t| my_file.write_f32::<LE>(*t).unwrap());

        let mut i = 0;

        loop {
            let ftap = taps_file.read_f32::<LE>();

            if ftap.is_err() {
                break;
            }

            let ftap = ftap.unwrap();

            assert_eq!(taps[i], ftap);

            i += 1;
        }
    }

    #[test]
    fn low_pass_iq() {
        LOGGER_INIT.call_once(|| simple_logger::init_with_level(log::Level::Debug).unwrap()); // this will panic on error
        let mut in_file = BufReader::new(OpenOptions::new().read(true).create(false).open("iq_10000_low_pass.dat").expect("Cannot open iq_10000_low_pass.dat file"));
        let mut file_output = Vec::<f32>::new();

        loop {
            let r = in_file.read_f32::<LE>();

            if r.is_err() {
                break;
            }

            file_output.push(r.unwrap());
            file_output.push(in_file.read_f32::<LE>().unwrap());
        }

        let mut iq_file = BufReader::new(OpenOptions::new().read(true).create(false).open("iq_10000.dat").expect("Cannot open iq_1000.dat file"));
        let mut input = Vec::<f32>::new();

        loop {
            let r = iq_file.read_f32::<LE>();

            if r.is_err() {
                break;
            }

            input.push(r.unwrap());
            input.push(iq_file.read_f32::<LE>().unwrap());
        }

        assert_eq!(input.len(), 10_000);

        let taps = Filter::generate_low_pass_taps(1.0, 10e6, 100e3, 10e3);
        let mut filter = Filter::new(&taps);
        let output = filter.filter(20, &input);

        assert_eq!(output.len(), input.len() / 20);
        assert_eq!(output.len(), file_output.len());

        for i in 0..output.len() {
            assert!((output[i] - file_output[i]).abs() < 0.0001, "Mismatch at {}: {:0.20} != {:0.20}", i, output[i], file_output[i]);
        }

    }

    #[test]
    fn low_pass_ones() {
        let mut in_file = BufReader::new(OpenOptions::new().read(true).create(false).open("ones_low_pass.dat").expect("Cannot open file"));
        let mut file_output = Vec::<f32>::new();

        loop {
            let r = in_file.read_f32::<LE>();

            if r.is_err() {
                break;
            }

            file_output.push(r.unwrap());
            file_output.push(in_file.read_f32::<LE>().unwrap());
        }

        let input = vec![1.0_f32; 200];
        let taps = Filter::generate_low_pass_taps(1.0, 10e6, 100e3, 10e3);
        let mut filter = Filter::new(&taps);
        let output = filter.filter(1, &input);

        assert_eq!(output.len(), input.len());

        for i in 0..200 {
            assert!((output[i] - file_output[i]).abs() < 0.0001, "Mismatch at {}: {:0.20} != {:0.20}", i, output[i], file_output[i]);
        }
    }

    #[test]
    fn low_pass_decimation() {
        LOGGER_INIT.call_once(|| simple_logger::init_with_level(log::Level::Debug).unwrap()); // this will panic on error

        let input = vec![1.0_f32 ; 200];

        let taps = Filter::generate_low_pass_taps(1.0, 10e6, 100e3, 10e3);
        let mut filter = Filter::new(&taps);
        let output = filter.filter(20, &input);

        println!("TAPS: {}", taps.len());

        assert_eq!(output.len(), input.len() / 20);

        // these were taken from running 100 (1.0, 1.0) numbers through the GNURadio low-pass filter
        assert_eq!(output[0], 0.00000526320945937186);
        assert_eq!(output[1], 0.00000526320945937186);

        assert_eq!(output[2], -0.00015483508468605578);
        assert_eq!(output[3], -0.00015483508468605578);

        assert_eq!(output[4], -0.00056682265130802989);
        assert_eq!(output[5], -0.00056682265130802989);

        assert_eq!(output[6], -0.00065905821975320578);
        assert_eq!(output[7], -0.00065905821975320578);

        assert_eq!(output[8], -0.00025988230481743813);
        assert_eq!(output[9], -0.00025988230481743813);
    }

    #[test]
    fn low_pass_large_input() {
        let input = vec![1.0_f32; 5000];

        let taps = Filter::generate_low_pass_taps(1.0, 10e6, 100e3, 10e3);
        let mut filter = Filter::new(&taps);
        let output = filter.filter(1, &input);

        assert_eq!(output.len(), input.len());

        // these were taken from running 100 (1.0, 1.0) numbers through the GNURadio low-pass filter
        assert_eq!(output[0], 0.0000052632095);
        assert_eq!(output[1], 0.0000052632095);

        assert_eq!(output[2], 0.000009232274);
        assert_eq!(output[3], 0.000009232274);

        assert_eq!(output[4], 0.000011889417);
        assert_eq!(output[5], 0.000011889417);

//        assert_eq!(output[6], 0.000013222045);
//        assert_eq!(output[7], 0.000013222045);
//
//        assert_eq!(output[8], 0.000013222048);
//        assert_eq!(output[9], 0.000013222048);
//
//        assert_eq!(output[4992], 1.0000006);
//        assert_eq!(output[4993], 1.0000006);
//
//        assert_eq!(output[4994], 1.0000006);
//        assert_eq!(output[4995], 1.0000006);
//
//        assert_eq!(output[4996], 1.0000006);
//        assert_eq!(output[4997], 1.0000006);
//
//        assert_eq!(output[4998], 1.0000006);
//        assert_eq!(output[4999], 1.0000006);
    }

    #[test]
    fn low_pass_large_input_decimation() {
        let input = vec![1.0_f32 ; 5000];

        let taps = Filter::generate_low_pass_taps(1.0, 10e6, 100e3, 10e3);
        let mut filter = Filter::new(&taps);
        let output = filter.filter(20, &input);

        assert_eq!(output.len(), input.len() / 20);

        // these were taken from running 100 (1.0, 1.0) numbers through the GNURadio low-pass filter
        assert_eq!(output[0], 0.0000052632095);
        assert_eq!(output[0], 0.0000052632095);

        assert_eq!(output[1], 0.0000052632095);
        assert_eq!(output[1], 0.0000052632095);

        assert_eq!(output[2], -0.00015483508);
        assert_eq!(output[2], -0.00015483508);

//        assert_eq!(output[3], 0.00082254934);
//        assert_eq!(output[3], 0.00082254934);
//
//        assert_eq!(output[4], 0.0013695889);
//        assert_eq!(output[4], 0.0013695889);
//
//        assert_eq!(output[38], 1.0027326);
//        assert_eq!(output[39], 1.0027326);
//
//        assert_eq!(output[40], 1.0017551);
//        assert_eq!(output[41], 1.0017551);
//
//        assert_eq!(output[42], 1.0011041);
//        assert_eq!(output[43], 1.0011041);
//
//        assert_eq!(output[44], 1.0007218);
//        assert_eq!(output[45], 1.0007218);
//
//        assert_eq!(output[46], 1.0005318);
//        assert_eq!(output[47], 1.0005318);
    }
}