#[macro_use]
extern crate log;

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;
use std::thread::sleep;
use std::io::{Write, stdout, BufWriter};
use std::fs::{File, OpenOptions};

use byteorder::{LE, BE, WriteBytesExt};

use std::f32::consts::PI;

use num::complex::Complex32;

use sdr::fir::FIR;
use sdr::fm::FMDemod;

use rs_libhackrf::hackrf::HackRF;
use rs_libhackrf::error::Error;

mod dsp;

use dsp::QuadratureDemodulator;

const FREQ :u64 = 95_900_000; // set to 95.9MHz
const SAMPLE_RATE :f64 = 10_000_000.0;

fn main() -> Result<(), Error> {
//    simple_logger::init_with_level(log::Level::Debug).unwrap();
//    simple_logger::init_with_level(log::Level::Info).unwrap();

    let mut hrf = HackRF::new().unwrap();
    let mut dev = hrf.open_device(0).expect("Error opening device; not plugged in?");

    dev.set_freq(FREQ)?;
    dev.set_sample_rate(SAMPLE_RATE)?;

    dev.set_vga_gain(15)?;
    dev.set_lna_gain(20)?;

    dev.set_amp_enable(true);
    dev.set_antenna_enable(true);

    let bb_bw = dev.compute_baseband_filter_bandwidth((0.75 * SAMPLE_RATE) as u32);
    dev.set_baseband_filter_bandwidth(bb_bw);

    println!("BB BW: {}", bb_bw);

    let mut test_file = BufWriter::new(OpenOptions::new().write(true).create(true).open("test.dat").expect("Cannot open phase file"));

//    let mut low_pass_fir :FIR<Complex32> = FIR::lowpass(2409, 0.02); // cut off / (sample/2)
//    let mut decimation_fir = FIR::new(low_pass_fir.taps(), 20, 1);
//    let mut resample_fir = FIR::resampler(4103, 500, 96);

    let mut fm_demod = QuadratureDemodulator::new();

    dev.start_rx(|iq| {

        iq.iter().for_each(|b| {
            test_file.write_f32::<LE>(b.re);
            test_file.write_f32::<LE>(b.im);
        });

        //
        // low-pass filter, and decimation
        //
//        let buff = decimation_fir.process(&buff);

//        buff.iter().for_each(|c| {
//            test_file.write_f32::<LE>(c.re);
//            test_file.write_f32::<LE>(c.im);
//        });
//
//        test_file.flush();

//        // demodulation
//        let res = fm_demod.demodulate(&buff);
//
//        res.iter().for_each(|f| {
//            test_file.write_f32::<LE>(*f);
//        });
//
//        test_file.flush();

        // re-sample
//        let res = resample_fir.process(&res);
//
//        res.iter().for_each(|r| {
//            test_file.write_f32::<LE>(*r);
//        });
//        test_file.flush();

        Error::SUCCESS
    })?;

    sleep(Duration::from_secs(5));
    dev.stop_rx();
    sleep(Duration::from_millis(10));

    test_file.flush();

    Ok(())
}


#[cfg(test)]
mod test {
    use super::*;
    use std::io::BufReader;
    use byteorder::{LE, ReadBytesExt};

    #[test]
    fn test_read() {
        let mut my_file = BufReader::new(OpenOptions::new().read(true).create(false).open("test.dat").expect("Cannot open phase file"));
        let mut gr_file = BufReader::new(OpenOptions::new().read(true).create(false).open("gnu_radio.dat").expect("Cannot open phase file"));

        for _ in 0..100 {
            println!("{:0.04} {:0.04}\t\t{:0.04} {:0.04}", my_file.read_f32::<LE>().unwrap(), my_file.read_f32::<LE>().unwrap(), gr_file.read_f32::<LE>().unwrap(), gr_file.read_f32::<LE>().unwrap());
        }
    }

    #[test]
    fn test_lut() {
        let buff :Vec<u8> = vec![0x00, 0x01, 0x02, 0x03];

        for i in buff.chunks(2) {
            let mut s = i[1] as u16;

            s <<= 8;
            s += i[0] as u16;

            println!("{:x}", s);
        }

    }

}

