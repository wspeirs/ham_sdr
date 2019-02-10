#[macro_use]
extern crate log;

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;
use std::thread::sleep;
use std::io::{Write, Read, stdout, BufWriter, BufReader};
use std::fs::{File, OpenOptions};

use byteorder::{LE, ReadBytesExt, WriteBytesExt};

use std::f32::consts::PI;

use rs_libhackrf::hackrf::HackRF;
use rs_libhackrf::error::Error;

mod modulation;
mod filter;

use modulation::QuadratureDemodulator;
use filter::Filter;

const FREQ :u64 = 95_900_000; // set to 95.9MHz
const SAMPLE_RATE :f64 = 10_000_000.0;

fn main() -> Result<(), Error> {
    simple_logger::init_with_level(log::Level::Debug).unwrap();
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

    let mut test_file = BufWriter::new(OpenOptions::new().write(true).create(true).truncate(true).open("test.dat").expect("Cannot open phase file"));
    let mut iq_file = BufReader::new(OpenOptions::new().read(true).create(false).open("iq.dat").expect("Cannot open IQ file"));

//    let mut fm_demod = QuadratureDemodulator::new();
    let taps = Filter::generate_low_pass_taps(1.0, 10e6, 100e3, 10e3);
    let mut filter = Filter::with_capacity(&taps, 131_100);

    // test for reading from file
    loop {
        let mut iq = Vec::<f32>::with_capacity(262_144);

        for _ in 0..262_144 {
            iq.push(iq_file.read_f32::<LE>().unwrap());
        }

        let output = filter.filter(20, &iq);

        assert_eq!(output.len()%2, 0, "Output not event");

        output.iter().for_each(|f| {
            test_file.write_f32::<LE>(*f);
        });

        test_file.flush();
    }

/*
    dev.start_rx(|iq| {

//        iq.iter().for_each(|b| {
//            test_file.write_f32::<LE>(b.re);
//            test_file.write_f32::<LE>(b.im);
//        });

        //
        // low-pass filter, and decimation
        //
        let output = filter.filter(20, iq);

        output.iter().for_each(|c| {
            test_file.write_f32::<LE>(c.re);
            test_file.write_f32::<LE>(c.im);
        });

        //
        // demodulation
        //
//        let res = fm_demod.demodulate(&buff);
//
//        res.iter().for_each(|f| {
//            test_file.write_f32::<LE>(*f);
//        });
//
//        test_file.flush();

        Error::SUCCESS
    })?;

    sleep(Duration::from_secs(5));
    dev.stop_rx();
    sleep(Duration::from_millis(10));

*/
    test_file.flush();
    Ok(())
}


#[cfg(test)]
mod test {
    use super::*;
    use std::io::BufReader;
    use byteorder::{LE, ReadBytesExt};
    use std::sync::{Once, ONCE_INIT};

    static LOGGER_INIT: Once = ONCE_INIT;

    #[test]
    fn test_read() {
        LOGGER_INIT.call_once(|| simple_logger::init_with_level(log::Level::Debug).unwrap()); // this will panic on error

        let mut my_file = BufReader::new(OpenOptions::new().read(true).create(false).open("test.dat").expect("Cannot open phase file"));
        let mut gr_file = BufReader::new(OpenOptions::new().read(true).create(false).open("gnu_radio.dat").expect("Cannot open phase file"));

        for _ in 0..100 {
            debug!("{:0.04} {:0.04}\t\t{:0.04} {:0.04}", my_file.read_f32::<LE>().unwrap(), my_file.read_f32::<LE>().unwrap(), gr_file.read_f32::<LE>().unwrap(), gr_file.read_f32::<LE>().unwrap());
        }
    }

    #[test]
    fn test_lut() {
        LOGGER_INIT.call_once(|| simple_logger::init_with_level(log::Level::Debug).unwrap()); // this will panic on error
        let buff :Vec<u8> = vec![0x00, 0x01, 0x02, 0x03];

        for i in buff.chunks(2) {
            let mut s = i[1] as u16;

            s <<= 8;
            s += i[0] as u16;

            debug!("{:x}", s);
        }

    }

}

