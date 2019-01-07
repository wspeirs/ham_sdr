use std::time::Duration;

use libusb::{DeviceHandle,Direction,RequestType,Recipient};
use libusb::request_type;
use libusb::Error as USBError;

enum UsbRegister {
    SYSCTL = 0x2000,
    CTRL = 0x2010,
    STAT = 0x2014,
    EPA_CFG = 0x2144,
    EPA_CTL = 0x2148,
    EPA_MAXPKT = 0x2158,
    EPA_MAXPKT_2 = 0x215a,
    EPA_FIFO_CFG = 0x2160,
}

enum SystemRegister {
    DEMOD_CTL = 0x3000,
    GPO = 0x3001,
    GPI = 0x3002,
    GPOE = 0x3003,
    GPD = 0x3004,
    SYSINTE = 0x3005,
    SYSINTS = 0x3006,
    GP_CFG0 = 0x3007,
    GP_CFG1 = 0x3008,
    SYSINTE_1 = 0x3009,
    SYSINTS_1 = 0x300a,
    DEMOD_CTL_1 = 0x300b,
    IR_SUSPEND = 0x300c,
}

enum Block {
    DEMOD = 0,
    USB   = 1,
    SYS   = 2,
    TUN   = 3,
    ROM   = 4,
    IR    = 5,
    IIC   = 6,
}

fn write_register(dev: &DeviceHandle, block: Block, register: u16, value: &[u8]) -> Result<usize, USBError> {
    let control_out :u8 = request_type(Direction::Out, RequestType::Vendor, Recipient::Endpoint);
    let index :u16 = ((block as u16) << 8) | 0x10;

    let ret = dev.write_control(control_out, 0u8, register, index, value, Duration::from_millis(300));

    debug!("write_register: {:?}", ret);

    ret
}

fn demod_read_register(dev: &DeviceHandle, page: u8, addr: u16) -> Result<u16, USBError> {
    let control_in :u8 = request_type(Direction::In, RequestType::Vendor, Recipient::Endpoint);
    let mut buff = [0x00u8; 2];
    let addr = (addr << 8) | 0x20;

    let ret = dev.read_control(control_in, 0u8, addr, page as u16, &mut buff, Duration::from_millis(300));

    debug!("demod_read_register: {:?}", ret);

    Ok(((buff[1] as u16) << 8) | buff[0] as u16)
}

fn demod_write_register(dev: &DeviceHandle, page: u8, addr: u16, value: &[u8]) -> Result<usize, USBError> {
    let control_out :u8 = request_type(Direction::Out, RequestType::Vendor, Recipient::Endpoint);
    let index = (0x10 | page) as u16;
    let addr = (addr << 8) | 0x20;

    let ret = dev.write_control(control_out, 0u8, addr, index, value, Duration::from_millis(300));

    debug!("demod_write_register: {:?}", ret);

    demod_read_register(dev, 0x0A, 0x01)?;

    ret
}

fn write_array(dev: &DeviceHandle, block: Block, address: u16, value: &[u8]) -> Result<usize, USBError> {
    let control_out :u8 = request_type(Direction::Out, RequestType::Vendor, Recipient::Endpoint);
    let index = ((block as u16) << 8) | 0x10;

    let ret = dev.write_control(control_out, 0, address, index, value, Duration::from_millis(300));

    debug!("write_array: {:?}", ret);

    ret
}

fn read_array(dev: &DeviceHandle, block: Block, address: u16, data: &mut [u8]) -> Result<usize, USBError> {
    let control_in :u8 = request_type(Direction::In, RequestType::Vendor, Recipient::Endpoint);
    let index = (block as u16) << 8;

    let ret = dev.read_control(control_in, 0, address, index, data, Duration::from_millis(300));

    debug!("read_array: {:?}", ret);

    ret
}

fn i2c_read_register(dev: &DeviceHandle, address: u8, register: u8) -> Result<u8, USBError> {
    let mut buff = [0x0u8];

    write_array(dev, Block::IIC, address as u16, &[register])?;
    read_array(dev, Block::IIC, address as u16, &mut buff)?;

    debug!("i2c_read_register: {:?}", buff);

    Ok(buff[0])
}

/// Sets the finite impulse response (FIR) of the device
//fn set_fir(dev: &DeviceHandle) {
//    uint8_t fir[20];
//
//    int i;
//
//    // format: int8_t[8]
//    for i in 0..8 {
//        const int val = dev->fir[i];
//
//        if (val < -128 || val > 127) {
//            return -1;
//        }
//
//        fir[i] = val;
//    }
//
//    // format: int12_t[8]
//    for i in (0..8).step_by(2) {
//        const int val0 = dev->fir[8+i];
//        const int val1 = dev->fir[8+i+1];
//
//        if (val0 < -2048 || val0 > 2047 || val1 < -2048 || val1 > 2047) {
//            return -1;
//        }
//
//        fir[8+i*3/2] = val0 >> 4;
//        fir[8+i*3/2+1] = (val0 << 4) | ((val1 >> 8) & 0x0f);
//        fir[8+i*3/2+2] = val1;
//    }
//
//    for (i = 0; i < (int)sizeof(fir); i++) {
//        if (rtlsdr_demod_write_reg(dev, 1, 0x1c + i, fir[i], 1))
//        return -1;
//    }
//
//    return 0;
//}

/// Initialize the baseband of the device
pub fn init_baseband(dev: &mut DeviceHandle) -> Result<(), USBError> {
    // initialize USB
    write_register(dev, Block::USB, UsbRegister::SYSCTL as u16, &[0x09u8])?;
    write_register(dev, Block::USB, UsbRegister::EPA_MAXPKT as u16, &[0x00, 0x02])?;
    write_register(dev, Block::USB, UsbRegister::EPA_CTL as u16, &[0x10, 0x02])?;

    // power on demod
    write_register(dev, Block::SYS, SystemRegister::DEMOD_CTL_1 as u16, &[0x22])?;
    write_register(dev, Block::SYS, SystemRegister::DEMOD_CTL as u16, &[0xe8])?;

    // reset demod (bit 3, soft_rst)
    demod_write_register(dev, 1, 0x01, &[0x14])?;
    demod_write_register(dev, 1, 0x01, &[0x10])?;

    // disable spectrum inversion and adjacent channel rejection
    demod_write_register(dev, 1, 0x15, &[0x00])?;
    demod_write_register(dev, 1, 0x16, &[0x00, 0x00])?;

    // clear both DDC shift and IF frequency registers
    for i in 0..6 {
        demod_write_register(dev, 1, 0x16 + i, &[0x00])?;
    }

//    rtlsdr_set_fir(dev);

    // enable SDR mode, disable DAGC (bit 5)
    demod_write_register(dev, 0, 0x19, &[0x05])?;

    // init FSM state-holding register
    demod_write_register(dev, 1, 0x93, &[0xf0])?;
    demod_write_register(dev, 1, 0x94, &[0x0f])?;

//    /* disable AGC (en_dagc, bit 0) (this seems to have no effect) */
//    rtlsdr_demod_write_reg(dev, 1, 0x11, 0x00, 1);

    // disable RF and IF AGC loop
    demod_write_register(dev, 1, 0x04, &[0x00])?;

    // disable PID filter (enable_PID = 0)
    demod_write_register(dev, 0, 0x61, &[0x60])?;

    // opt_adc_iq = 0, default ADC_I/ADC_Q datapath
    demod_write_register(dev, 0, 0x06, &[0x80])?;

    // Enable Zero-IF mode (en_bbin bit), DC cancellation (en_dc_est),
    // IQ estimation/compensation (en_iq_comp, en_iq_est)
    demod_write_register(dev, 1, 0xb1, &[0x1b])?;

    // disable 4.096 MHz clock output on pin TP_CK0
    demod_write_register(dev, 0, 0x0d, &[0x83])?;

    // probe tuners
    demod_write_register(dev, 1, 0x01, &[0x18]);

    // check for E4K
    let reg = i2c_read_register(dev, 0xC8, 0x02);

    if let Ok(r) = reg {
        if r == 0x40 { println!("Got a E4K"); }
    }

    // check for FC0013
    let reg = i2c_read_register(dev, 0xC6, 0x00);

    if let Ok(r) = reg {
        if r == 0xA3 { println!("Got a FC0013"); }
    }

    // check for R820T
    let reg = i2c_read_register(dev, 0x34, 0x00);

    if let Ok(r) = reg {
        if r == 0x69 { println!("Got a R820T"); }
    }

    // check for R828D
    let reg = i2c_read_register(dev, 0x74, 0x00);

    if let Ok(r) = reg {
        if r == 0x69 { println!("Got a R828D"); }
    }


    Ok( () )
}