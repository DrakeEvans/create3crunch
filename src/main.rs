use alloy_primitives::{Address, FixedBytes};
use clap::Parser;
use clap_num::maybe_hex;
use create3crunch::{gpu, Config};

fn parse_worksize(s: &str) -> Result<u32, String> {
    let work_size = maybe_hex::<u32>(s)?;
    if work_size < 0x1540000 {
        return Err("Work size cannot be below 0x15400000".to_string());
    }
    Ok(work_size)
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, help = "Address of the CREATE3 factory contract")]
    factory: Address,

    #[arg(
        short,
        long,
        help = "Owner / Caller address (first 20-bytes of the top-level salt will be set to the address)"
    )]
    owner: Address,

    #[arg(short, long, help = "Hash of the factory's deploy proxy initcode")]
    initcode_hash: FixedBytes<32>,

    #[arg(short, long, help = "GPU Device")]
    gpu_device: u8,

    #[arg(
        short,
        long,
        help = "Minimum amount of leading zeros for the solution to be considered valuable"
    )]
    leading_zeros: Option<u8>,

    #[arg(
        short,
        long,
        help = "Minimum amount of total zero bytes for the address to be consiered valuable"
    )]
    total_zeros: Option<u8>,

    #[arg(
        short,
        long,
        help = "Specifies the upper bound for the nonces that will be inclusively checked (1 - n), at most 127",
        default_value_t = 32
    )]
    max_create3_nonce: u8,

    #[arg(short, long, value_parser=parse_worksize, default_value_t=0x4000000)]
    work_size: u32,
}

impl TryInto<Config> for Args {
    type Error = String;

    fn try_into(self) -> Result<Config, Self::Error> {
        if self.max_create3_nonce >= 0x80 {
            return Err(format!(
                "Create3 Nonces up to {} not supported",
                self.max_create3_nonce
            ));
        }
        if self.leading_zeros.is_none() && self.total_zeros.is_none() {
            return Err("Must specify at least either the total zeros or leading zeros threshold, cannot leave both empty".to_string());
        }
        Ok(Config {
            factory: self.factory,
            owner: self.owner,
            init_code_hash: self.initcode_hash,
            work_size: self.work_size,
            gpu_device: self.gpu_device,
            leading_zeroes_threshold: self.leading_zeros,
            total_zeroes_threshold: self.total_zeros,
            max_create3_nonce: self.max_create3_nonce,
        })
    }
}

fn main() {
    let args = Args::parse();

    gpu(args.try_into().unwrap()).unwrap()
}
