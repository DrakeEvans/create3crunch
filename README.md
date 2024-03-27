# Create3 Crunch

> A Rust program for finding salts that create gas-efficient Ethereum addresses via CREATE3.

Unlike normal CREATE3, this miner allows you to test for multiple nonces in the "deploy proxy"
contract (the standard contract that gets deployed with create2 that eventually deploys your contract).
This allows the miner to approach CREATE2 in mining speeds as checking different nonces ammortizes
the initial fixed cost of computing the deploy proxy's address.

> [!CAUTION]
> Non-default (nonce = 1) nonces is not supported by the majority of CREATE3 libraries, set the
> max-nonce to `1` if you only want to search with `nonce = 1`, note this will degrade the
> performance of the miner.

## Installation Instructions

1. Install Rust
2. Clone repo
3. Build with `cargo build --release` (performance is mostly GPU bound so a debug build is probably
   fine too).
4. Run with `./target/release/create3crunch`.

## Usage

TODO, but in the meantime check the available options using `-h` or `--help`.
