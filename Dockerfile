FROM nvidia/opencl:latest

RUN apt-get update && apt-get install -y curl build-essential ocl-icd-opencl-dev git pkg-config libssl-dev

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

# RUN git clone https://github.com/DrakeEvans/create3crunch.git
COPY . .

RUN cargo build --release

CMD ["./target/release/create3crunch","--factory","000000000000b361194cfe6312EE3210d53C15AA","--owner","276CE87E04b7a74A69e2898689D5F5f55623dCa4","--initcode-hash","1decbcf04b355d500cbc3bd83c892545b4df34bd5b2c9d91b9f7f8165e2095c3","-l","1","--gpu-device","2"]