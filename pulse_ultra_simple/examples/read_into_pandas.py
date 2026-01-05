import numpy as np
import pandas as pd
import struct

HEADER_FMT = "<IIQ"   # magic:uint32, version:uint32, n:uint64
HEADER_SIZE = struct.calcsize(HEADER_FMT)
MAGIC = 0x31534C50    # 'PLS1'

def read_pulse_file(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        header = f.read(HEADER_SIZE)
        magic, version, n = struct.unpack(HEADER_FMT, header)

        if magic != MAGIC:
            raise ValueError("Not a pulse file")
        if version != 1:
            raise ValueError(f"Unsupported version {version}")

        # Memory-map the rest of the file
        mm = np.memmap(f, dtype=np.uint8, mode="r")

        offset = HEADER_SIZE
        acc_q = mm[offset : offset + n]
        offset += n

        # The remaining data is loss_q. No need to update offset after this.
        loss_q = mm[offset : offset + n]

        # Dequantize (vectorized, fast)
        df = pd.DataFrame({
            "accuracy": acc_q.astype(np.float32) / 10.0,
            "loss": loss_q.astype(np.float32) / 10.0,
        })

        return df

def main():
    df = read_pulse_file("pulse_simple.bin")
    print(df.head())
    print(df.tail())
    print("rows:", len(df))

if __name__ == "__main__":
    main()
