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
        
        # Map the file starting after the header
        mm = np.memmap(path, dtype=np.uint8, mode="r", offset=HEADER_SIZE)
        
        # Slicing based on the 'n' found in the header
        acc_q = mm[0 : n]
        loss_q = mm[n : 2 * n]

        return pd.DataFrame({
            "accuracy": acc_q.astype(np.float32) / 10.0,
            "loss": loss_q.astype(np.float32) / 10.0,
        })

def main():
    fpath = "pulse_log.bin"
    try:
        df = read_pulse_file(fpath)
        print("First 5 records:")
        print(df.head())
        print("\nLast 5 records:")
        print(df.tail())
        print(f"\nTotal rows: {len(df)}")
    except FileNotFoundError:
        print(f"Error: {fpath} not found. Run the simulation first.")
    except Exception as e:
        print(f"Error loading file: {e}")

if __name__ == "__main__":
    main()