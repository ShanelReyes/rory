import argparse
import os
from enum import Enum
from rory.core.security.cryptosystem.pqc.ckks import Ckks,CkksModes
# Assuming you import your Ckks class here
# from your_module import Ckks


def main():
    parser = argparse.ArgumentParser(description="Generate and save Pyfhel CKKS keys.")

    # String / Integer arguments
    parser.add_argument(
        "--scheme", 
        type=str, 
        default="CKKS", 
        help="Encryption scheme to use (default: CKKS)"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=[e.value for e in CkksModes], 
        default=CkksModes.DEFAULT.value, 
        help="Mode of operation (default: default)"
    )
    parser.add_argument(
        "--security-level", 
        type=int, 
        default=128, 
        help="Security level e.g., 128, 192, 256 (default: 128)"
    )
    parser.add_argument(
        "--decimals", 
        type=int, 
        default=5, 
        help="Number of decimals to preserve (default: 5)"
    )
    parser.add_argument(
        "--output-path", 
        type=str, 
        required=True, 
        help="Directory path to save the generated keys"
    )

    # Boolean flags (Store True if the flag is passed)
    parser.add_argument(
        "--round", 
        action="store_true", 
        help="Enable rounding"
    )
    parser.add_argument(
        "--no-save", 
        action="store_false", 
        dest="save",
        help="Do NOT save the keys to disk (by default, keys are saved)"
    )
    parser.add_argument(
        "--enable-relinearize", 
        action="store_true", 
        help="Generate relinearization keys"
    )
    parser.add_argument(
        "--enable-rotate", 
        action="store_true", 
        help="Generate rotation keys"
    )

    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    # Convert the string mode back to the Enum
    mode_enum = CkksModes(args.mode)

    print(f"Generating keys with security level {args.security_level} in {args.output_path}...")

    # Initialize the client
    # Uncomment this when running with your actual Ckks class
    import time as T
    t1 = T.time()
    ckks = Ckks.create_client(
        scheme             = args.scheme,
        mode               = mode_enum,
        security_level     = args.security_level,
        _round             = args.round,
        decimals           = args.decimals,
        output_path        = args.output_path,
        save               = args.save,
        enable_relinearize = args.enable_relinearize,
        enable_rotate      = args.enable_rotate,
    )
    st = T.time() - t1
    
    if ckks is not None:
       print(f"Completado {args.security_level} bits en {st:.2f} segundos.")
    else:
        print("Failed to generate keys.")

if __name__ == "__main__":
    main()