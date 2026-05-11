import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pywt
from Crypto.Cipher import AES


@dataclass
class StegoResult:
    stego_image: np.ndarray
    encrypted_secret_shape: tuple
    encrypted_secret_len: int
    nonce: bytes


def _read_image(path: str, as_gray: bool = False, size=None) -> np.ndarray:
    if as_gray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def _derive_aes_key(password: str, key_file_path: Optional[str]) -> bytes:
    pass_bytes = password.encode("utf-8")
    if key_file_path:
        key_file_data = Path(key_file_path).read_bytes()
    else:
        key_file_data = b""
    return hashlib.sha256(pass_bytes + key_file_data).digest()


def encrypt_secret_image(secret_image: np.ndarray, password: str, key_file_path: Optional[str]):
    key = _derive_aes_key(password, key_file_path)
    cipher = AES.new(key, AES.MODE_CTR)
    data = secret_image.astype(np.uint8).tobytes()
    encrypted = cipher.encrypt(data)
    encrypted_arr = np.frombuffer(encrypted, dtype=np.uint8).reshape(secret_image.shape)
    return encrypted_arr, cipher.nonce


def decrypt_secret_image(
    encrypted_image: np.ndarray, password: str, nonce: bytes, key_file_path: Optional[str]
) -> np.ndarray:
    key = _derive_aes_key(password, key_file_path)
    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
    data = encrypted_image.astype(np.uint8).tobytes()
    decrypted = cipher.decrypt(data)
    dec_arr = np.frombuffer(decrypted, dtype=np.uint8).reshape(encrypted_image.shape)
    return dec_arr

def encrypt_text(secret_text: str, password: str, key_file_path: Optional[str]):
    key = _derive_aes_key(password, key_file_path)
    cipher = AES.new(key, AES.MODE_CTR)
    data = secret_text.encode('utf-8')
    encrypted = cipher.encrypt(data)
    enc_shape = (len(encrypted), 1)
    if len(encrypted) == 0:
        # Dummy shape if empty
        enc_shape = (1, 1)
        encrypted = b'\x00'
    encrypted_arr = np.frombuffer(encrypted, dtype=np.uint8).reshape(enc_shape)
    return encrypted_arr, cipher.nonce

def decrypt_text(
    encrypted_arr: np.ndarray, password: str, nonce: bytes, key_file_path: Optional[str]
) -> str:
    key = _derive_aes_key(password, key_file_path)
    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
    data = encrypted_arr.astype(np.uint8).tobytes()
    decrypted = cipher.decrypt(data)
    return decrypted.decode('utf-8', errors='replace')



def _pack_header(secret_shape: tuple, nonce: bytes) -> bytes:
    if len(secret_shape) == 2:
        h, w = secret_shape
        c = 1
    else:
        h, w, c = secret_shape
    header = np.array([h, w, c, len(nonce)], dtype=np.uint16).tobytes() + nonce
    return header


def _unpack_header(raw: bytes):
    h, w, c, nonce_len = np.frombuffer(raw[:8], dtype=np.uint16)
    nonce = raw[8:8 + nonce_len]
    return int(h), int(w), int(c), nonce


def _to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def _from_bits(bits: np.ndarray) -> bytes:
    return np.packbits(bits.astype(np.uint8)).tobytes()


def embed_dwt_svd(cover: np.ndarray, encrypted_secret: np.ndarray, nonce: bytes, alpha: float = 0.08) -> StegoResult:
    cA, (cH, cV, cD) = pywt.dwt2(cover.astype(np.float32), "haar")
    u_c, s_c, vt_c = np.linalg.svd(cA, full_matrices=False)

    # 1. Embed structural watermark into SVD LL band
    s_mod = s_c.copy()
    mark_bits = _to_bits(nonce)
    s_mod[:mark_bits.size] += (alpha * 500.0) * np.where(mark_bits == 1, 1.0, -1.0)
    cA_stego = u_c @ np.diag(s_mod) @ vt_c

    # 2. Embed header and payload into robust DWT high-frequency bands (cH, cV, cD)
    payload_bytes = _pack_header(encrypted_secret.shape, nonce) + encrypted_secret.astype(np.uint8).tobytes()
    payload_bits = _to_bits(payload_bytes)

    h_shape = cH.shape
    ac_flat = np.concatenate([cH.flatten(), cV.flatten(), cD.flatten()])

    if payload_bits.size > ac_flat.size:
        raise ValueError(f"Payload too large. Need {payload_bits.size} bits, got {ac_flat.size} AC coefficients.")

    ac_alpha = alpha * 40.0  # High resilience for integer casting
    ac_flat[:payload_bits.size] += ac_alpha * np.where(payload_bits == 1, 1.0, -1.0)

    sz = cH.size
    cH_s = ac_flat[0:sz].reshape(h_shape)
    cV_s = ac_flat[sz:2*sz].reshape(h_shape)
    cD_s = ac_flat[2*sz:3*sz].reshape(h_shape)

    stego = pywt.idwt2((cA_stego, (cH_s, cV_s, cD_s)), "haar")
    stego = np.clip(stego, 0, 255).astype(np.uint8)

    return StegoResult(stego, encrypted_secret.shape, encrypted_secret.size, nonce)


def extract_dwt_svd(
    cover: np.ndarray,
    stego: np.ndarray,
    alpha: float = 0.08,
    header_base_bytes: int = 8,
):
    cA_c, (cH_c, cV_c, cD_c) = pywt.dwt2(cover.astype(np.float32), "haar")
    cA_s, (cH_s, cV_s, cD_s) = pywt.dwt2(stego.astype(np.float32), "haar")

    ac_cover = np.concatenate([cH_c.flatten(), cV_c.flatten(), cD_c.flatten()])
    ac_stego = np.concatenate([cH_s.flatten(), cV_s.flatten(), cD_s.flatten()])
    delta_ac = ac_stego - ac_cover

    # Extract base header first (8 bytes = 64 bits)
    base_bits = (delta_ac[:64] > 0).astype(np.uint8)
    base_header = _from_bits(base_bits)
    h, w, c, nonce_len = np.frombuffer(base_header[:8], dtype=np.uint16)
    h, w, c, nonce_len = int(h), int(w), int(c), int(nonce_len)

    if h == 0 or w == 0 or nonce_len > 256 or c == 0:
        raise ValueError("Extraction failed: Garbage header decoded. The image might be heavily corrupted or cover mismatch.")

    # Extract full header
    full_header_bits = (header_base_bytes + nonce_len) * 8
    full_header_bits_arr = (delta_ac[:full_header_bits] > 0).astype(np.uint8)
    full_header = _from_bits(full_header_bits_arr)
    h, w, c, nonce = _unpack_header(full_header)

    # Extract full payload
    secret_len = h * w * c
    payload_bits_count = full_header_bits + secret_len * 8

    bitstream = (delta_ac[:payload_bits_count] > 0).astype(np.uint8)
    payload_bytes = _from_bits(bitstream)

    encrypted_secret_bytes = payload_bytes[(header_base_bytes + nonce_len):]
    if c == 1:
        encrypted_secret = np.frombuffer(encrypted_secret_bytes, dtype=np.uint8).reshape((h, w))
    else:
        encrypted_secret = np.frombuffer(encrypted_secret_bytes, dtype=np.uint8).reshape((h, w, c))
    return encrypted_secret, nonce


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return float(10 * np.log10((255.0 ** 2) / m))


def bcr(a: np.ndarray, b: np.ndarray) -> float:
    bits_a = np.unpackbits(a.astype(np.uint8).flatten())
    bits_b = np.unpackbits(b.astype(np.uint8).flatten())
    n = min(bits_a.size, bits_b.size)
    return float(np.mean(bits_a[:n] == bits_b[:n]) * 100.0)


def run_pipeline(cover_path: str, secret_path: str, out_stego: str, out_recovered: str, password: str, alpha: float, as_gray: bool = False):
    cover = _read_image(cover_path, as_gray=as_gray)
    c_h, c_w = cover.shape[:2]
    
    max_payload_bytes = (3 * (c_h // 2) * (c_w // 2)) // 8
    secret = _read_image(secret_path, as_gray=as_gray)
    
    req_bytes = secret.size
    if req_bytes > max_payload_bytes:
        import math
        scale = math.sqrt(max_payload_bytes / float(req_bytes))
        new_w = max(1, int(secret.shape[1] * scale))
        new_h = max(1, int(secret.shape[0] * scale))
        while (new_w * new_h * (3 if not as_gray else 1)) > max_payload_bytes:
            if new_w > new_h:
                new_w -= 1
            else:
                new_h -= 1
        secret = cv2.resize(secret, (max(1, new_w), max(1, new_h)), interpolation=cv2.INTER_AREA)

    encrypted_secret, nonce = encrypt_secret_image(secret, password, key_file_path=None)
    
    if len(cover.shape) == 3:
        cover_ycrcb = cv2.cvtColor(cover, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(cover_ycrcb)
        result = embed_dwt_svd(y, encrypted_secret, nonce, alpha=alpha)
        stego_y = result.stego_image
        stego_img = cv2.cvtColor(cv2.merge((stego_y, cr, cb)), cv2.COLOR_YCrCb2BGR)
    else:
        result = embed_dwt_svd(cover, encrypted_secret, nonce, alpha=alpha)
        stego_img = result.stego_image
        
    cv2.imwrite(out_stego, stego_img)

    if len(stego_img.shape) == 3:
        stego_y = cv2.split(cv2.cvtColor(stego_img, cv2.COLOR_BGR2YCrCb))[0]
        cover_y = cv2.split(cv2.cvtColor(cover, cv2.COLOR_BGR2YCrCb))[0]
        extracted_enc_secret, ext_nonce = extract_dwt_svd(cover_y, stego_y, alpha=alpha)
    else:
        extracted_enc_secret, ext_nonce = extract_dwt_svd(cover, stego_img, alpha=alpha)

    recovered_secret = decrypt_secret_image(extracted_enc_secret, password, ext_nonce, key_file_path=None)
    cv2.imwrite(out_recovered, recovered_secret)

    print("=== Metrics ===")
    print(f"Cover vs Stego MSE:  {mse(cover, stego_img):.6f}")
    if mse(cover, stego_img) > 0:
        print(f"Cover vs Stego PSNR: {psnr(cover, stego_img):.4f} dB")
    else:
        print("Cover vs Stego PSNR: inf dB")

    # Safeguard shapes due to possible header corruption bitflips
    s_h = min(secret.shape[0], recovered_secret.shape[0])
    s_w = min(secret.shape[1], recovered_secret.shape[1])
    
    if len(secret.shape) == len(recovered_secret.shape):
        if len(secret.shape) == 3:
            s1 = secret[:s_h, :s_w, :]
            s2 = recovered_secret[:s_h, :s_w, :]
        else:
            s1 = secret[:s_h, :s_w]
            s2 = recovered_secret[:s_h, :s_w]
            
        print(f"Secret vs Recovered BCR: {bcr(s1, s2):.4f}%")
        print(f"Secret vs Recovered PSNR: {psnr(s1, s2):.4f} dB")
    else:
        print("Secret vs Recovered BCR: Shape mismatch (Header corrupted)")
        print("Secret vs Recovered PSNR: Shape mismatch (Header corrupted)")

    print(f"Output stego: {out_stego}")
    print(f"Output recovered secret: {out_recovered}")


def embed_only(
    cover_path: str,
    secret_path: str,
    out_stego: str,
    password: str,
    key_file_path: str,
    alpha: float,
    payload_type: str = "image",
    secret_text: str = "",
    as_gray: bool = False
):
    cover = _read_image(cover_path, as_gray=as_gray)
    c_h, c_w = cover.shape[:2]

    # Calculate actual maximum capacity mapping to our Hybrid DWT-SVD
    s_c_size = min(c_h // 2, c_w // 2)
    ac_capacity_bytes = (3 * (c_h // 2) * (c_w // 2)) // 8
    max_payload_bytes = ac_capacity_bytes

    if s_c_size < 176:
        raise ValueError(f"Cover image is too incredibly small. Try a normal image.")

    if payload_type == "text":
        orig_shape = None
        if len(secret_text.encode('utf-8')) > max_payload_bytes:
            raise ValueError(f"Text too long! Max capacity is {max_payload_bytes} bytes.")
        encrypted_secret, nonce = encrypt_text(secret_text, password, key_file_path=key_file_path)
    else:
        # Secret is image
        secret = _read_image(secret_path, as_gray=as_gray)
        orig_shape = secret.shape
        req_bytes = secret.size
        if req_bytes > max_payload_bytes:
            import math
            scale = math.sqrt(max_payload_bytes / float(req_bytes))
            new_w = max(1, int(secret.shape[1] * scale))
            new_h = max(1, int(secret.shape[0] * scale))
            
            # Failsafe loop to guarantee math exactness
            while (new_w * new_h * (3 if not as_gray else 1)) > max_payload_bytes:
                if new_w > new_h:
                    new_w -= 1
                else:
                    new_h -= 1
                
            secret = cv2.resize(secret, (max(1, new_w), max(1, new_h)), interpolation=cv2.INTER_AREA)

        encrypted_secret, nonce = encrypt_secret_image(secret, password, key_file_path=key_file_path)

    if len(cover.shape) == 3:
        cover_ycrcb = cv2.cvtColor(cover, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(cover_ycrcb)
        result = embed_dwt_svd(y, encrypted_secret, nonce, alpha=alpha)
        stego_img = cv2.cvtColor(cv2.merge((result.stego_image, cr, cb)), cv2.COLOR_YCrCb2BGR)
    else:
        result = embed_dwt_svd(cover, encrypted_secret, nonce, alpha=alpha)
        stego_img = result.stego_image

    cv2.imwrite(out_stego, stego_img)
    return {
        "mse": mse(cover, stego_img),
        "psnr": psnr(cover, stego_img),
        "capacity_bytes": max_payload_bytes,
        "used_bytes": encrypted_secret.size,
        "original_shape": orig_shape,
        "resized_shape": secret.shape if payload_type != "text" else None
    }


def extract_only(
    cover_path: str,
    stego_path: str,
    out_recovered: str,
    password: str,
    key_file_path: str,
    alpha: float,
    payload_type: str = "image",
    as_gray: bool = False
):
    cover = _read_image(cover_path, as_gray=as_gray)
    stego = _read_image(stego_path, as_gray=as_gray)
    
    if len(cover.shape) == 3:
        stego_y = cv2.split(cv2.cvtColor(stego, cv2.COLOR_BGR2YCrCb))[0]
        cover_y = cv2.split(cv2.cvtColor(cover, cv2.COLOR_BGR2YCrCb))[0]
        extracted_enc_secret, ext_nonce = extract_dwt_svd(cover_y, stego_y, alpha=alpha)
    else:
        extracted_enc_secret, ext_nonce = extract_dwt_svd(cover, stego, alpha=alpha)

    if payload_type == "text":
        return decrypt_text(extracted_enc_secret, password, ext_nonce, key_file_path=key_file_path)
    else:
        recovered_secret = decrypt_secret_image(
            extracted_enc_secret,
            password,
            ext_nonce,
            key_file_path=key_file_path,
        )
        cv2.imwrite(out_recovered, recovered_secret)
        return out_recovered


def main():
    parser = argparse.ArgumentParser(description="AES + DWT-SVD image steganography.")
    sub = parser.add_subparsers(dest="cmd", required=False)

    embed_parser = sub.add_parser("embed", help="Embed encrypted secret into cover image.")
    embed_parser.add_argument("--cover", required=True, help="Path to cover image.")
    embed_parser.add_argument("--secret", required=True, help="Path to secret image.")
    embed_parser.add_argument("--stego-out", default="stego.png", help="Output stego image path.")
    embed_parser.add_argument("--password", required=True, help="Password.")
    embed_parser.add_argument("--key-file", required=True, help="Path to key file.")
    embed_parser.add_argument("--alpha", type=float, default=0.08, help="Embedding strength.")
    embed_parser.add_argument("--as-gray", action="store_true", help="Apply grayscale conversion.")

    extract_parser = sub.add_parser("extract", help="Extract and decrypt hidden image from stego image.")
    extract_parser.add_argument("--cover", required=True, help="Path to original cover image.")
    extract_parser.add_argument("--stego", required=True, help="Path to stego image.")
    extract_parser.add_argument("--recovered-out", default="recovered_secret.png", help="Recovered image path.")
    extract_parser.add_argument("--password", required=True, help="Password.")
    extract_parser.add_argument("--key-file", required=True, help="Path to key file.")
    extract_parser.add_argument("--alpha", type=float, default=0.08, help="Embedding strength.")
    extract_parser.add_argument("--as-gray", action="store_true", help="Apply grayscale conversion.")

    # Backward-compatible all-in-one mode.
    parser.add_argument("--cover", help="Path to cover image.")
    parser.add_argument("--secret", help="Path to secret image.")
    parser.add_argument("--stego-out", default="stego.png", help="Output stego image path.")
    parser.add_argument("--recovered-out", default="recovered_secret.png", help="Output recovered secret image path.")
    parser.add_argument("--password", help="Password used to derive AES key.")
    parser.add_argument("--alpha", type=float, default=0.08, help="Embedding strength.")
    parser.add_argument("--as-gray", action="store_true", help="Apply grayscale conversion.")
    args = parser.parse_args()

    if args.cmd == "embed":
        out = embed_only(
            cover_path=args.cover,
            secret_path=args.secret,
            out_stego=args.stego_out,
            password=args.password,
            key_file_path=args.key_file,
            alpha=args.alpha,
            as_gray=args.as_gray,
        )
        print("Embed complete.")
        print(f"Cover vs Stego MSE:  {out['mse']:.6f}")
        print(f"Cover vs Stego PSNR: {out['psnr']:.4f} dB")
        print(f"Output stego: {args.stego_out}")
    elif args.cmd == "extract":
        extract_only(
            cover_path=args.cover,
            stego_path=args.stego,
            out_recovered=args.recovered_out,
            password=args.password,
            key_file_path=args.key_file,
            alpha=args.alpha,
            as_gray=args.as_gray,
        )
        print("Extract complete.")
        print(f"Recovered image: {args.recovered_out}")
    else:
        if not args.cover or not args.secret or not args.password:
            parser.error("For default mode, provide --cover --secret --password.")
        run_pipeline(
            cover_path=args.cover,
            secret_path=args.secret,
            out_stego=args.stego_out,
            out_recovered=args.recovered_out,
            password=args.password,
            alpha=args.alpha,
            as_gray=args.as_gray,
        )


if __name__ == "__main__":
    main()
