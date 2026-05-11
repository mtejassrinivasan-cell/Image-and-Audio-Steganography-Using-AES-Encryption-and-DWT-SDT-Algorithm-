import wave
import numpy as np
import os
import hashlib
from Crypto.Cipher import AES
from typing import Optional

def _derive_aes_key(password: str, key_file_path: Optional[str]) -> bytes:
    pass_bytes = password.encode("utf-8")
    if key_file_path and os.path.exists(key_file_path):
        with open(key_file_path, "rb") as f:
            key_file_data = f.read()
    else:
        key_file_data = b""
    return hashlib.sha256(pass_bytes + key_file_data).digest()

def _to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

def _from_bits(bits: np.ndarray) -> bytes:
    return np.packbits(bits.astype(np.uint8)).tobytes()

def embed_audio(cover_path: str, secret_path: str, out_stego: str, password: str, key_file_path: str = None):
    # Read secret audio exactly as bytes
    with open(secret_path, "rb") as f:
        secret_bytes = f.read()

    # Encrypt secret audio
    key = _derive_aes_key(password, key_file_path)
    cipher = AES.new(key, AES.MODE_CTR)
    encrypted_secret = cipher.encrypt(secret_bytes)
    nonce = cipher.nonce

    # Pack header: [secret_len (4 bytes)] [nonce_len (2 bytes)] [nonce] [encrypted_secret]
    secret_len = len(encrypted_secret)
    nonce_len = len(nonce)
    
    header = np.array([secret_len], dtype=np.uint32).tobytes() + \
             np.array([nonce_len], dtype=np.uint16).tobytes() + \
             nonce
             
    payload_bytes = header + encrypted_secret
    payload_bits = _to_bits(payload_bytes)

    # Read cover audio
    with wave.open(cover_path, 'rb') as cw:
        params = cw.getparams()
        nframes = params.nframes
        sampwidth = params.sampwidth
        nchannels = params.nchannels
        frames = cw.readframes(nframes)

    # audio data in bytes
    audio_bytes = np.frombuffer(frames, dtype=np.uint8).copy()

    # Total samples
    total_samples = nframes * nchannels

    if len(payload_bits) > total_samples:
        max_bytes = (total_samples // 8) - len(header)
        raise ValueError(f"Cover audio too small! Need {len(payload_bits)} bits but only have {total_samples} samples. Max payload is ~{max_bytes} bytes.")

    # We modify the LSB of the lowest byte of each sample.
    # In WAV (little-endian), the lowest byte of sample i is at index i * sampwidth
    indices = np.arange(total_samples) * sampwidth
    
    # Extract the lowest bytes
    lowest_bytes = audio_bytes[indices]
    
    # Modify LSBs
    # Clear LSB
    lowest_bytes[:len(payload_bits)] &= 254
    # Set LSB to payload bit
    lowest_bytes[:len(payload_bits)] |= payload_bits
    
    # Put back
    audio_bytes[indices] = lowest_bytes

    # Write stego audio
    with wave.open(out_stego, 'wb') as sw:
        sw.setparams(params)
        sw.writeframes(audio_bytes.tobytes())
        
    return {
        "capacity_bytes": total_samples // 8,
        "used_bytes": len(payload_bytes)
    }

def extract_audio(stego_path: str, out_recovered: str, password: str, key_file_path: str = None):
    with wave.open(stego_path, 'rb') as sw:
        params = sw.getparams()
        nframes = params.nframes
        sampwidth = params.sampwidth
        nchannels = params.nchannels
        frames = sw.readframes(nframes)
        
    audio_bytes = np.frombuffer(frames, dtype=np.uint8)
    total_samples = nframes * nchannels
    indices = np.arange(total_samples) * sampwidth
    lowest_bytes = audio_bytes[indices]
    
    # Extract all LSBs
    extracted_bits = lowest_bytes & 1
    
    # We first need to parse the header to know how many bits are there.
    # header has: secret_len (4 bytes) + nonce_len (2 bytes) = 6 bytes = 48 bits
    if len(extracted_bits) < 48:
        raise ValueError("Audio file is too small to contain a valid hidden payload.")
        
    base_header_bits = extracted_bits[:48]
    base_header_bytes = _from_bits(base_header_bits)
    
    secret_len = np.frombuffer(base_header_bytes[:4], dtype=np.uint32)[0]
    nonce_len = np.frombuffer(base_header_bytes[4:6], dtype=np.uint16)[0]
    
    # Sanity check
    if nonce_len > 256 or secret_len == 0:
        raise ValueError("Failed to extract: Garbage header. Incorrect key/audio or corrupted audio.")
        
    total_payload_bytes = 6 + nonce_len + secret_len
    total_payload_bits = total_payload_bytes * 8
    
    if len(extracted_bits) < total_payload_bits:
        raise ValueError("Failed to extract: Payload size exceeds audio capacity.")
        
    payload_bits = extracted_bits[:total_payload_bits]
    payload_bytes = _from_bits(payload_bits)
    
    nonce = payload_bytes[6:6+nonce_len]
    encrypted_secret = payload_bytes[6+nonce_len:6+nonce_len+secret_len]
    
    # Decrypt
    key = _derive_aes_key(password, key_file_path)
    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
    decrypted_secret = cipher.decrypt(encrypted_secret)
    
    with open(out_recovered, "wb") as f:
        f.write(decrypted_secret)
        
    return out_recovered

