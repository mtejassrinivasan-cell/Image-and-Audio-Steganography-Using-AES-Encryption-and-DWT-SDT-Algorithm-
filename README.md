# AES + DWT-SVD Image Steganography (Password + Key File)

This version matches your requirement:

- Uses **AES encryption** (instead of Walsh/switched system)
- Uses **DWT-SVD embedding**
- Uses **password + key file** to derive the AES key
- On extraction, it gives the **original hidden image directly** (no unreadable intermediate files)

## Install

```bash
pip install -r requirements.txt
```

## GUI (Recommended)

```bash
python stego_gui.py
```

## Dashboard (Web UI)

```bash
streamlit run dashboard.py
```

Dashboard features:
- Embed and Extract in separate tabs
- Password + key file workflow
- One-click key generation and download
- Preview images and download output files
- PSNR/MSE metrics shown after embedding

### Embed tab
- Select `Cover Image`
- Select `Hidden Image`
- Select `Key File` (or click **Generate Key**)
- Enter `Password`
- Click **Embed Now**

### Extract tab
- Select `Original Cover`
- Select `Stego Image`
- Select the same `Key File` used for embedding
- Enter `Password`
- Click **Extract Original Image**

You will directly get a normal recovered image file (`.png`).

## CLI (Optional)

### Embed

```bash
python stego_aes_dwt_svd.py embed --cover cover.png --secret secret.png --password "mypassword" --key-file mykey.bin --stego-out stego.png --alpha 0.08
```

### Extract

```bash
python stego_aes_dwt_svd.py extract --cover cover.png --stego stego.png --password "mypassword" --key-file mykey.bin --recovered-out recovered_secret.png --alpha 0.08
```

## Evaluation Metrics + Robustness

Run:

```bash
python evaluation_metrics.py --cover cover.png --secret secret.png --password "mypassword" --key-file mykey.bin --alpha 0.08 --report-json evaluation_report.json
```

This computes:
- Image quality: `MSE`, `PSNR`, `SSIM` (Cover vs Stego)
- Recovery quality: `BCR`, `PSNR`, `SSIM` (Secret vs Recovered)
- Robustness under attacks:
  - JPEG compression (`Q=50`)
  - Gaussian noise (`sigma=8`)
  - Salt-and-pepper noise (`1%`)
  - Median blur (`k=3`)
  - Rotation attack (`1 deg`)
  - Crop attack (`5%`, top-left)

## Notes

- Keep the same `password`, `key file`, and `alpha` for successful extraction.
- Use `.png` files for better quality and reliability.
- If payload is too large, use a bigger cover image or a smaller hidden image.
