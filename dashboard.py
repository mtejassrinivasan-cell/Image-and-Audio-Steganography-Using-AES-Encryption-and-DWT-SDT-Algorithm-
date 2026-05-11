from pathlib import Path
from secrets import token_bytes
from tempfile import TemporaryDirectory

import cv2
import pandas as pd
import streamlit as st
import qrcode
from io import BytesIO

from stego_aes_dwt_svd import embed_only, extract_only  # type: ignore
from evaluation_metrics import evaluate_project  # type: ignore
from audio_stego import embed_audio, extract_audio


st.set_page_config(page_title="Steganography Dashboard", layout="wide")
st.title("AES + DWT-SVD Steganography Dashboard")
st.caption("Hide and recover grayscale images using password + key file.")


def _save_upload(uploaded_file, dst_path: Path):
    dst_path.write_bytes(uploaded_file.getbuffer())
    return str(dst_path)


def _show_image(path: str, caption: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption=caption, clamp=True)


tab_embed, tab_extract, tab_embed_audio, tab_extract_audio, tab_eval = st.tabs(["Image Embed", "Image Extract", "Audio Embed", "Audio Extract", "Evaluation"])

with tab_embed:
    st.subheader("Embed Hidden Image")
    col1, col2 = st.columns(2)

    with col1:
        cover_up = st.file_uploader("Cover image (.png/.jpg)", type=["png", "jpg", "jpeg"], key="cover_embed")
        ptype = st.radio("Payload Type", ["Image", "Text"])
        if ptype == "Image":
            secret_up = st.file_uploader("Hidden image (.png/.jpg)", type=["png", "jpg", "jpeg"], key="secret_embed")
            secret_text = ""
        else:
            secret_text = st.text_area("Hidden text message", key="secret_embed_text")
            secret_up = None

        password = st.text_input("Password", type="password", key="pass_embed")
        auto_alpha = st.checkbox("Auto-Optimize Alpha (Aim >40dB PSNR)", value=False)
        as_gray_embed = st.checkbox("Convert to Grayscale", value=False, key="gray_embed")
        if not auto_alpha:
            alpha = st.slider("Alpha (embedding strength)", 0.03, 0.20, 0.08, 0.01, key="alpha_embed")
        else:
            alpha = 0.08
        stego_name = st.text_input("Stego output filename", value="stego.png", key="stego_name")

        st.markdown("**Key file**")
        key_up = st.file_uploader("Upload existing key file (.bin)", type=["bin"], key="key_embed_upload")
        gen_key = st.button("Generate New Key", key="gen_key_btn")
        if gen_key:
            st.session_state["generated_key_bytes"] = token_bytes(32)
            st.success("New key generated. Download it and keep it safe.")

        if "generated_key_bytes" in st.session_state:
            st.download_button(
                "Download Generated Key (.bin)",
                data=st.session_state["generated_key_bytes"],
                file_name="stego_key.bin",
                mime="application/octet-stream",
                key="download_key_btn",
            )
            try:
                qr = qrcode.QRCode(version=1, box_size=10, border=4)
                qr.add_data(st.session_state["generated_key_bytes"].hex())
                qr.make(fit=True)
                img_qr = qr.make_image(fill_color="black", back_color="white")
                buf = BytesIO()
                img_qr.save(buf, format="PNG")
                st.download_button("Download QR Key (_qr.png)", data=buf.getvalue(), file_name="stego_key_qr.png", mime="image/png", key="qr_dl")
            except Exception as e:
                pass

        run_embed = st.button("Run Embed", type="primary", key="run_embed")

    with col2:
        if run_embed:
            if not cover_up:
                st.error("Please upload cover image.")
            elif ptype == "Image" and not secret_up:
                st.error("Please upload the hidden image.")
            elif ptype == "Text" and not secret_text:
                st.error("Please enter a text message.")
            elif not password:
                st.error("Please enter a password.")
            elif not key_up and "generated_key_bytes" not in st.session_state:
                st.error("Upload a key file or generate a new key.")
            else:
                with TemporaryDirectory() as td:
                    td_path = Path(td)
                    cover_path = _save_upload(cover_up, td_path / "cover.png")
                    secret_path = _save_upload(secret_up, td_path / "secret.png") if secret_up else ""

                    if key_up:
                        key_path = _save_upload(key_up, td_path / "key.bin")
                    else:
                        key_path = str(td_path / "generated_key.bin")
                        Path(key_path).write_bytes(st.session_state["generated_key_bytes"])

                    stego_path = str(td_path / stego_name)

                    try:
                        def _run_emb(a):
                            return embed_only(
                                cover_path=cover_path,
                                secret_path=secret_path if secret_up else "",
                                out_stego=stego_path,
                                password=password,
                                key_file_path=key_path,
                                alpha=float(a),
                                payload_type=ptype.lower(),
                                secret_text=secret_text,
                                as_gray=as_gray_embed
                            )

                        if auto_alpha:
                            best_a = 0.08
                            best_psnr = 0
                            metrics = None
                            for test_a in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]:
                                try:
                                    m = _run_emb(test_a)
                                    if m['psnr'] > best_psnr and m['psnr'] >= 40.0:
                                        best_psnr = m['psnr']
                                        best_a = test_a
                                        metrics = m
                                except:
                                    pass
                            if metrics is None:
                                metrics = _run_emb(0.08)
                                st.warning("Auto-optimize could not find an ideal alpha. Defaulted to 0.08.")
                            else:
                                _run_emb(best_a)  # generate final with best alpha
                                st.success(f"Auto-optimized Alpha to {best_a}")
                        else:
                            metrics = _run_emb(alpha)

                        st.success("Embedding complete.")
                        if metrics.get("original_shape") and metrics.get("resized_shape") and metrics["original_shape"] != metrics["resized_shape"]:
                            orig_str = f"{metrics['original_shape'][1]}x{metrics['original_shape'][0]}"
                            rsz_str = f"{metrics['resized_shape'][1]}x{metrics['resized_shape'][0]}"
                            st.warning(f"Note: The hidden image was automatically resized from **{orig_str}** to **{rsz_str}** to mathematically fit within the cover image's maximum data capacity ({metrics['capacity_bytes']} bytes).")
                        st.metric("PSNR (Cover vs Stego)", f"{metrics['psnr']:.4f} dB")
                        st.metric("MSE (Cover vs Stego)", f"{metrics['mse']:.6f}")
                        _show_image(cover_path, "Cover Image")
                        if ptype == "Image":
                            _show_image(secret_path, "Hidden Image (rescaled input)")
                        _show_image(stego_path, "Stego Image (output)")

                        st.download_button(
                            "Download Stego Image",
                            data=Path(stego_path).read_bytes(),
                            file_name=stego_name,
                            mime="image/png",
                            key="download_stego_btn",
                        )
                    except Exception as exc:
                        st.error(f"Embedding failed: {exc}")

with tab_extract:
    st.subheader("Extract Original Hidden Image")
    col3, col4 = st.columns(2)

    with col3:
        cover_up_e = st.file_uploader("Original cover image", type=["png", "jpg", "jpeg"], key="cover_extract")
        stego_up = st.file_uploader("Stego image", type=["png", "jpg", "jpeg"], key="stego_extract")
        ptype_e = st.radio("Payload Type", ["Image", "Text"], key="pt_extract")
        k_method = st.radio("Key Input Method", [".bin File", "QR Code Image"])
        if k_method == ".bin File":
            key_up_e = st.file_uploader("Key file (.bin)", type=["bin"], key="key_extract")
            qr_up = None
        else:
            qr_up = st.file_uploader("Upload QR Code Image", type=["png", "jpg", "jpeg"], key="qr_extract")
            key_up_e = None
        password_e = st.text_input("Password", type="password", key="pass_extract")
        alpha_e = st.slider("Alpha used during embedding", 0.03, 0.20, 0.08, 0.01, key="alpha_extract")
        as_gray_extract = st.checkbox("Convert to Grayscale", value=False, key="gray_extract")
        recovered_name = st.text_input("Recovered output filename", value="recovered_secret.png", key="recover_name")
        run_extract = st.button("Run Extract", type="primary", key="run_extract")

    with col4:
        if run_extract:
            if not cover_up_e or not stego_up:
                st.error("Please upload cover and stego images.")
            elif not key_up_e and not qr_up:
                st.error("Please provide the key (.bin or QR code)")
            elif not password_e:
                st.error("Please enter a password.")
            else:
                with TemporaryDirectory() as td:
                    td_path = Path(td)
                    cover_path = _save_upload(cover_up_e, td_path / "cover.png")
                    stego_path = _save_upload(stego_up, td_path / "stego.png")
                    if key_up_e:
                        key_path = _save_upload(key_up_e, td_path / "key.bin")
                    else:
                        qr_path = _save_upload(qr_up, td_path / "qr.png")
                        img = cv2.imread(qr_path)
                        detector = cv2.QRCodeDetector()
                        data, _, _ = detector.detectAndDecode(img)
                        if data:
                            key_path = str(td_path / "key.bin")
                            Path(key_path).write_bytes(bytes.fromhex(data))
                        else:
                            st.error("Could not decode QR code.")
                            st.stop()

                    recovered_path = str(td_path / recovered_name)

                    try:
                        result = extract_only(
                            cover_path=cover_path,
                            stego_path=stego_path,
                            out_recovered=recovered_path,
                            password=password_e,
                            key_file_path=key_path,
                            alpha=float(alpha_e),
                            payload_type=ptype_e.lower(),
                            as_gray=as_gray_extract
                        )
                        st.success("Extraction complete.")
                        _show_image(cover_path, "Original Cover")
                        _show_image(stego_path, "Stego Image")

                        if ptype_e == "Image":
                            _show_image(recovered_path, "Recovered Hidden Image")
                            st.download_button(
                                "Download Recovered Image",
                                data=Path(recovered_path).read_bytes(),
                                file_name=recovered_name,
                                mime="image/png",
                                key="download_recovered_btn",
                            )
                        else:
                            st.text_area("Recovered Text Payload", result, height=150)
                    except Exception as exc:
                        st.error(f"Extraction failed: {exc}")

with tab_embed_audio:
    st.subheader("Embed Hidden Audio")
    colA, colB = st.columns(2)

    with colA:
        cover_audio_up = st.file_uploader("Cover audio (.wav)", type=["wav"], key="cover_audio_embed")
        secret_audio_up = st.file_uploader("Secret audio (.wav, .mp3)", type=["wav", "mp3"], key="secret_audio_embed")
        password_audio_emb = st.text_input("Password", type="password", key="pass_audio_embed")
        stego_audio_name = st.text_input("Stego output filename", value="stego.wav", key="stego_audio_name")
        
        st.markdown("**Key file**")
        key_up_audio_emb = st.file_uploader("Upload existing key file (.bin)", type=["bin"], key="key_audio_embed_upload")
        gen_key_audio = st.button("Generate New Key", key="gen_key_audio_btn")
        if gen_key_audio:
            st.session_state["generated_key_bytes_audio"] = token_bytes(32)
            st.success("New key generated. Download it and keep it safe.")

        if "generated_key_bytes_audio" in st.session_state:
            st.download_button(
                "Download Generated Key (.bin)",
                data=st.session_state["generated_key_bytes_audio"],
                file_name="stego_audio_key.bin",
                mime="application/octet-stream",
                key="download_key_audio_btn",
            )
            try:
                qr_audio = qrcode.QRCode(version=1, box_size=10, border=4)
                qr_audio.add_data(st.session_state["generated_key_bytes_audio"].hex())
                qr_audio.make(fit=True)
                img_qr_audio = qr_audio.make_image(fill_color="black", back_color="white")
                buf_audio = BytesIO()
                img_qr_audio.save(buf_audio, format="PNG")
                st.download_button("Download QR Key (_qr.png)", data=buf_audio.getvalue(), file_name="stego_audio_key_qr.png", mime="image/png", key="qr_dl_audio")
            except Exception as e:
                pass
        
        run_audio_embed = st.button("Run Audio Embed", type="primary", key="run_audio_embed")

    with colB:
        if run_audio_embed:
            if not cover_audio_up:
                st.error("Please upload cover audio.")
            elif not secret_audio_up:
                st.error("Please upload the hidden audio.")
            elif not password_audio_emb:
                st.error("Please enter a password.")
            elif not key_up_audio_emb and "generated_key_bytes_audio" not in st.session_state:
                st.error("Upload a key file or generate a new key.")
            else:
                with TemporaryDirectory() as td:
                    td_path = Path(td)
                    cover_audio_path = _save_upload(cover_audio_up, td_path / "cover.wav")
                    secret_audio_path = _save_upload(secret_audio_up, td_path / "secret.wav")
                    
                    if key_up_audio_emb:
                        key_path = _save_upload(key_up_audio_emb, td_path / "key.bin")
                    else:
                        key_path = str(td_path / "generated_audio_key.bin")
                        Path(key_path).write_bytes(st.session_state["generated_key_bytes_audio"])
                        
                    stego_audio_path = str(td_path / stego_audio_name)
                    
                    try:
                        metrics = embed_audio(
                            cover_path=cover_audio_path,
                            secret_path=secret_audio_path,
                            out_stego=stego_audio_path,
                            password=password_audio_emb,
                            key_file_path=key_path
                        )
                        st.success("Audio Embedding complete.")
                        st.metric("Cover Capacity (Bytes)", metrics["capacity_bytes"])
                        st.metric("Used Payload (Bytes)", metrics["used_bytes"])
                        st.audio(stego_audio_path, format="audio/wav")
                        st.download_button(
                            "Download Stego Audio",
                            data=Path(stego_audio_path).read_bytes(),
                            file_name=stego_audio_name,
                            mime="audio/wav",
                            key="download_stego_audio_btn",
                        )
                    except Exception as exc:
                        st.error(f"Audio Embedding failed: {exc}")
                        if "too small" in str(exc):
                            st.info("💡 **Tip:** Steganography requires the cover audio to be physically larger than the secret data. At 1 bit per sample, a standard 44.1kHz stereo WAV file can hide about **11 KB per second**. Try using a longer cover audio file (e.g., > 1 minute).")


with tab_extract_audio:
    st.subheader("Extract Original Hidden Audio")
    colC, colD = st.columns(2)

    with colC:
        stego_audio_up = st.file_uploader("Stego audio (.wav)", type=["wav"], key="stego_audio_extract")
        key_up_audio_ext = st.file_uploader("Key file (.bin)", type=["bin"], key="key_audio_extract")
        password_audio_ext = st.text_input("Password", type="password", key="pass_audio_extract")
        recovered_audio_name = st.text_input("Recovered output filename (e.g., .wav, .mp3)", value="recovered_secret.wav", key="recover_audio_name")
        run_audio_extract = st.button("Run Audio Extract", type="primary", key="run_audio_extract")

    with colD:
        if run_audio_extract:
            if not stego_audio_up:
                st.error("Please upload stego audio.")
            elif not password_audio_ext:
                st.error("Please enter a password.")
            else:
                with TemporaryDirectory() as td:
                    td_path = Path(td)
                    stego_audio_path = _save_upload(stego_audio_up, td_path / "stego.wav")
                    if key_up_audio_ext:
                        key_path = _save_upload(key_up_audio_ext, td_path / "key.bin")
                    else:
                        key_path = None
                    
                    recovered_audio_path = str(td_path / recovered_audio_name)
                    
                    try:
                        extract_audio(
                            stego_path=stego_audio_path,
                            out_recovered=recovered_audio_path,
                            password=password_audio_ext,
                            key_file_path=key_path
                        )
                        st.success("Audio Extraction complete.")
                        st.audio(recovered_audio_path, format="audio/wav")
                        mime_type = "audio/wav" if recovered_audio_name.endswith(".wav") else "audio/mpeg"
                        st.download_button(
                            "Download Recovered Audio",
                            data=Path(recovered_audio_path).read_bytes(),
                            file_name=recovered_audio_name,
                            mime=mime_type,
                            key="download_recovered_audio_btn",
                        )
                    except Exception as exc:
                        st.error(f"Audio Extraction failed: {exc}")

with tab_eval:
    st.subheader("System Evaluation")
    col5, col6 = st.columns(2)

    with col5:
        cover_up_val = st.file_uploader("Cover image", type=["png", "jpg", "jpeg"], key="cover_eval")
        ptype_eval = st.radio("Payload Type", ["Image", "Text"], key="pt_eval")
        if ptype_eval == "Image":
            secret_up_val = st.file_uploader("Secret image", type=["png", "jpg", "jpeg"], key="secret_eval")
            secret_text_val = ""
        else:
            secret_text_val = st.text_area("Hidden text message", key="secret_eval_text")
            secret_up_val = None
        key_up_val = st.file_uploader("Key file (.bin)", type=["bin"], key="key_eval")
        password_val = st.text_input("Password", type="password", key="pass_eval")
        alpha_val = st.slider("Alpha", 0.03, 0.20, 0.08, 0.01, key="alpha_eval")
        run_eval = st.button("Run Evaluation", type="primary", key="run_eval")

    with col6:
        if run_eval:
            if not cover_up_val:
                st.error("Please upload cover image.")
            elif ptype_eval == "Image" and not secret_up_val:
                st.error("Please upload the secret image.")
            elif ptype_eval == "Text" and not secret_text_val:
                st.error("Please enter a text message.")
            elif not key_up_val:
                st.error("Please provide the key (.bin)")
            elif not password_val:
                st.error("Please enter a password.")
            else:
                with st.spinner("Evaluating... This may take a moment."):
                    with TemporaryDirectory() as td:
                        td_path = Path(td)
                        cover_path = _save_upload(cover_up_val, td_path / "cover.png")
                        secret_path = _save_upload(secret_up_val, td_path / "secret.png") if secret_up_val else ""
                        key_path = _save_upload(key_up_val, td_path / "key.bin")

                        try:
                            report = evaluate_project(
                                cover_path=cover_path,
                                secret_path=secret_path if secret_up_val else "",
                                password=password_val,
                                key_file=key_path,
                                alpha=float(alpha_val),
                                payload_type=ptype_eval.lower(),
                                secret_text=secret_text_val
                            )
                            st.success("Evaluation complete.")

                            st.markdown("### Quality & Statistical Metrics")
                            q_metrics = report["quality_metrics"]
                            q_df = pd.DataFrame([q_metrics]).T
                            q_df.columns = ["Value"]
                            st.dataframe(q_df, use_container_width=True)

                            st.markdown("### Pixel Intensity Histograms")
                            hist_data = pd.DataFrame({
                                "Cover": report["histograms"]["cover"],
                                "Stego": report["histograms"]["stego"]
                            })
                            st.line_chart(hist_data)

                            st.markdown("### Robustness Metrics")
                            r_metrics = report["robustness_metrics"]
                            chart_data = []
                            for atk, m in r_metrics.items():
                                if "error" not in m:
                                    chart_data.append({
                                        "Attack": atk,
                                        "BCR (%)": m.get("bcr_percent", 0),
                                        "PSNR (dB)": m.get("psnr_secret_db", 0),
                                        "SSIM": m.get("ssim_secret", 0)
                                    })
                                else:
                                    st.warning(f"Attack {atk} failed: {m['error']}")

                            if chart_data:
                                r_df = pd.DataFrame(chart_data).set_index("Attack")
                                st.dataframe(r_df, use_container_width=True)
                                st.bar_chart(r_df[["BCR (%)", "PSNR (dB)", "SSIM"]])
                            else:
                                st.error("No robustness metrics available.")

                        except Exception as exc:
                            st.error(f"Evaluation failed: {exc}")
