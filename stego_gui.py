import tkinter as tk
from pathlib import Path
from secrets import token_bytes
from tkinter import filedialog, messagebox, ttk
import tempfile
import cv2
import qrcode

from stego_aes_dwt_svd import embed_only, extract_only  # type: ignore


class StegoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AES + DWT-SVD Steganography")
        self.geometry("880x480")
        self.resizable(False, False)

        tabs = ttk.Notebook(self)
        tabs.pack(fill="both", expand=True, padx=10, pady=10)

        self.embed_tab = ttk.Frame(tabs)
        self.extract_tab = ttk.Frame(tabs)
        tabs.add(self.embed_tab, text="Embed")
        tabs.add(self.extract_tab, text="Extract")

        self.td = tempfile.TemporaryDirectory()
        self.td_path = Path(self.td.name)

        self._build_embed_tab()
        self._build_extract_tab()

    def _browse_file(self, var: tk.StringVar, title: str, save=False, def_ext=".png"):
        if save:
            p = filedialog.asksaveasfilename(
                title=title,
                defaultextension=def_ext,
                filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")],
            )
        else:
            p = filedialog.askopenfilename(title=title, filetypes=[("All Files", "*.*")])
        if p:
            var.set(p)

    def _generate_key_file(self, var: tk.StringVar):
        p = filedialog.asksaveasfilename(
            title="Save key file",
            defaultextension=".bin",
            filetypes=[("Binary Key File", "*.bin"), ("All Files", "*.*")],
        )
        if not p:
            return
        key_bytes = token_bytes(32)
        Path(p).write_bytes(key_bytes)
        var.set(p)
        
        if messagebox.askyesno("QR Code Key", "Would you also like to save this key as a QR Code image?"):
            qr_p = filedialog.asksaveasfilename(
                title="Save QR Code",
                defaultextension=".png",
                filetypes=[("PNG", "*.png")]
            )
            if qr_p:
                qr = qrcode.QRCode(version=1, box_size=10, border=4)
                qr.add_data(key_bytes.hex())
                qr.make(fit=True)
                img = qr.make_image(fill_color="black", back_color="white")
                img.save(qr_p)
                messagebox.showinfo("Success", f"Binary key and QR Code saved!")
        else:
            messagebox.showinfo("Key file created", f"New key file generated:\n{p}")

    def _row(self, parent, r, label, var, browse_title, save=False, key_row=False):
        ttk.Label(parent, text=label, width=18).grid(row=r, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(parent, textvariable=var, width=70).grid(row=r, column=1, sticky="we", padx=6, pady=6)
        ttk.Button(
            parent,
            text="Browse",
            command=lambda: self._browse_file(var, browse_title, save=save),
        ).grid(row=r, column=2, padx=6, pady=6)
        if key_row:
            ttk.Button(
                parent,
                text="Gen Key",
                command=lambda: self._generate_key_file(var),
            ).grid(row=r, column=3, padx=6, pady=6)

    def _build_embed_tab(self):
        f = self.embed_tab
        self.cover_embed = tk.StringVar()
        self.ptype_embed = tk.StringVar(value="image")
        self.secret_embed = tk.StringVar()
        self.key_embed = tk.StringVar()
        self.stego_out = tk.StringVar(value=str(Path.cwd() / "stego.png"))
        self.pass_embed = tk.StringVar()
        self.alpha_embed = tk.StringVar(value="0.08")
        self.auto_alpha = tk.BooleanVar(value=False)
        self.as_gray_embed = tk.BooleanVar(value=False)

        self._row(f, 0, "Cover Image", self.cover_embed, "Select cover image")
        
        ttk.Label(f, text="Payload Type", width=18).grid(row=1, column=0, sticky="w", padx=8, pady=6)
        frm_pt = ttk.Frame(f)
        frm_pt.grid(row=1, column=1, sticky="w", padx=6, pady=6)
        ttk.Radiobutton(frm_pt, text="Image", variable=self.ptype_embed, value="image", command=self._toggle_emb).pack(side="left", padx=5)
        ttk.Radiobutton(frm_pt, text="Text", variable=self.ptype_embed, value="text", command=self._toggle_emb).pack(side="left", padx=5)

        self.lbl_sec = ttk.Label(f, text="Hidden Image", width=18)
        self.lbl_sec.grid(row=2, column=0, sticky="nw", padx=8, pady=6)
        
        self.ent_sec_img = ttk.Entry(f, textvariable=self.secret_embed, width=70)
        self.ent_sec_img.grid(row=2, column=1, sticky="we", padx=6, pady=6)
        self.btn_sec_img = ttk.Button(f, text="Browse", command=lambda: self._browse_file(self.secret_embed, "Select hidden"))
        self.btn_sec_img.grid(row=2, column=2, padx=6, pady=6)
        
        self.txt_sec_msg = tk.Text(f, width=53, height=4)

        self._row(f, 3, "Key File (.bin)", self.key_embed, "Select key file", key_row=True)
        self._row(f, 4, "Stego Output", self.stego_out, "Save stego image", save=True)

        ttk.Label(f, text="Password", width=18).grid(row=5, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(f, textvariable=self.pass_embed, show="*", width=70).grid(row=5, column=1, sticky="we", padx=6, pady=6)

        ttk.Label(f, text="Alpha", width=18).grid(row=6, column=0, sticky="w", padx=8, pady=6)
        frm_al = ttk.Frame(f)
        frm_al.grid(row=6, column=1, sticky="w", padx=6, pady=6)
        ttk.Entry(frm_al, textvariable=self.alpha_embed, width=10).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(frm_al, text="Auto-Optimize Alpha (>40dB)", variable=self.auto_alpha).pack(side="left")
        ttk.Checkbutton(frm_al, text="Convert to Grayscale", variable=self.as_gray_embed).pack(side="left", padx=(10, 0))

        ttk.Button(f, text="Embed Now", command=self._do_embed).grid(row=7, column=1, sticky="w", padx=6, pady=12)

    def _toggle_emb(self):
        if self.ptype_embed.get() == "image":
            self.lbl_sec.config(text="Hidden Image")
            self.txt_sec_msg.grid_remove()
            self.ent_sec_img.grid(row=2, column=1, sticky="we", padx=6, pady=6)
            self.btn_sec_img.grid(row=2, column=2, padx=6, pady=6)
        else:
            self.lbl_sec.config(text="Hidden Text")
            self.ent_sec_img.grid_remove()
            self.btn_sec_img.grid_remove()
            self.txt_sec_msg.grid(row=2, column=1, sticky="we", padx=6, pady=6)

    def _build_extract_tab(self):
        f = self.extract_tab
        self.cover_extract = tk.StringVar()
        self.stego_extract = tk.StringVar()
        self.ptype_extr = tk.StringVar(value="image")
        
        self.key_extr_meth = tk.StringVar(value="bin")
        self.key_extract = tk.StringVar()
        self.qr_extract = tk.StringVar()
        
        self.recovered_out = tk.StringVar(value=str(Path.cwd() / "recovered_secret.png"))
        self.pass_extract = tk.StringVar()
        self.alpha_extract = tk.StringVar(value="0.08")
        self.as_gray_extr = tk.BooleanVar(value=False)

        self._row(f, 0, "Original Cover", self.cover_extract, "Select original cover image")
        self._row(f, 1, "Stego Image", self.stego_extract, "Select stego image")
        
        ttk.Label(f, text="Payload Type", width=18).grid(row=2, column=0, sticky="w", padx=8, pady=6)
        frm_pt = ttk.Frame(f)
        frm_pt.grid(row=2, column=1, sticky="w", padx=6, pady=6)
        ttk.Radiobutton(frm_pt, text="Image", variable=self.ptype_extr, value="image").pack(side="left", padx=5)
        ttk.Radiobutton(frm_pt, text="Text", variable=self.ptype_extr, value="text").pack(side="left", padx=5)

        ttk.Label(f, text="Key Method", width=18).grid(row=3, column=0, sticky="w", padx=8, pady=6)
        frm_km = ttk.Frame(f)
        frm_km.grid(row=3, column=1, sticky="w", padx=6, pady=6)
        ttk.Radiobutton(frm_km, text=".bin File", variable=self.key_extr_meth, value="bin", command=self._toggle_ext_km).pack(side="left", padx=5)
        ttk.Radiobutton(frm_km, text="QR Code Image", variable=self.key_extr_meth, value="qr", command=self._toggle_ext_km).pack(side="left", padx=5)

        self.lbl_key = ttk.Label(f, text="Key File (.bin)", width=18)
        self.lbl_key.grid(row=4, column=0, sticky="w", padx=8, pady=6)
        self.ent_key = ttk.Entry(f, textvariable=self.key_extract, width=70)
        self.ent_key.grid(row=4, column=1, sticky="we", padx=6, pady=6)
        self.btn_key = ttk.Button(f, text="Browse", command=lambda: self._browse_file(self.key_extract, "Select Key"))
        self.btn_key.grid(row=4, column=2, padx=6, pady=6)

        self.ent_qr = ttk.Entry(f, textvariable=self.qr_extract, width=70)
        self.btn_qr = ttk.Button(f, text="Browse UI", command=lambda: self._browse_file(self.qr_extract, "Select QR Code Image"))

        self.lbl_rec = ttk.Label(f, text="Recovered Output", width=18)
        self.lbl_rec.grid(row=5, column=0, sticky="w", padx=8, pady=6)
        self.ent_rec = ttk.Entry(f, textvariable=self.recovered_out, width=70)
        self.ent_rec.grid(row=5, column=1, sticky="we", padx=6, pady=6)
        self.btn_rec = ttk.Button(f, text="Browse", command=lambda: self._browse_file(self.recovered_out, "Save Recovered", save=True))
        self.btn_rec.grid(row=5, column=2, padx=6, pady=6)

        ttk.Label(f, text="Password", width=18).grid(row=6, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(f, textvariable=self.pass_extract, show="*", width=70).grid(row=6, column=1, sticky="we", padx=6, pady=6)

        ttk.Label(f, text="Alpha", width=18).grid(row=7, column=0, sticky="w", padx=8, pady=6)
        frm_al_ex = ttk.Frame(f)
        frm_al_ex.grid(row=7, column=1, sticky="w", padx=6, pady=6)
        ttk.Entry(frm_al_ex, textvariable=self.alpha_extract, width=20).pack(side="left")
        ttk.Checkbutton(frm_al_ex, text="Convert to Grayscale", variable=self.as_gray_extr).pack(side="left", padx=(10, 0))

        ttk.Button(f, text="Extract Payload", command=self._do_extract).grid(row=8, column=1, sticky="w", padx=6, pady=12)

    def _toggle_ext_km(self):
        if self.key_extr_meth.get() == "bin":
            self.lbl_key.config(text="Key File (.bin)")
            self.ent_qr.grid_remove()
            self.btn_qr.grid_remove()
            self.ent_key.grid(row=4, column=1, sticky="we", padx=6, pady=6)
            self.btn_key.grid(row=4, column=2, padx=6, pady=6)
        else:
            self.lbl_key.config(text="QR Code Image")
            self.ent_key.grid_remove()
            self.btn_key.grid_remove()
            self.ent_qr.grid(row=4, column=1, sticky="we", padx=6, pady=6)
            self.btn_qr.grid(row=4, column=2, padx=6, pady=6)

    def _do_embed(self):
        txt = self.txt_sec_msg.get("1.0", tk.END).strip()
        try:
            def _run_e(a):
                return embed_only(
                    cover_path=self.cover_embed.get(),
                    secret_path=self.secret_embed.get() if self.ptype_embed.get()=="image" else "",
                    out_stego=self.stego_out.get(),
                    password=self.pass_embed.get(),
                    key_file_path=self.key_embed.get(),
                    alpha=float(a),
                    payload_type=self.ptype_embed.get(),
                    secret_text=txt,
                    as_gray=self.as_gray_embed.get()
                )

            if self.auto_alpha.get():
                best_psnr = 0
                best_m = None
                best_alpha = 0.08
                for t_a in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]:
                    try:
                        m = _run_e(t_a)
                        if m['psnr'] > best_psnr and m['psnr'] >= 40.0:
                            best_psnr = m['psnr']
                            best_alpha = t_a
                            best_m = m
                    except:
                        pass
                if best_m is None:
                    res = _run_e(float(self.alpha_embed.get()))
                    msg = f"Embedded with selected alpha.\nPSNR: {res['psnr']:.4f}\nSaved: {self.stego_out.get()}"
                else:
                    res = _run_e(best_alpha)
                    msg = f"Auto-optimized Alpha to {best_alpha}.\nPSNR: {res['psnr']:.4f} dB\nSaved at:\n{self.stego_out.get()}"
            else:
                res = _run_e(self.alpha_embed.get())
                msg = f"Stego image created.\nPSNR: {res['psnr']:.4f} dB\nSaved at:\n{self.stego_out.get()}"

            if res.get("original_shape") and res.get("resized_shape") and res["original_shape"] != res["resized_shape"]:
                orig_str = f"{res['original_shape'][1]}x{res['original_shape'][0]}"
                rsz_str = f"{res['resized_shape'][1]}x{res['resized_shape'][0]}"
                msg += f"\n\nWARNING: Hidden image was aggressively resized from {orig_str} to {rsz_str} to fit the cover capacity ({res['capacity_bytes']} bytes)."
            
            messagebox.showinfo("Success", msg)

        except Exception as e:
            messagebox.showerror("Embed failed", str(e))

    def _do_extract(self):
        try:
            key_path = self.key_extract.get()
            if self.key_extr_meth.get() == "qr":
                img = cv2.imread(self.qr_extract.get())
                decoder = cv2.QRCodeDetector()
                data, _, _ = decoder.detectAndDecode(img)
                if not data:
                    raise ValueError("Failed to decode QR code. Ensure image is clear.")
                kp = self.td_path / "decoded.bin"
                kp.write_bytes(bytes.fromhex(data))
                key_path = str(kp)

            result = extract_only(
                cover_path=self.cover_extract.get(),
                stego_path=self.stego_extract.get(),
                out_recovered=self.recovered_out.get(),
                password=self.pass_extract.get(),
                key_file_path=key_path,
                alpha=float(self.alpha_extract.get()),
                payload_type=self.ptype_extr.get(),
                as_gray=self.as_gray_extr.get()
            )

            if self.ptype_extr.get() == "image":
                messagebox.showinfo("Success", f"Original hidden image recovered.\nSaved at:\n{self.recovered_out.get()}")
            else:
                win = tk.Toplevel(self)
                win.title("Recovered Text Payload")
                txt = tk.Text(win, width=60, height=20)
                txt.pack(padx=20, pady=20)
                txt.insert("1.0", result)
                txt.config(state="disabled")

        except Exception as e:
            messagebox.showerror("Extract failed", str(e))

if __name__ == "__main__":
    app = StegoApp()
    app.mainloop()
