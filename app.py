import os
import sys
import threading
import tempfile
import random
import time
import gc
from pathlib import Path
import wx
import torch
import soundfile as sf
import numpy as np
from qwen_tts import Qwen3TTSModel
from faster_whisper import WhisperModel
from pydub import AudioSegment
import librosa
import sentencex

# Optimización global para matmul (TensorFloat32 en Ampere+)
torch.set_float32_matmul_precision('high')

# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
WHISPER_SIZE = "small"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
ATTN_IMPL = "flash_attention_2"
MAX_NEW_TOKENS = 2048
CHUNK_SIZE_LIMIT = 900          # Caracteres máximos por fragmento (ajustable)

BASE_DIR = Path(__file__).parent
VOICES_DIR = BASE_DIR / "voces"
AUDIOS_DIR = BASE_DIR / "audios"
VOICES_DIR.mkdir(exist_ok=True)
AUDIOS_DIR.mkdir(exist_ok=True)

LANG_CODE_MAP = {
    "Spanish": "es", "English": "en", "Chinese": "zh", "Japanese": "ja",
    "Korean": "ko", "German": "de", "French": "fr", "Russian": "ru",
    "Portuguese": "pt", "Italian": "it",
}

# ----------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------
def normalize_ref_audio(audio_path):
    """Convierte audio a 24 kHz mono y normaliza."""
    try:
        audio_segment = AudioSegment.from_file(audio_path)
        audio_segment = audio_segment.set_frame_rate(24000).set_channels(1)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            audio_segment.export(temp_path, format="wav")
        wav, sr = librosa.load(temp_path, sr=24000, mono=True)
        os.unlink(temp_path)
        wav = wav.astype(np.float32)
        peak = np.max(np.abs(wav))
        if peak > 0:
            wav = wav / (peak + 1e-12)
        wav = np.clip(wav, -1.0, 1.0)
        return wav, int(sr)
    except Exception as e:
        print(f"Error procesando audio: {e}")
        return None

def split_text_into_chunks(text, language_code, max_chars=CHUNK_SIZE_LIMIT):
    """Divide texto en fragmentos respetando límite de caracteres y oraciones."""
    if len(text) <= max_chars:
        return [text]
    sentences = list(sentencex.segment(language_code, text))
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            chunks.append(sentence.strip())
        elif len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += (" " + sentence if current_chunk else sentence)
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_audio_filename(prefix="generado"):
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.wav"

# ----------------------------------------------------------------------
# Hilo de carga de modelos (con optimizaciones completas y warmup)
# ----------------------------------------------------------------------
class ModelLoaderThread(threading.Thread):
    def __init__(self, parent, status_callback):
        super().__init__()
        self.parent = parent
        self.status_callback = status_callback

    def run(self):
        wx.CallAfter(self.status_callback, "Cargando modelo TTS...")
        try:
            tts = Qwen3TTSModel.from_pretrained(
                MODEL_NAME,
                device_map=DEVICE,
                dtype=DTYPE,
                attn_implementation=ATTN_IMPL,
            )
        except Exception as e:
            wx.CallAfter(self.status_callback, f"Error TTS: {e}")
            wx.CallAfter(self.parent.on_model_load_failed, str(e))
            return

        # Aplicar optimizaciones completas (como en test_optimized_no_streaming.py)
        wx.CallAfter(self.status_callback, "Aplicando optimizaciones completas (esto puede tomar varios minutos)...")

        try:
            print("\n" + "=" * 60)
            print("Aplicando optimizaciones: max-autotune + compile_talker + fast codebook")
            print("=" * 60)

            tts.enable_streaming_optimizations(
                decode_window_frames=300,
                use_compile=True,
                use_cuda_graphs=False,
                compile_mode="max-autotune",
                use_fast_codebook=True,
                compile_codebook_predictor=True,
                compile_talker=True,
            )

            # Calentamiento con textos de distintas longitudes para compilar todos los shapes
            wx.CallAfter(self.status_callback, "Calentando el modelo (primera compilación, puede tardar 1-2 minutos)...")

            warmup_texts = [
                "Hola",
                "Este es un texto de prueba.",
                "La inteligencia artificial está transformando el mundo.",
                "Este cuarto texto tiene aproximadamente cien caracteres, suficiente para cubrir un tamaño mediano.",
                "Este quinto texto tiene alrededor de doscientos caracteres y se usa para compilar tamaños grandes."
            ]

            dummy_sr = 24000
            dummy_audio = np.zeros(dummy_sr, dtype=np.float32)  # 1 segundo de silencio

            print("\nRealizando calentamiento con múltiples textos...")
            for idx, wtext in enumerate(warmup_texts, 1):
                warmup_start = time.time()
                try:
                    wavs, sr = tts.generate_voice_clone(
                        text=wtext,
                        language="Spanish",
                        ref_audio=(dummy_audio, dummy_sr),
                        ref_text=wtext[:50],
                        x_vector_only_mode=False,
                        max_new_tokens=200
                    )
                    elapsed = time.time() - warmup_start
                    print(f"  Warmup {idx}: {elapsed:.2f}s para {len(wtext)} caracteres")
                    del wavs
                except Exception as e:
                    print(f"  Warmup {idx} falló (no crítico): {e}")
                torch.cuda.empty_cache()
                gc.collect()

            print("Calentamiento finalizado. Modelo listo para generación rápida.")

        except Exception as e:
            print(f"Advertencia: No se pudieron aplicar todas las optimizaciones ({e})")
            print("El modelo funcionará sin ellas, pero será más lento.")

        wx.CallAfter(self.status_callback, "Cargando Whisper...")
        try:
            whisper = WhisperModel(
                WHISPER_SIZE,
                device="cuda" if "cuda" in DEVICE else "cpu",
                compute_type="float16"
            )
        except Exception as e:
            wx.CallAfter(self.status_callback, f"Error Whisper: {e}")
            wx.CallAfter(self.parent.on_model_load_failed, str(e))
            return

        wx.CallAfter(self.parent.on_models_loaded, tts, whisper)

# ----------------------------------------------------------------------
# Hilo de transcripción
# ----------------------------------------------------------------------
class TranscriptionThread(threading.Thread):
    def __init__(self, parent, whisper_model, audio_path, txt_path, callback):
        super().__init__()
        self.parent = parent
        self.whisper = whisper_model
        self.audio_path = audio_path
        self.txt_path = txt_path
        self.callback = callback

    def run(self):
        try:
            wx.CallAfter(self.parent.update_status, "Transcribiendo...")
            segments, info = self.whisper.transcribe(str(self.audio_path), language=None, beam_size=5)
            text = " ".join([seg.text for seg in segments])
            with open(self.txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            wx.CallAfter(self.callback, text.strip())
        except Exception as e:
            wx.CallAfter(self.callback, None, error=str(e))
        finally:
            wx.CallAfter(self.parent.unload_whisper)

# ----------------------------------------------------------------------
# Hilo de generación
# ----------------------------------------------------------------------
class GenerationThread(threading.Thread):
    def __init__(self, parent, tts_model, ref_audio_path, ref_text,
                 target_text, language, seed, temperature, top_p,
                 x_vector_only, status_callback, done_callback):
        super().__init__()
        self.parent = parent
        self.tts = tts_model
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.target_text = target_text
        self.language = language
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        self.x_vector_only = x_vector_only
        self.status_callback = status_callback
        self.done_callback = done_callback

    def run(self):
        try:
            wx.CallAfter(self.status_callback, "Normalizando audio...")
            audio_tuple = normalize_ref_audio(self.ref_audio_path)
            if audio_tuple is None:
                raise Exception("Error al procesar el audio de referencia.")

            lang_code = LANG_CODE_MAP.get(self.language, "es") if self.language != "Auto" else "es"

            wx.CallAfter(self.status_callback, "Preparando texto...")
            if len(self.target_text) > CHUNK_SIZE_LIMIT:
                chunks = split_text_into_chunks(self.target_text, lang_code)
                wx.CallAfter(self.status_callback, f"Texto dividido en {len(chunks)} fragmentos.")
            else:
                chunks = [self.target_text]

            if self.seed != -1:
                torch.manual_seed(self.seed)
                random.seed(self.seed)
                np.random.seed(self.seed)

            gen_kwargs = {"max_new_tokens": MAX_NEW_TOKENS}
            if abs(self.temperature - 0.9) > 1e-3:
                gen_kwargs["temperature"] = self.temperature
            if abs(self.top_p - 1.0) > 1e-3:
                gen_kwargs["top_p"] = self.top_p

            all_audio = []
            sample_rate = None
            total_start = time.time()

            for i, chunk in enumerate(chunks):
                wx.CallAfter(self.status_callback, f"Generando fragmento {i+1}/{len(chunks)}...")
                start_gen = time.time()

                wavs, sr_out = self.tts.generate_voice_clone(
                    text=chunk,
                    language=self.language,
                    ref_audio=audio_tuple,
                    ref_text=self.ref_text,
                    x_vector_only_mode=self.x_vector_only,
                    **gen_kwargs
                )

                elapsed = time.time() - start_gen
                wx.CallAfter(self.status_callback, f"Fragmento {i+1} generado en {elapsed:.1f}s")

                if sample_rate is None:
                    sample_rate = sr_out
                all_audio.append(wavs[0])

                del wavs
                torch.cuda.empty_cache()
                gc.collect()

            if all_audio:
                final_audio = np.concatenate(all_audio)
                total_time = time.time() - total_start
                audio_duration = len(final_audio) / sample_rate if sample_rate else 0
                rtf = total_time / audio_duration if audio_duration > 0 else 0

                print("\n" + "=" * 50)
                print("RESUMEN DE GENERACIÓN")
                print("=" * 50)
                print(f"Texto: {len(self.target_text)} caracteres en {len(chunks)} fragmentos")
                print(f"Tiempo total: {total_time:.2f}s")
                print(f"Duración audio: {audio_duration:.2f}s")
                print(f"RTF: {rtf:.2f}")
                print(f"Modo rápido (x-vector): {'ACTIVADO' if self.x_vector_only else 'DESACTIVADO'}")
                print("=" * 50)

                wx.CallAfter(self.done_callback, final_audio, sample_rate, len(chunks))
            else:
                wx.CallAfter(self.done_callback, None, None, 0)

        except Exception as e:
            wx.CallAfter(self.done_callback, None, None, 0, error=str(e))

# ----------------------------------------------------------------------
# Marco principal (interfaz wxPython)
# ----------------------------------------------------------------------
class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Clonación de Voz Qwen3-TTS (Ultra Rápido)", size=(950, 700))
        self.tts_model = None
        self.whisper_model = None
        self.current_audio_data = None
        self.last_saved_path = None

        self._create_accelerators()
        self._create_ui()
        self.load_models()
        self.Centre()
        self.Show()

    def _create_accelerators(self):
        self.accel_id_folder = wx.NewIdRef()
        self.accel_id_file = wx.NewIdRef()
        self.accel_id_text = wx.NewIdRef()
        self.accel_id_lang = wx.NewIdRef()
        self.accel_id_generate = wx.NewIdRef()
        entries = [
            (wx.ACCEL_ALT, ord('C'), self.accel_id_folder),
            (wx.ACCEL_ALT, ord('A'), self.accel_id_file),
            (wx.ACCEL_ALT, ord('T'), self.accel_id_text),
            (wx.ACCEL_ALT, ord('I'), self.accel_id_lang),
            (wx.ACCEL_ALT, ord('G'), self.accel_id_generate),
        ]
        self.SetAcceleratorTable(wx.AcceleratorTable(entries))
        self.Bind(wx.EVT_MENU, self.on_accel_folder, id=self.accel_id_folder)
        self.Bind(wx.EVT_MENU, self.on_accel_file, id=self.accel_id_file)
        self.Bind(wx.EVT_MENU, self.on_accel_text, id=self.accel_id_text)
        self.Bind(wx.EVT_MENU, self.on_accel_lang, id=self.accel_id_lang)
        self.Bind(wx.EVT_MENU, self.on_accel_generate, id=self.accel_id_generate)

    def on_accel_folder(self, e): self.folder_choice.SetFocus()
    def on_accel_file(self, e): self.file_choice.SetFocus()
    def on_accel_text(self, e): self.target_text.SetFocus()
    def on_accel_lang(self, e): self.lang_choice.SetFocus()
    def on_accel_generate(self, e):
        if self.generate_btn.IsEnabled():
            self.on_generate(e)

    def _create_ui(self):
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # --- Selección de referencia ---
        ref_box = wx.StaticBox(panel, label="Voz de referencia")
        ref_sizer = wx.StaticBoxSizer(ref_box, wx.VERTICAL)
        grid1 = wx.FlexGridSizer(cols=2, vgap=5, hgap=5)
        grid1.AddGrowableCol(1)

        lbl_folder = wx.StaticText(panel, label="&Carpeta (Alt+C):")
        self.folder_choice = wx.Choice(panel, size=(300, -1))
        grid1.Add(lbl_folder, 0, wx.ALIGN_CENTER_VERTICAL)
        grid1.Add(self.folder_choice, 1, wx.EXPAND)

        lbl_file = wx.StaticText(panel, label="&Archivo (Alt+A):")
        self.file_choice = wx.Choice(panel, size=(300, -1))
        grid1.Add(lbl_file, 0, wx.ALIGN_CENTER_VERTICAL)
        grid1.Add(self.file_choice, 1, wx.EXPAND)

        ref_sizer.Add(grid1, 0, wx.EXPAND | wx.ALL, 5)

        lbl_trans = wx.StaticText(panel, label="Texto transcrito (puedes editarlo):")
        ref_sizer.Add(lbl_trans, 0, wx.TOP | wx.LEFT | wx.RIGHT, 5)
        self.transcript_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE, size=(-1, 60))
        ref_sizer.Add(self.transcript_text, 0, wx.EXPAND | wx.ALL, 5)

        main_sizer.Add(ref_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # --- Texto a sintetizar ---
        text_box = wx.StaticBox(panel, label="Texto a sintetizar")
        text_sizer = wx.StaticBoxSizer(text_box, wx.VERTICAL)
        lbl_target = wx.StaticText(panel, label="&Texto (Alt+T):")
        text_sizer.Add(lbl_target, 0, wx.TOP | wx.LEFT | wx.RIGHT, 5)
        self.target_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE, size=(-1, 100))
        text_sizer.Add(self.target_text, 1, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(text_sizer, 1, wx.EXPAND | wx.ALL, 5)

        # --- Parámetros ---
        params_box = wx.StaticBox(panel, label="Parámetros")
        params_sizer = wx.StaticBoxSizer(params_box, wx.VERTICAL)
        grid2 = wx.FlexGridSizer(cols=4, vgap=5, hgap=5)
        grid2.AddGrowableCol(1)
        grid2.AddGrowableCol(3)

        lbl_lang = wx.StaticText(panel, label="&Idioma (Alt+I):")
        self.lang_choice = wx.Choice(panel, choices=[
            "Auto", "Spanish", "English", "Chinese", "Japanese", "Korean",
            "German", "French", "Russian", "Portuguese", "Italian"
        ])
        self.lang_choice.SetSelection(1)  # Spanish
        grid2.Add(lbl_lang, 0, wx.ALIGN_CENTER_VERTICAL)
        grid2.Add(self.lang_choice, 1, wx.EXPAND)

        lbl_seed = wx.StaticText(panel, label="Semilla (-1 aleatorio):")
        self.seed_spin = wx.SpinCtrl(panel, min=-1, max=999999, initial=-1)
        grid2.Add(lbl_seed, 0, wx.ALIGN_CENTER_VERTICAL)
        grid2.Add(self.seed_spin, 1, wx.EXPAND)

        lbl_temp = wx.StaticText(panel, label="Temperatura (0.1-1.5):")
        self.temp_spin = wx.SpinCtrlDouble(panel, min=0.1, max=1.5, inc=0.05, initial=0.9)
        self.temp_spin.SetDigits(2)
        grid2.Add(lbl_temp, 0, wx.ALIGN_CENTER_VERTICAL)
        grid2.Add(self.temp_spin, 1, wx.EXPAND)

        lbl_top_p = wx.StaticText(panel, label="Top P (0.1-1.0):")
        self.top_p_spin = wx.SpinCtrlDouble(panel, min=0.1, max=1.0, inc=0.05, initial=1.0)
        self.top_p_spin.SetDigits(2)
        grid2.Add(lbl_top_p, 0, wx.ALIGN_CENTER_VERTICAL)
        grid2.Add(self.top_p_spin, 1, wx.EXPAND)

        self.xvector_check = wx.CheckBox(panel, label="Modo rápido (x-vector)")
        self.xvector_check.SetValue(False)  # Desactivado por defecto
        grid2.Add(self.xvector_check, 0, wx.ALIGN_CENTER_VERTICAL)
        grid2.AddStretchSpacer()

        # Información sobre fragmentación automática
        lbl_chunk_info = wx.StaticText(panel, label=f"Fragmentación automática >{CHUNK_SIZE_LIMIT} caracteres")
        grid2.Add(lbl_chunk_info, 0, wx.ALIGN_CENTER_VERTICAL)
        grid2.AddStretchSpacer()

        params_sizer.Add(grid2, 0, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(params_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # --- Botones ---
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.generate_btn = wx.Button(panel, label="&GENERAR (Alt+G)")
        self.generate_btn.Disable()
        self.generate_btn.Bind(wx.EVT_BUTTON, self.on_generate)
        btn_sizer.Add(self.generate_btn, 0, wx.ALL, 5)

        self.save_btn = wx.Button(panel, label="Guardar como...")
        self.save_btn.Disable()
        self.save_btn.Bind(wx.EVT_BUTTON, self.on_save)
        btn_sizer.Add(self.save_btn, 0, wx.ALL, 5)

        self.open_folder_btn = wx.Button(panel, label="Abrir carpeta de audios")
        self.open_folder_btn.Bind(wx.EVT_BUTTON, self.on_open_audios_folder)
        btn_sizer.Add(self.open_folder_btn, 0, wx.ALL, 5)

        main_sizer.Add(btn_sizer, 0, wx.ALIGN_CENTER)

        # --- Estado ---
        self.status_label = wx.StaticText(panel, label="Inicializando...", style=wx.ALIGN_CENTER)
        main_sizer.Add(self.status_label, 0, wx.EXPAND | wx.ALL, 5)

        panel.SetSizer(main_sizer)

        self.folder_choice.Bind(wx.EVT_CHOICE, self.on_folder_selected)
        self.file_choice.Bind(wx.EVT_CHOICE, self.on_file_selected)

        self.refresh_folder_list()

    def refresh_folder_list(self):
        folders = [d.name for d in VOICES_DIR.iterdir() if d.is_dir()]
        folders.sort()
        self.folder_choice.SetItems(folders)
        if folders:
            self.folder_choice.SetSelection(0)
            self.on_folder_selected(None)
        else:
            self.status_label.SetLabel("No hay carpetas en ./voces/")

    def on_folder_selected(self, event):
        folder = self.folder_choice.GetStringSelection()
        if not folder:
            return
        path = VOICES_DIR / folder
        audio_files = []
        for ext in ('*.wav', '*.mp3', '*.m4a'):
            audio_files.extend(path.glob(ext))
        audio_files = sorted([f.name for f in audio_files])
        self.file_choice.SetItems(audio_files)
        if audio_files:
            self.file_choice.SetSelection(0)
            self.on_file_selected(None)
        else:
            self.file_choice.Clear()
            self.transcript_text.SetValue("")

    def on_file_selected(self, event):
        folder = self.folder_choice.GetStringSelection()
        file = self.file_choice.GetStringSelection()
        if not folder or not file:
            return
        audio_path = VOICES_DIR / folder / file
        txt_path = audio_path.with_suffix(".txt")

        if txt_path.exists():
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read()
                self.transcript_text.SetValue(text)
                self.status_label.SetLabel("Transcripción cargada.")
                if self.tts_model:
                    self.generate_btn.Enable()
            except Exception as e:
                self.status_label.SetLabel(f"Error leyendo .txt: {e}")
            return

        if self.whisper_model is None:
            self.status_label.SetLabel("Whisper no disponible.")
            return

        self.generate_btn.Disable()
        self.status_label.SetLabel("Transcribiendo...")
        thread = TranscriptionThread(self, self.whisper_model, audio_path, txt_path, self.on_transcription_done)
        thread.start()

    def on_transcription_done(self, text, error=None):
        if error:
            self.status_label.SetLabel(f"Error: {error}")
        elif text:
            self.transcript_text.SetValue(text)
            self.status_label.SetLabel("Transcripción lista.")
        if self.tts_model:
            self.generate_btn.Enable()

    def unload_whisper(self):
        if self.whisper_model:
            del self.whisper_model
            self.whisper_model = None
            torch.cuda.empty_cache()
            gc.collect()

    def load_models(self):
        thread = ModelLoaderThread(self, self.update_status)
        thread.start()

    def update_status(self, msg):
        self.status_label.SetLabel(msg)

    def on_models_loaded(self, tts, whisper):
        self.tts_model = tts
        self.whisper_model = whisper
        self.status_label.SetLabel("Modelos listos. Listo para generar.")
        folder = self.folder_choice.GetStringSelection()
        file = self.file_choice.GetStringSelection()
        if folder and file:
            audio_path = VOICES_DIR / folder / file
            txt_path = audio_path.with_suffix(".txt")
            if txt_path.exists() or self.transcript_text.GetValue().strip():
                self.generate_btn.Enable()
        wx.Bell()

    def on_model_load_failed(self, error):
        wx.MessageBox(f"Error cargando modelos:\n{error}", "Error", wx.OK | wx.ICON_ERROR)

    def on_generate(self, event):
        folder = self.folder_choice.GetStringSelection()
        file = self.file_choice.GetStringSelection()
        if not folder or not file:
            wx.MessageBox("Selecciona carpeta y archivo.", "Error", wx.OK | wx.ICON_WARNING)
            return
        audio_path = VOICES_DIR / folder / file
        ref_text = self.transcript_text.GetValue().strip()
        if not ref_text:
            wx.MessageBox("El texto transcrito no puede estar vacío.", "Error", wx.OK | wx.ICON_WARNING)
            return
        target_text = self.target_text.GetValue().strip()
        if not target_text:
            wx.MessageBox("Ingresa el texto a sintetizar.", "Error", wx.OK | wx.ICON_WARNING)
            return

        language = self.lang_choice.GetStringSelection()
        seed = self.seed_spin.GetValue()
        temp = self.temp_spin.GetValue()
        top_p = self.top_p_spin.GetValue()
        xvec = self.xvector_check.GetValue()

        self.generate_btn.Disable()
        self.save_btn.Disable()
        self.status_label.SetLabel("Iniciando generación...")

        thread = GenerationThread(
            self, self.tts_model, audio_path, ref_text, target_text,
            language, seed, temp, top_p, xvec,
            self.update_status, self.on_generation_done
        )
        thread.start()

    def on_generation_done(self, audio, sr, num_chunks, error=None):
        if error:
            self.status_label.SetLabel(f"Error: {error}")
            wx.MessageBox(f"Error: {error}", "Error", wx.OK | wx.ICON_ERROR)
        elif audio is not None:
            self.current_audio_data = (audio, sr)
            filename = generate_audio_filename()
            save_path = AUDIOS_DIR / filename
            try:
                sf.write(save_path, audio, sr)
                self.last_saved_path = save_path
                self.status_label.SetLabel(f"✅ Completado en {num_chunks} fragmentos. Audio guardado en: {save_path}")
            except Exception as e:
                self.status_label.SetLabel(f"Error guardando: {e}")
            wx.Bell()
            self.save_btn.Enable()
        else:
            self.status_label.SetLabel("No se generó audio.")
        self.generate_btn.Enable()

    def on_save(self, event):
        if self.current_audio_data is None:
            return
        audio, sr = self.current_audio_data
        with wx.FileDialog(self, "Guardar audio", wildcard="WAV (*.wav)|*.wav",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                try:
                    sf.write(dlg.GetPath(), audio, sr)
                    wx.MessageBox("Audio guardado.", "Éxito", wx.OK | wx.ICON_INFORMATION)
                except Exception as e:
                    wx.MessageBox(f"Error: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def on_open_audios_folder(self, event):
        try:
            os.startfile(AUDIOS_DIR)
        except Exception as e:
            wx.MessageBox(f"No se pudo abrir: {e}", "Error", wx.OK | wx.ICON_ERROR)

# ----------------------------------------------------------------------
# Punto de entrada
# ----------------------------------------------------------------------
if __name__ == "__main__":
    app = wx.App()
    frame = MainFrame()
    app.MainLoop()