import os
import sys
import threading
import tempfile
import random
import time
import gc
import subprocess
import re
import shutil
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

torch.set_float32_matmul_precision('high')

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
WHISPER_SIZE = "small"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
ATTN_IMPL = "flash_attention_2"
MAX_NEW_TOKENS = 4096
CHUNK_SIZE_LIMIT = 300

BASE_DIR = Path(__file__).parent
VOICES_DIR = BASE_DIR / "voces"
AUDIOS_DIR = BASE_DIR / "audios"
SRT_DIR = BASE_DIR / "srt"
VOICES_DIR.mkdir(exist_ok=True)
AUDIOS_DIR.mkdir(exist_ok=True)
SRT_DIR.mkdir(exist_ok=True)

LANG_CODE_MAP = {
    "Spanish": "es", "English": "en", "Chinese": "zh", "Japanese": "ja",
    "Korean": "ko", "German": "de", "French": "fr", "Russian": "ru",
    "Portuguese": "pt", "Italian": "it",
}

def normalize_ref_audio(audio_path):
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
        return None

def split_text_into_chunks(text, language_code, max_chars=CHUNK_SIZE_LIMIT):
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

def parse_multispeaker_text(text, default_voice):
    pattern = r'#(\w+)'
    parts = re.split(pattern, text)
    segments = []
    if parts[0].strip():
        segments.append((default_voice, parts[0].strip()))
    for i in range(1, len(parts), 2):
        if i+1 < len(parts):
            voz = parts[i]
            texto = parts[i+1].strip()
            if texto:
                segments.append((voz, texto))
    return segments

def parse_srt_file(srt_path):
    entries = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = re.split(r'\n\s*\n', content.strip())
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            time_line = lines[1]
            text = ' '.join(lines[2:]).strip()
            time_match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2}),(\d{3})', time_line)
            if time_match:
                start_h, start_m, start_s, start_ms = map(int, time_match.group(1, 2, 3, 4))
                end_h, end_m, end_s, end_ms = map(int, time_match.group(5, 6, 7, 8))
                start_time = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000
                end_time = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000
                entries.append((start_time, end_time, text))
    return entries

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
        wx.CallAfter(self.status_callback, "Aplicando optimizaciones completas...")
        try:
            tts.enable_streaming_optimizations(
                decode_window_frames=300,
                use_compile=True,
                use_cuda_graphs=False,
                compile_mode="max-autotune",
                use_fast_codebook=True,
                compile_codebook_predictor=True,
                compile_talker=True,
            )
            wx.CallAfter(self.status_callback, "Calentando el modelo...")
            warmup_texts = ["Hola", "Este es un texto de prueba."]
            dummy_sr = 24000
            dummy_audio = np.zeros(dummy_sr, dtype=np.float32)
            for wtext in warmup_texts:
                try:
                    wavs, sr = tts.generate_voice_clone(
                        text=wtext,
                        language="Spanish",
                        ref_audio=(dummy_audio, dummy_sr),
                        ref_text=wtext[:50],
                        x_vector_only_mode=False,
                        max_new_tokens=200
                    )
                    del wavs
                except Exception as e:
                    pass
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            pass
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

class SRTGenerationThread(threading.Thread):
    def __init__(self, parent, tts_model, srt_files, voice_path, ref_text,
                 language, seed, temperature, top_p, x_vector_only,
                 status_callback, done_callback):
        super().__init__()
        self.parent = parent
        self.tts = tts_model
        self.srt_files = srt_files
        self.voice_path = voice_path
        self.ref_text = ref_text
        self.language = language
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        self.x_vector_only = x_vector_only
        self.status_callback = status_callback
        self.done_callback = done_callback

    def run(self):
        try:
            if self.seed != -1:
                torch.manual_seed(self.seed)
                random.seed(self.seed)
                np.random.seed(self.seed)
            gen_kwargs = {"max_new_tokens": MAX_NEW_TOKENS * 3}
            if abs(self.temperature - 0.9) > 1e-3:
                gen_kwargs["temperature"] = self.temperature
            if abs(self.top_p - 1.0) > 1e-3:
                gen_kwargs["top_p"] = self.top_p
            wx.CallAfter(self.status_callback, "Normalizando audio de referencia...")
            audio_tuple = normalize_ref_audio(self.voice_path)
            if audio_tuple is None:
                raise Exception("Error al procesar el audio de referencia.")
            for srt_file in self.srt_files:
                wx.CallAfter(self.status_callback, f"Procesando: {srt_file.name}")
                srt_entries = parse_srt_file(srt_file)
                total_duration = srt_entries[-1][1] if srt_entries else 0
                all_audio_segments = []
                sample_rate = None
                for start_time, end_time, text in srt_entries:
                    if not text.strip():
                        continue
                    wx.CallAfter(self.status_callback, f"Generando línea: {text[:50]}...")
                    wavs, sr_out = self.tts.generate_voice_clone(
                        text=text,
                        language=self.language,
                        ref_audio=audio_tuple,
                        ref_text=self.ref_text,
                        x_vector_only_mode=self.x_vector_only,
                        **gen_kwargs
                    )
                    if sample_rate is None:
                        sample_rate = sr_out
                    audio_data = wavs[0]
                    all_audio_segments.append((start_time, audio_data))
                    del wavs
                    torch.cuda.empty_cache()
                    gc.collect()
                if all_audio_segments and sample_rate:
                    final_duration = int(total_duration * sample_rate)
                    final_audio = np.zeros(final_duration, dtype=np.float32)
                    for start_time, audio_data in all_audio_segments:
                        start_sample = int(start_time * sample_rate)
                        end_sample = start_sample + len(audio_data)
                        if end_sample > len(final_audio):
                            final_audio = np.pad(final_audio, (0, end_sample - len(final_audio)))
                        final_audio[start_sample:end_sample] = audio_data[:len(final_audio)-start_sample]
                    output_path = SRT_DIR / f"{srt_file.stem}.wav"
                    sf.write(output_path, final_audio, sample_rate)
                    wx.CallAfter(self.status_callback, f"✅ Completado: {output_path}")
                torch.cuda.empty_cache()
                gc.collect()
            wx.CallAfter(self.done_callback, True)
        except Exception as e:
            wx.CallAfter(self.done_callback, False, str(e))

class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Qwen3-TTS - Conversor SRT a Audio", size=(950, 750))
        self.tts_model = None
        self.whisper_model = None
        self.current_audio_data = None
        self.last_saved_path = None
        self.play_process = None
        self.voice_map = {}
        self._create_accelerators()
        self._create_ui()
        self.scan_voices()
        self.load_models()
        self.Centre()
        self.Show()

    def scan_voices(self):
        self.voice_map.clear()
        for folder in VOICES_DIR.iterdir():
            if folder.is_dir():
                for ext in ('*.wav', '*.mp3', '*.m4a'):
                    for audio_file in folder.glob(ext):
                        etiqueta = audio_file.stem
                        txt_path = audio_file.with_suffix('.txt')
                        ref_text = None
                        if txt_path.exists():
                            try:
                                with open(txt_path, 'r', encoding='utf-8') as f:
                                    ref_text = f.read().strip()
                            except:
                                pass
                        self.voice_map[etiqueta] = {'path': audio_file, 'ref_text': ref_text}

    def _create_accelerators(self):
        self.accel_id_folder = wx.NewIdRef()
        self.accel_id_file = wx.NewIdRef()
        self.accel_id_text = wx.NewIdRef()
        self.accel_id_lang = wx.NewIdRef()
        self.accel_id_generate = wx.NewIdRef()
        self.accel_id_play = wx.NewIdRef()
        self.accel_id_stop = wx.NewIdRef()
        self.accel_id_delete = wx.NewIdRef()
        self.accel_id_srt = wx.NewIdRef()
        entries = [
            (wx.ACCEL_ALT, ord('C'), self.accel_id_folder),
            (wx.ACCEL_ALT, ord('A'), self.accel_id_file),
            (wx.ACCEL_ALT, ord('T'), self.accel_id_text),
            (wx.ACCEL_ALT, ord('I'), self.accel_id_lang),
            (wx.ACCEL_ALT, ord('G'), self.accel_id_generate),
            (wx.ACCEL_ALT, ord('P'), self.accel_id_play),
            (wx.ACCEL_ALT, ord('S'), self.accel_id_stop),
            (wx.ACCEL_ALT, ord('D'), self.accel_id_delete),
            (wx.ACCEL_ALT, ord('R'), self.accel_id_srt),
        ]
        self.SetAcceleratorTable(wx.AcceleratorTable(entries))
        self.Bind(wx.EVT_MENU, self.on_accel_folder, id=self.accel_id_folder)
        self.Bind(wx.EVT_MENU, self.on_accel_file, id=self.accel_id_file)
        self.Bind(wx.EVT_MENU, self.on_accel_text, id=self.accel_id_text)
        self.Bind(wx.EVT_MENU, self.on_accel_lang, id=self.accel_id_lang)
        self.Bind(wx.EVT_MENU, self.on_accel_generate, id=self.accel_id_generate)
        self.Bind(wx.EVT_MENU, self.on_accel_play, id=self.accel_id_play)
        self.Bind(wx.EVT_MENU, self.on_accel_stop, id=self.accel_id_stop)
        self.Bind(wx.EVT_MENU, self.on_accel_delete, id=self.accel_id_delete)
        self.Bind(wx.EVT_MENU, self.on_srt_button, id=self.accel_id_srt)

    def on_accel_folder(self, e): self.folder_choice.SetFocus()
    def on_accel_file(self, e): self.file_choice.SetFocus()
    def on_accel_text(self, e): self.target_text.SetFocus()
    def on_accel_lang(self, e): self.lang_choice.SetFocus()
    def on_accel_generate(self, e):
        if self.generate_btn.IsEnabled():
            self.on_generate(e)
    def on_accel_play(self, e):
        if self.play_btn.IsEnabled():
            self.on_play(e)
    def on_accel_stop(self, e):
        self.on_stop(e)
    def on_accel_delete(self, e):
        if self.delete_btn.IsEnabled():
            self.on_delete(e)

    def _create_ui(self):
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
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
        text_box = wx.StaticBox(panel, label="Texto a sintetizar")
        text_sizer = wx.StaticBoxSizer(text_box, wx.VERTICAL)
        lbl_target = wx.StaticText(panel, label="&Texto (Alt+T) - Usa #nombre para múltiples voces:")
        text_sizer.Add(lbl_target, 0, wx.TOP | wx.LEFT | wx.RIGHT, 5)
        self.target_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE, size=(-1, 100))
        text_sizer.Add(self.target_text, 1, wx.EXPAND | wx.ALL, 5)
        self.target_text.Bind(wx.EVT_CONTEXT_MENU, self.on_text_context_menu)
        main_sizer.Add(text_sizer, 1, wx.EXPAND | wx.ALL, 5)
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
        self.lang_choice.SetSelection(1)
        grid2.Add(lbl_lang, 0, wx.ALIGN_CENTER_VERTICAL)
        grid2.Add(self.lang_choice, 1, wx.EXPAND)
        lbl_seed = wx.StaticText(panel, label="Semilla (-1 aleatorio):")
        self.seed_spin = wx.SpinCtrl(panel, min=-1, max=999999, initial=-1)
        grid2.Add(lbl_seed, 0, wx.ALIGN_CENTER_VERTICAL)
        grid2.Add(self.seed_spin, 1, wx.EXPAND)
        lbl_temp = wx.StaticText(panel, label="Temperatura (0.1-1.5):")
        self.temp_spin = wx.SpinCtrlDouble(panel, min=0.1, max=1.5, inc=0.05, initial=0.4)
        self.temp_spin.SetDigits(2)
        grid2.Add(lbl_temp, 0, wx.ALIGN_CENTER_VERTICAL)
        grid2.Add(self.temp_spin, 1, wx.EXPAND)
        lbl_top_p = wx.StaticText(panel, label="Top P (0.1-1.0):")
        self.top_p_spin = wx.SpinCtrlDouble(panel, min=0.1, max=1.0, inc=0.05, initial=0.4)
        self.top_p_spin.SetDigits(2)
        grid2.Add(lbl_top_p, 0, wx.ALIGN_CENTER_VERTICAL)
        grid2.Add(self.top_p_spin, 1, wx.EXPAND)
        self.xvector_check = wx.CheckBox(panel, label="Modo rápido (x-vector)")
        self.xvector_check.SetValue(False)
        grid2.Add(self.xvector_check, 0, wx.ALIGN_CENTER_VERTICAL)
        self.export_separated_check = wx.CheckBox(panel, label="Exportar audios separados por carpeta")
        self.export_separated_check.SetValue(False)
        grid2.Add(self.export_separated_check, 0, wx.ALIGN_CENTER_VERTICAL)
        lbl_chunk_info = wx.StaticText(panel, label=f"Fragmentación automática >{CHUNK_SIZE_LIMIT} caracteres")
        grid2.Add(lbl_chunk_info, 0, wx.ALIGN_CENTER_VERTICAL)
        grid2.AddStretchSpacer()
        params_sizer.Add(grid2, 0, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(params_sizer, 0, wx.EXPAND | wx.ALL, 5)
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.srt_btn = wx.Button(panel, label="&Cargar archivo SRT (Alt+R)")
        self.srt_btn.Bind(wx.EVT_BUTTON, self.on_srt_button)
        btn_sizer.Add(self.srt_btn, 0, wx.ALL, 5)
        self.generate_btn = wx.Button(panel, label="&GENERAR (Alt+G)")
        self.generate_btn.Disable()
        self.generate_btn.Bind(wx.EVT_BUTTON, self.on_generate)
        btn_sizer.Add(self.generate_btn, 0, wx.ALL, 5)
        self.save_btn = wx.Button(panel, label="Guardar como...")
        self.save_btn.Disable()
        self.save_btn.Bind(wx.EVT_BUTTON, self.on_save)
        btn_sizer.Add(self.save_btn, 0, wx.ALL, 5)
        self.play_btn = wx.Button(panel, label="Reproducir (Alt+P)")
        self.play_btn.Disable()
        self.play_btn.Bind(wx.EVT_BUTTON, self.on_play)
        btn_sizer.Add(self.play_btn, 0, wx.ALL, 5)
        self.stop_btn = wx.Button(panel, label="Detener (Alt+S)")
        self.stop_btn.Disable()
        self.stop_btn.Bind(wx.EVT_BUTTON, self.on_stop)
        btn_sizer.Add(self.stop_btn, 0, wx.ALL, 5)
        self.delete_btn = wx.Button(panel, label="Eliminar (Alt+D)")
        self.delete_btn.Disable()
        self.delete_btn.Bind(wx.EVT_BUTTON, self.on_delete)
        btn_sizer.Add(self.delete_btn, 0, wx.ALL, 5)
        self.open_folder_btn = wx.Button(panel, label="Abrir carpeta de audios")
        self.open_folder_btn.Bind(wx.EVT_BUTTON, self.on_open_audios_folder)
        btn_sizer.Add(self.open_folder_btn, 0, wx.ALL, 5)
        main_sizer.Add(btn_sizer, 0, wx.ALIGN_CENTER)
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

    def on_text_context_menu(self, event):
        menu = wx.Menu()
        for etiqueta, info in self.voice_map.items():
            nombre_mostrado = etiqueta.replace('_', ' ').title()
            item = menu.Append(wx.ID_ANY, nombre_mostrado)
            self.Bind(wx.EVT_MENU, lambda evt, e=etiqueta: self.insert_voice_tag(e), item)
        if menu.MenuItemCount == 0:
            item = menu.Append(wx.ID_ANY, "No hay voces disponibles")
            item.Enable(False)
        self.PopupMenu(menu)
        menu.Destroy()

    def insert_voice_tag(self, etiqueta):
        pos = self.target_text.GetInsertionPoint()
        self.target_text.WriteText(f"#{etiqueta} ")
        self.target_text.SetInsertionPoint(pos + len(etiqueta) + 2)

    def on_srt_button(self, event):
        if self.tts_model is None:
            wx.MessageBox("Los modelos aún no están cargados.", "Error", wx.OK | wx.ICON_WARNING)
            return
        folder = self.folder_choice.GetStringSelection()
        file = self.file_choice.GetStringSelection()
        if not folder or not file:
            wx.MessageBox("Selecciona carpeta y archivo de voz.", "Error", wx.OK | wx.ICON_WARNING)
            return
        ref_text = self.transcript_text.GetValue().strip()
        if not ref_text:
            wx.MessageBox("El texto transcrito no puede estar vacío.", "Error", wx.OK | wx.ICON_WARNING)
            return
        with wx.FileDialog(self, "Seleccionar archivos SRT", wildcard="SRT files (*.srt)|*.srt",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                srt_files = [Path(path) for path in dlg.GetPaths()]
                language = self.lang_choice.GetStringSelection()
                seed = self.seed_spin.GetValue()
                temp = self.temp_spin.GetValue()
                top_p = self.top_p_spin.GetValue()
                xvec = self.xvector_check.GetValue()
                voice_path = VOICES_DIR / folder / file
                self.generate_btn.Disable()
                self.save_btn.Disable()
                self.play_btn.Disable()
                self.delete_btn.Disable()
                self.srt_btn.Disable()
                self.status_label.SetLabel(f"Procesando {len(srt_files)} archivo(s) SRT...")
                thread = SRTGenerationThread(
                    self, self.tts_model, srt_files, voice_path, ref_text,
                    language, seed, temp, top_p, xvec,
                    self.update_status, self.on_srt_generation_done
                )
                thread.start()

    def on_srt_generation_done(self, success, error=None):
        self.generate_btn.Enable()
        self.srt_btn.Enable()
        if success:
            self.status_label.SetLabel("✅ Todos los archivos SRT procesados correctamente.")
            wx.Bell()
        else:
            self.status_label.SetLabel(f"❌ Error: {error}")
            wx.MessageBox(f"Error: {error}", "Error", wx.OK | wx.ICON_ERROR)

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
        export_separated = self.export_separated_check.GetValue()
        default_voice = Path(file).stem if file else None
        self.generate_btn.Disable()
        self.save_btn.Disable()
        self.play_btn.Disable()
        self.delete_btn.Disable()
        self.status_label.SetLabel("Iniciando generación...")
        thread = GenerationThread(
            self, self.tts_model, audio_path, ref_text, target_text,
            language, seed, temp, top_p, xvec, export_separated,
            self.voice_map, default_voice,
            self.update_status, self.on_generation_done
        )
        thread.start()

    def on_generation_done(self, audio_data, sr, num_chunks, is_separated, error=None):
        if error:
            self.status_label.SetLabel(f"Error: {error}")
            wx.MessageBox(f"Error: {error}", "Error", wx.OK | wx.ICON_ERROR)
        elif audio_data is not None:
            if is_separated:
                proj_base = BASE_DIR / "proyecto"
                if proj_base.exists():
                    try:
                        shutil.rmtree(proj_base)
                    except:
                        pass
                proj_base.mkdir(parents=True, exist_ok=True)
                m3u_path = proj_base / "lista_reproduccion.m3u"
                with open(m3u_path, "w", encoding="utf-8") as m3u:
                    for i, (voz, arr) in enumerate(audio_data, 1):
                        speaker_dir = proj_base / voz
                        speaker_dir.mkdir(exist_ok=True)
                        filepath = speaker_dir / f"{i}.wav"
                        sf.write(filepath, arr, sr)
                        m3u.write(f"{voz}/{i}.wav\n")
                self.last_saved_path = m3u_path
                self.current_audio_data = None
                self.status_label.SetLabel(f"✅ Audios separados en: {proj_base}")
                wx.Bell()
                wx.MessageBox(
                    f"El proyecto ha sido generado exitosamente en:\n{proj_base}",
                    "Proyecto Creado", 
                    wx.OK | wx.ICON_INFORMATION
                )
                self.save_btn.Disable()
                self.play_btn.Enable()
                self.delete_btn.Enable()
            else:
                self.current_audio_data = (audio_data, sr)
                filename = generate_audio_filename()
                save_path = AUDIOS_DIR / filename
                try:
                    sf.write(save_path, audio_data, sr)
                    self.last_saved_path = save_path
                    self.status_label.SetLabel(f"✅ Completado en {num_chunks} partes. Guardado en: {save_path}")
                except Exception as e:
                    self.status_label.SetLabel(f"Error guardando: {e}")
                wx.Bell()
                self.save_btn.Enable()
                self.play_btn.Enable()
                self.delete_btn.Enable()
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

    def on_play(self, event):
        if self.last_saved_path and self.last_saved_path.exists():
            self.on_stop(None)
            try:
                if self.last_saved_path.suffix == '.m3u':
                    self.play_process = subprocess.Popen(["ffplay", "-nodisp", "-autoexit", self.last_saved_path.name],
                        cwd=str(self.last_saved_path.parent),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                else:
                    self.play_process = subprocess.Popen(["ffplay", "-nodisp", "-autoexit", str(self.last_saved_path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                self.stop_btn.Enable()
                self.status_label.SetLabel("Reproduciendo...")
            except Exception as e:
                wx.MessageBox(f"No se pudo reproducir: {e}\n¿Tienes ffplay instalado?", "Error", wx.OK | wx.ICON_ERROR)
        else:
            wx.MessageBox("No hay audio para reproducir.", "Información", wx.OK | wx.ICON_INFORMATION)

    def on_stop(self, event):
        if self.play_process and self.play_process.poll() is None:
            self.play_process.terminate()
            self.play_process = None
            self.stop_btn.Disable()
            self.status_label.SetLabel("Reproducción detenida.")

    def on_delete(self, event):
        if self.last_saved_path and self.last_saved_path.exists():
            self.on_stop(None)
            try:
                if self.last_saved_path.suffix == '.m3u':
                    shutil.rmtree(self.last_saved_path.parent)
                else:
                    os.remove(self.last_saved_path)
                self.last_saved_path = None
                self.current_audio_data = None
                self.save_btn.Disable()
                self.play_btn.Disable()
                self.delete_btn.Disable()
                self.status_label.SetLabel("Último audio (o proyecto) eliminado.")
            except Exception as e:
                wx.MessageBox(f"Error al eliminar: {e}", "Error", wx.OK | wx.ICON_ERROR)
        else:
            wx.MessageBox("No hay audio para eliminar.", "Información", wx.OK | wx.ICON_INFORMATION)

    def on_open_audios_folder(self, event):
        try:
            os.startfile(AUDIOS_DIR)
        except Exception as e:
            wx.MessageBox(f"No se pudo abrir: {e}", "Error", wx.OK | wx.ICON_ERROR)

class GenerationThread(threading.Thread):
    def __init__(self, parent, tts_model, ref_audio_path, ref_text,
                 target_text, language, seed, temperature, top_p,
                 x_vector_only, export_separated, voice_map, default_voice,
                 status_callback, done_callback):
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
        self.export_separated = export_separated
        self.voice_map = voice_map
        self.default_voice = default_voice
        self.status_callback = status_callback
        self.done_callback = done_callback

    def run(self):
        try:
            if self.seed != -1:
                torch.manual_seed(self.seed)
                random.seed(self.seed)
                np.random.seed(self.seed)
            gen_kwargs = {"max_new_tokens": MAX_NEW_TOKENS}
            if abs(self.temperature - 0.9) > 1e-3:
                gen_kwargs["temperature"] = self.temperature
            if abs(self.top_p - 1.0) > 1e-3:
                gen_kwargs["top_p"] = self.top_p
            if re.search(r'#\w+', self.target_text):
                segments = parse_multispeaker_text(self.target_text, self.default_voice)
                all_audio_segments = []
                sample_rate = None
                normalized_cache = {}
                for voz, texto_seg in segments:
                    if voz not in self.voice_map:
                        wx.CallAfter(self.status_callback, f"Error: Voz '{voz}' no encontrada")
                        raise Exception(f"Voz '{voz}' no encontrada")
                    info = self.voice_map[voz]
                    audio_path = info['path']
                    ref_text = info.get('ref_text')
                    if ref_text is None:
                        txt_path = audio_path.with_suffix('.txt')
                        if txt_path.exists():
                            with open(txt_path, 'r', encoding='utf-8') as f:
                                ref_text = f.read().strip()
                            self.voice_map[voz]['ref_text'] = ref_text
                        else:
                            wx.CallAfter(self.status_callback, f"Error: No hay texto de referencia para {voz}")
                            raise Exception(f"No hay texto de referencia para {voz}")
                    if audio_path not in normalized_cache:
                        wx.CallAfter(self.status_callback, f"Normalizando audio para {voz}...")
                        audio_tuple = normalize_ref_audio(audio_path)
                        if audio_tuple is None:
                            raise Exception(f"Error procesando audio para {voz}")
                        normalized_cache[audio_path] = audio_tuple
                    else:
                        audio_tuple = normalized_cache[audio_path]
                    lang_code = LANG_CODE_MAP.get(self.language, "es") if self.language != "Auto" else "es"
                    if len(texto_seg) > CHUNK_SIZE_LIMIT:
                        chunks = split_text_into_chunks(texto_seg, lang_code)
                    else:
                        chunks = [texto_seg]
                    current_segment_audio = []
                    for i, chunk in enumerate(chunks):
                        wx.CallAfter(self.status_callback, f"Generando {voz} fragmento {i+1}/{len(chunks)}...")
                        wavs, sr_out = self.tts.generate_voice_clone(
                            text=chunk,
                            language=self.language,
                            ref_audio=audio_tuple,
                            ref_text=ref_text,
                            x_vector_only_mode=self.x_vector_only,
                            **gen_kwargs
                        )
                        if sample_rate is None:
                            sample_rate = sr_out
                        current_segment_audio.append(wavs[0])
                        del wavs
                        torch.cuda.empty_cache()
                        gc.collect()
                    if current_segment_audio:
                        all_audio_segments.append((voz, np.concatenate(current_segment_audio)))
                if all_audio_segments:
                    if self.export_separated:
                        wx.CallAfter(self.done_callback, all_audio_segments, sample_rate, len(segments), True)
                    else:
                        final_audio = np.concatenate([audio for _, audio in all_audio_segments])
                        wx.CallAfter(self.done_callback, final_audio, sample_rate, len(segments), False)
                else:
                    wx.CallAfter(self.done_callback, None, None, 0, False)
            else:
                wx.CallAfter(self.status_callback, "Normalizando audio...")
                audio_tuple = normalize_ref_audio(self.ref_audio_path)
                if audio_tuple is None:
                    raise Exception("Error al procesar el audio de referencia.")
                lang_code = LANG_CODE_MAP.get(self.language, "es") if self.language != "Auto" else "es"
                wx.CallAfter(self.status_callback, "Preparando texto...")
                if len(self.target_text) > CHUNK_SIZE_LIMIT:
                    chunks = split_text_into_chunks(self.target_text, lang_code)
                else:
                    chunks = [self.target_text]
                all_audio = []
                sample_rate = None
                for i, chunk in enumerate(chunks):
                    wx.CallAfter(self.status_callback, f"Generando fragmento {i+1}/{len(chunks)}...")
                    wavs, sr_out = self.tts.generate_voice_clone(
                        text=chunk,
                        language=self.language,
                        ref_audio=audio_tuple,
                        ref_text=self.ref_text,
                        x_vector_only_mode=self.x_vector_only,
                        **gen_kwargs
                    )
                    if sample_rate is None:
                        sample_rate = sr_out
                    all_audio.append(wavs[0])
                    del wavs
                    torch.cuda.empty_cache()
                    gc.collect()
                if all_audio:
                    final_audio = np.concatenate(all_audio)
                    if self.export_separated:
                        wx.CallAfter(self.done_callback, [(self.default_voice, final_audio)], sample_rate, len(chunks), True)
                    else:
                        wx.CallAfter(self.done_callback, final_audio, sample_rate, len(chunks), False)
                else:
                    wx.CallAfter(self.done_callback, None, None, 0, False)
        except Exception as e:
            wx.CallAfter(self.done_callback, None, None, 0, False, str(e))

if __name__ == "__main__":
    app = wx.App()
    frame = MainFrame()
    app.MainLoop()