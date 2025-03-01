#!/usr/bin/env python3
# https://github.com/NVIDIA-AI-IOT/whisper_trt
from whisper_trt.model import *
import whisper_trt

version = whisper_trt.__version__

MODEL="large-v3-turbo"
verbose=True

path = os.path.join(get_cache_dir(), MODEL_FILENAMES[MODEL])
make_cache_dir()

builder = MODEL_BUILDERS[MODEL]
builder.verbose = verbose

output_dir = get_cache_dir()

def build_decoder():
    decoder_path = os.path.join(output_dir, MODEL + '-decoder.pth')
    if os.path.exists(decoder_path):
        decoder = torch.load(decoder_path)
        return decoder
    else:
        decoder = builder.build_text_decoder_engine().state_dict()
        torch.save(decoder, decoder_path)
        return decoder

def build_encoder():
    encoder_path = os.path.join(output_dir, MODEL + '-encoder.pth')
    if os.path.exists(encoder_path):
        encoder = torch.load(encoder_path)
        return encoder
    else:
        encoder = builder.build_audio_encoder_engine().state_dict()
        torch.save(encoder, encoder_path)
        return encoder

def build():
    checkpoint = {
        "whisper_trt_version": version,
        "dims": asdict(load_model(builder.model).dims),
        "text_decoder_engine": build_decoder(),
        "text_decoder_extra_state": builder.get_text_decoder_extra_state(),
        "audio_encoder_engine": build_encoder(),
        "audio_encoder_extra_state": builder.get_audio_encoder_extra_state()
    }

    torch.save(checkpoint, path)

if __name__ == "__main__":
    build()

