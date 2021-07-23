from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from utils.modelutils import check_model_paths
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import argparse
import torch
import sys
import os
from audioread.exceptions import NoBackendError

def main(in_fpath,text):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    args = parser.parse_args()
    encoder.load_model(Path("encoder/saved_models/pretrained.pt"))
    synthesizer = Synthesizer(Path("synthesizer/saved_models/pretrained/pretrained.pt"))
    vocoder.load_model(Path("vocoder/saved_models/pretrained/pretrained.pt"))
    encoder.embed_utterance(np.zeros(encoder.sampling_rate))
    embed = np.random.rand(speaker_embedding_size)
    embed /= np.linalg.norm(embed)
    embeds = [embed, np.zeros(speaker_embedding_size)]
    texts = ["test 1", "test 2"]
    mels = synthesizer.synthesize_spectrograms(texts, embeds)
    mel = np.concatenate(mels, axis=1)
    no_action = lambda *args: None
    vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)
    num_generated = 0
    try:
#         message = "Reference voice: enter an audio filepath of a voice to be cloned (mp3, " \
#                   "wav, m4a, flac, ...):\n"
#         in_fpath = Path(input(message).replace("\"", "").replace("\'", ""))
        in_fpath = Path(in_fpath)
        preprocessed_wav = encoder.preprocess_wav(in_fpath)
        original_wav, sampling_rate = librosa.load(str(in_fpath))
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
#         print("Loaded file succesfully")
        embed = encoder.embed_utterance(preprocessed_wav)
#         print("Created the embedding")
#         text = input("Write a sentence (+-20 words) to be synthesized:\n")
#         print(text)
#         print(type(text))
        # If seed is specified, reset torch seed and force synthesizer reload
#         if args.seed is not None:
#             torch.manual_seed(args.seed)
#             synthesizer = Synthesizer(args.syn_model_fpath)

        # The synthesizer works in batch, so you need to put your data in a list or numpy array
        texts = [text]
        embeds = [embed]
        # If you know what the attention layer alignments are, you can retrieve them here by
        # passing return_alignments=True
        specs = synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]
#         print("Created the mel spectrogram")


        ## Generating the waveform
#         print("Synthesizing the waveform:")


        # Synthesizing the waveform is fairly straightforward. Remember that the longer the
        # spectrogram, the more time-efficient the vocoder.
        generated_wav = vocoder.infer_waveform(spec)


        ## Post-generation
        # There's a bug with sounddevice that makes the audio cut one second earlier, so we
        # pad it.
        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

        # Trim excess silences to compensate for gaps in spectrograms (issue #53)
        generated_wav = encoder.preprocess_wav(generated_wav)


        filename = "./static/demo_output.wav"
#         print(generated_wav.dtype)
        sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
#         print("Already saved the file to" + str(filename))

    except Exception as e:
        print("Caught exception: %s" % repr(e))
        print("Restarting\n")
    
    
    
    
    
    
    
    
# if __name__ == "__main__":
#     main('./trump11.wav',"This is the python world")
