import soundfile as sf
import os

input_folder = "/content/drive/MyDrive/BURSC_AUDIO_FINAL/"
output_folder = "/content/drive/MyDrive/wav_BURSC_AUDIO_FINAL"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through .sph and .spn files and convert them to .wav
for file_name in os.listdir(input_folder):
    if file_name.endswith((".SPH", ".SPN")):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name.replace(".SPH", ".wav").replace(".SPN", ".wav"))
        data, samplerate = sf.read(input_path)
        sf.write(output_path, data, samplerate)
