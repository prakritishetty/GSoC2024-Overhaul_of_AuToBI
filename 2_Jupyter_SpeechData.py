import glob
import numpy as np
import pandas as pd
import os
import zipfile

# Path to the ZIP file
zip_file_path = './wav_BURSC_AUDIO_FINAL.zip'

# Directory where you want to extract the contents
extract_dir = './wav_BURSC_AUDIO_FINAL'

# Create the extraction directory if it doesn't exist
os.makedirs(extract_dir, exist_ok=True)

# Extract the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Extraction complete!")

from transformers import AutoFeatureExtractor, WavLMForAudioFrameClassification
import torchaudio
import torch

# Initialize the feature extractor and model
feature_extractor = AutoFeatureExtractor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus", padding=True)
model = WavLMForAudioFrameClassification.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus", output_hidden_states=True )

# Set the path to the audio folder
audio_folder = "./wav_BURSC_AUDIO_FINAL/wav_BURSC_AUDIO_FINAL"
all_files = os.listdir(audio_folder)

# Define frame and step sizes
frame_size = int(0.025*feature_extractor.sampling_rate) # 50ms frame size
step_size =int( 0.02*feature_extractor.sampling_rate) #1ms step size
print(frame_size)
print(step_size)


frames=[]
k=-1;

# Process each audio file
for index, audio_file in enumerate(all_files):
  path_to_audio_file = os.path.join(audio_folder, audio_file)
  audio_input, sample_rate = torchaudio.load(path_to_audio_file)
  # print(audio_input)

  # Process each frame in the audio file
  for i in range(step_size, len(audio_input[0]), step_size):
    k+=1
    # print("I",i)
    min_frame_size = 400
    start_time = float( (i - step_size)/sample_rate)
    end_time = float((i + max(frame_size, min_frame_size) - step_size)/sample_rate)


    # print("i",i/sample_rate)
    # print("start", (i - step_size)/sample_rate)
    # print("end", (i + max(frame_size, min_frame_size) - step_size)/sample_rate)

    # Extract the frame
    frame = audio_input[: , (i - step_size) :(i + max(frame_size, min_frame_size) - step_size)] # channels, time

    if frame.shape[1] < min_frame_size: #to handle end of audio files cases
      # print(f"Skipping frame {k} due to insufficient size: {frame.shape[1]}")
      # frame.append(torch.zeros(1,min_frame_size - frame.shape[1]))

    # Pad the frame if it's smaller than the minimum size
      padding_size = min_frame_size - frame.shape[1]
      frame = torch.nn.functional.pad(frame, (0, padding_size))
      # continue
    # print("frame_size", frame_size)
    # print("FRAME SIZE",frame.shape)
    #process the frame using the feature extractor
    inputs = feature_extractor(frame.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt")
    # print("INPUTS", inputs.input_values)

    with torch.no_grad():
      # use the model to generate embeddings for the frames
      embeddings = model(inputs.input_values).hidden_states[-1]
      # .last_hidden_state

    # Create a dictionary with frame information
    full_frame = {
        "ID":k,
        "name": audio_file,
        "start_time": start_time,
        "end_time": end_time,
        "embeddings": embeddings[0].numpy()
    }


    # print("FULL_FRAME",full_frame)


    frames.append(full_frame)
    # Print progress information
    remaining_files = len(all_files) - index
    print(f"Processed frame for audio '{audio_file}':")
    print(f"ID: {k}")
    print(f"  Start time: {start_time} seconds")
    print(f"  End time: {end_time} seconds")
    print(f"  Embeddings shape: {embeddings[0].shape}")
    print(f"Remaining files: {remaining_files}")
    print("=" * 40)


np.savez("all_frame_snippets.npz", frame_snippets=frames)

print("All frame snippets saved step by step to all_frame_snippets.npz")
print(f"Total number of frame snippets: {len(frames)}")


# Create a pandas DataFrame from the frames
data = pd.DataFrame(frames)
print(data)
