# GSoC2024-Overhaul_of_AuToBI

Hi everyone!
I'm Prakriti Shetty, a first year Masters in Computer Science student at the University of Massachusetts Amherst.
This is my code documentation for my GSoC 2024 project on the Automatic Detection and Classification of Prosodic Events in Spoken Language. It is an attempt to be an overhaul to the AuToBI system first proposed by Andrew Rosenberg in his PhD thesis at Columbia University, with advanced state-of-the-art algorithms as of today.

For a detailed walkthrough of my documentation of entire GSoC 2024 program, this is a link to my blog: https://medium.com/@prakritishetty/gsoc-2024-experience-with-red-hen-labs-8dbf3e54056f

## How to run the code and reproduce the results

### Dataset Preprocessing
First get access to the Boston University Radio Speech Corpus (BURSC) data.
 > Let's begin with how we will handle our speech files:
   
Save and compress the audio files into a folder called BURSC_AUDIO_FINAL. These files will be in the SPH format, and our next step will be to use the 1_SPHtoWAV.py file to convert all the SPH files to a more recognizable WAV format for audio files. The output folder of WAV files will be called WAV_BURSC_AUDIO_FINAL. Save and compress all of these files in a folder called audio.

Now, each of the SPH files are about 20s long. We can't use such huge files for processing since each of the files will have multiple phrase-endings. Hence, we need to reduce it to smaller frames that are 25ms long, with a stride of 20ms. Use 2_Jupyter_SpeechData.py to extract these frames, and save this information in speech_bursc_output.xlsx which has the following columns: ['ID','NAME','START_TIME', 'END_TIME']
![image](https://github.com/user-attachments/assets/db1facf3-cae0-4e68-8e5a-73d7967b357c)



> Next, let's look at how to process the annotation files.

Since we were concerned with identifying and annotating only the intermediate and intonational phrases in our project, the files of interest were just the .BRK files. We saved and compressed all of these files into a folder called BURSC_FINAL.zip. 

Next, we use 3_Jupyter_Annotations_df.py to extract all of the information from our annotation files to an excel file called bursc_output.xlsx that has the following columns: ['ID', 'TIMESTAMP', 'INDEX']
![image](https://github.com/user-attachments/assets/e07b060b-9a80-4d55-8ac0-30f53e50def9)


> Let's now begin building the final structure for our dataset files

The audio files are pretty straightforward, just bundle up all the WAV files and compress them into a folder called audio.

Next, we need a mapping between each of our audio frames and the corresponding annotation label. 
We use 4_Jupyter_Label.py to build an excel file final_bursc_data.xlsx that has the following columns: ['ID', 'NAME','START_TIME', 'END_TIME', 'LABEL']
![image](https://github.com/user-attachments/assets/4b91bc9e-9a0c-4ec4-93ac-f9bb9aaca669)

We then deposit this excel file into a folder called text and compress it

Finally, we place the compressed audio and text folder into a folder called BURSC_DATA_TRIAL and compress it, to get our final dataset folder after the initial preprocessing.


### Running the model code

Now, we need to shift our focus to running the actual code for our model, which is contained in 5_JupyterModel.py
At the end of this, we will have folders called wavlm-base-plus-finetuned-dummy and results that will have our trained(finetuned) model parameters. 

### Other notes
To run this on the Case HPC GPU, you will need to follow the following steps - 
1. First set up your account in the Case HPC system by following Mark Turner's instructions. In the following instructions, these credentials will be referred to as accountID.
2. ssh into your account with the ssh <accountID>.pioneer.case.edu (For eg, "ssh pss107@pioneer.case.edu")
3. Request for GPU access using the sbatch <script> command (For eg, "batch batchfile.sh"). I have attached two training scripts to this repo, batchfile.sh and training.sh. Batchfile allows you to run these codes interactively in a Jupyter Notebook. The catch here is that you will need to stay connected to the Case VPN the entire time. training.sh is a workaround for that, which essentially derives from batchfile.sh, but allows you to run the .py file independently, without reliance on the VPN being connected.
4. Once you run the sbatch command, you may want to check the status of your request by running the squeue | grep <accountID> (For eg, "squeue | grep pss107")
5. For the interactive jupyter notebook users only: Once you have a GPU assigned (note your assigned GPU number), you can setup local port forwarding to access the notebook on your system using the "ssh -J <accountID>@pioneer.case.edu -L 8888:localhost:8888 -4 <accountID>@gput0<assignedGPUNumber>"). Then navigate to localhost:8888 on your system and you'll see the jupyter notebook!







