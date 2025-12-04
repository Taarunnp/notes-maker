import os   
import pytube               
from moviepy.editor import VideoFileClip     
import speech_recognition as sr 
from transformers import pipeline  
import wave  
import contextlib     
      
def download_video(youtube_url, download_path="downloads"):  
    if not os.path.exists(download_path): 
        os.makedirs(download_path)
    
    yt = pytube.YouTube(youtube_url)
    video = yt.streams.filter(only_audio=True).first()
    audio_file_path = video.download(download_path)
    
    return audio_file_path

def extract_audio_from_video(video_file_path):
    audio_file_path = video_file_path.replace(".mp4", ".wav")
    
    video_clip = VideoFileClip(video_file_path)
    video_clip.audio.write_audiofile(audio_file_path)
    return audio_file_path

# Function to transcribe audio to text using SpeechRecognition
def transcribe_audio(audio_file_path):
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
    
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Sorry, could not understand the audio.")
        return ""
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service.")
        return ""

# Function to summarize text into notes using a pre-trained NLP model (transformers pipeline)
def generate_notes(text):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=500, min_length=50, do_sample=False)
    
    return summary[0]['summary_text']

# Function to save the generated notes to a file
def save_notes(notes, filename="video_notes.txt"):
    with open(filename, "w") as f:
        f.write(notes)

# Main application function
def main():
    youtube_url = input("Enter YouTube video URL: ")
    print("Downloading video...")
    
    audio_file_path = download_video(youtube_url)
    
    print(f"Audio file downloaded: {audio_file_path}")
    
    audio_file_path = extract_audio_from_video(audio_file_path)
    
    print(f"Audio extracted to: {audio_file_path}")
    
    print("Transcribing audio...")
    transcription = transcribe_audio(audio_file_path)
    
    if transcription:
        print("Audio transcribed successfully!")
        print("Generating notes...")
        
        notes = generate_notes(transcription)
        print("Notes generated successfully!")
        
        print("Saving notes to 'video_notes.txt'...")
        save_notes(notes)
        print("Notes saved successfully!")
    else:
        print("No transcription available. Please try another video.")

if __name__ == "__main__":
    main()
