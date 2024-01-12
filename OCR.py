import os  
import ffmpeg  
import pytesseract  
import cv2  
import pandas as pd  
import re  
import feather  

# Define a function to clean text by removing special characters
def clean_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s/\\]', '', text)  # Remove special characters
    return cleaned_text.strip()  # Remove whitespace

# convert seconds to a formatted minute
def seconds_to_minutes_seconds(seconds):
    minutes = int(seconds // 60) 
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}" # length of 2 (00:00)

# video analysis, creating breakpoints from OCR results
def analyze_video(video_path, duration_limit=35*60, frame_skip=30, threshold_motion=10000, prev_frame=None):  # frame_skip - scans each 30th frame for text
    # Get video information
    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)  # gets video info the stream type is video 

    if video_stream:
        cap = cv2.VideoCapture(video_path) # reads video
        frame_number = 0

        frame_rate = cap.get(cv2.CAP_PROP_FPS) # n frame per sedon 
        total_frames = int(frame_rate * duration_limit) # tot frame for video
        
        # Create a directory to store frames
        frames_directory =  "/path_placeholder" # for pd frame
        os.makedirs(frames_directory, exist_ok=True)

        frame_data = []
        breakpoints = [] 
        current_text = ""
        start_time = 0.0

        while cap.isOpened() and frame_number < total_frames: # checks if the frames are within the limit
            ret, frame = cap.read()

            if not ret:
                break

            if frame_number % frame_skip == 0: # checks if there are no more frames left to skip
                resized_frame = cv2.resize(frame, (960, 540)) # smaller size for faster op
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # chante bg to gray for better text detection

                # calculates pixel dif.
                if prev_frame is None: # if there is no prev_frame then current one will be starting one
                    prev_frame = gray_frame
                    frame_number += 1
                    continue

                frame_diff = cv2.absdiff(gray_frame, prev_frame) # pix dif between current and prev frame
                motion_sum = frame_diff.sum()

                # if dif is greater than treshold, perform OCR
                if motion_sum > threshold_motion:
                    text = pytesseract.image_to_string(gray_frame) # if pix change is confirmed, change current bg to gray for better detection
                    timestamp = frame_number / frame_rate # time text detected
                    timestamp_str = seconds_to_minutes_seconds(timestamp) # convert to 00:00 format

                    cleaned_text = clean_text(text)  # Clean the text
                    if cleaned_text.strip():
                        if cleaned_text != current_text:
                            if current_text:
                                end_time = timestamp
                                breakpoints.append({"Start Time": seconds_to_minutes_seconds(start_time),
                                                   "End Time": seconds_to_minutes_seconds(end_time), "Text": current_text}) # df management
                            start_time = timestamp
                            current_text = cleaned_text

                    frame_filename = f"frame_{frame_number:05d}.jpg"
                    frame_path = os.path.join(frames_directory, frame_filename)
                    cv2.imwrite(frame_path, resized_frame)

            frame_number += 1

        if current_text:
            end_time = timestamp
            breakpoints.append({"Start Time": seconds_to_minutes_seconds(start_time),
                               "End Time": seconds_to_minutes_seconds(end_time), "Text": current_text})

        cap.release()
        cv2.destroyAllWindows()

        return breakpoints

    else:
        print("No video stream found.")

def main():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_filename = "terzake10.mp4"
    output_file = os.path.join(script_directory, output_filename)

    if os.path.exists(output_file):
        breakpoints = analyze_video(output_file)

        df = pd.DataFrame(breakpoints)

        # Filter the DataFrame to find rows with "terzake" in the text
        terzake_rows = df[df['Text'].str.contains('terzake', case=False)]

        if not terzake_rows.empty:
            print(terzake_rows[['Start Time', 'End Time']])

            output_directory = "/path_placeholder" 
            os.makedirs(output_directory, exist_ok=True)

            # Save the filtered rows as a JSON file
            json_filename = os.path.join(output_directory, "terzake_occurrences.json")
            terzake_rows.reset_index(drop=True, inplace=True)
            terzake_rows.to_json(json_filename, orient='records')

            # breakpoints_dataset_filename = os.path.join(output_directory, "breakpoints_dataset.feather")
            # df.reset_index(drop=True, inplace=True)
            # feather.write_dataframe(df, breakpoints_dataset_filename)
            # print(f"Breakpoints DataFrame saved as {breakpoints_dataset_filename}")
        else:
            print("No occurrences of 'terzake' found in the video.")

    else:
        print("Output file does not exist.")

if __name__ == "__main__":
    main()
