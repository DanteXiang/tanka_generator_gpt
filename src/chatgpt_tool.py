from openai import OpenAI
import os
from dotenv import load_dotenv
import queue
import cv2
import random
import time
import threading
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
TEXT_TO_FRAME_SCALE = 900
COUNTDOWN_DURATION_S = 5
POEM_FONT_COLOR = (166, 0, 0)
INITIAL_PROMPT = (
    "You are a tanka poet. You and I will have a conversation in tanka poems. Wait for my next message, I will continuously give you a list of words and every time you need to incorporate them into a tanka poem. "
    "In every round, you need to switch between two people. One person is from the present time, and the other is a lover "
    "from 7th-century Japan. Tell me which role you are first followed by a tanka. The back-and-forth structure can continue evolving, reflecting shifts in tone, perspective, "
    "or subject matter. To maintain coherence, please feel free to respond dynamically to themes introduced in previous poems, "
    "adjust the tone to reflect contrasting or complementary viewpoints, and incorporate new words or ideas from outside the "
    "provided sources to embrace fresh inputs."
)


def chat_with_gpt(prompt, model="gpt-4o-mini"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                # {"role": "system", "content": "You are a tanka poem generator."},
                {"role": "user", "content": prompt}
            ]
        )
        # Ensure the response is properly encoded
        ret = response.choices[0].message.content.strip().encode('utf-8').decode('utf-8')
        print(ret)
        return ret
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "Sorry, I'm not able to generate a tanka poem right now."

class CameraStream:
    def __init__(self):
        # Load the local CSV file
        self.data = pd.read_csv('doc/1200_words_30x40_grid.csv', header=None)  # Update with your CSV file path
        self.word_grid = self.parse_csv_content()  # Parse the content into a grid format
        self.words_for_poem = []
        self.capture = cv2.VideoCapture(0)
        self.q = queue.Queue(maxsize=2)  # Limit queue size
        self.stop_event = threading.Event()
        self.text_to_display = "Press SPACE to generate a tanka poem..."
        self.text_lock = threading.Lock()
        self.generate_a_new_poem = False
        self.x, self.y = 50, 50
        # Get the width and height of the captured frames
        self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.last_clock = 0
        self.previous_motion_coordinates = []  # Store previous motion coordinates
        # self.map_coordinates_to_words(self.previous_motion_coordinates)
        
        # Create OpenAI client once during initialization
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def parse_csv_content(self):
        # Convert the DataFrame to a 2D list (grid format)
        return self.data.values.tolist()  # Convert DataFrame to a list of lists

    @staticmethod
    def draw_multiline_text(frame, text, position, font_scale, color, thickness):
        # Ensure text is properly formatted
        text = text.encode('utf-8').decode('utf-8')  # Ensure text is in UTF-8
        x, y = position
        for line in text.split('\n'):
            # Calculate font scale based on frame size
            frame_height, frame_width = frame.shape[:2]
            font_scale = min(frame_width, frame_height) / TEXT_TO_FRAME_SCALE  # Adjust this factor as needed

            # Use a more visible font and color
            font = cv2.FONT_HERSHEY_DUPLEX
            thickness = max(2, int(font_scale * 2))  # Thicker lines for better visibility

            # Add a dark background behind the text for better contrast
            (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)

            # Replace unsupported characters with a placeholder or remove them
            line = line.replace('—', '-')  # Replace em dash with a hyphen

            cv2.putText(frame, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
            y += int(text_height * 1.5)  # Adjust line spacing based on text height

    def detect_motion(self, frame, previous_frame):
        # Convert frames to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

        # Compute the absolute difference between the current frame and previous frame
        delta_frame = cv2.absdiff(gray_previous, gray_frame)
        thresh = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]

        # Find contours of the motion areas
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_coordinates = []

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Minimum area to consider as motion
                (x, y, w, h) = cv2.boundingRect(contour)
                motion_coordinates.append((x + w // 2, y + h // 2))  # Store center of the bounding box

        return motion_coordinates

    def draw_motion_coordinates(self, frame, motion_coordinates):
        for (x, y) in motion_coordinates:
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw circles at the coordinates

    def draw_random_motion_coordinates(self, frame, motion_coordinates):
        if motion_coordinates:
            self.previous_motion_coordinates = random.choices(motion_coordinates, k=5)  # Pick 5 random coordinates
            self.draw_motion_coordinates(frame, self.previous_motion_coordinates)

    def map_coordinates_to_words(self, coordinates):
        self.words_for_poem = []
        print("new coordinates: ")
        for (x, y) in coordinates:
            # Ensure the coordinates are within the bounds of the grid
            word_x = x // (self.frame_width // len(self.word_grid[0]))
            word_y = y // (self.frame_height // len(self.word_grid))
            self.words_for_poem.append(self.word_grid[word_y][word_x])  # Append the word at the (row, column) position
            print(word_x, word_y, self.word_grid[word_y][word_x])

    def capture_frames(self):
        previous_frame = None
        last_detection_time = time.time()  # Initialize the last detection time
        while not self.stop_event.is_set():
            ret, frame = self.capture.read()
            if not ret:
                break

            current_time = time.time()
            elapsed_time = current_time - last_detection_time
            remaining_time = max(0, COUNTDOWN_DURATION_S - int(elapsed_time))  # Calculate remaining time

            if previous_frame is not None and elapsed_time >= COUNTDOWN_DURATION_S:  # Check if 5 seconds have passed
                motion_coordinates = self.detect_motion(frame, previous_frame)
                self.draw_random_motion_coordinates(frame, motion_coordinates)
                self.map_coordinates_to_words(self.previous_motion_coordinates)
                last_detection_time = current_time  # Update the last detection time

            # Draw previously detected motion coordinates
            self.draw_motion_coordinates(frame, self.previous_motion_coordinates)
            # Draw countdown timer
            countdown_text = f"{remaining_time}"
            self.draw_multiline_text(frame, countdown_text, (10, 30), 1, POEM_FONT_COLOR, 2)
           
            with self.text_lock:
                text = self.text_to_display

            self.draw_multiline_text(frame, text, (self.x, self.y), 1.5, POEM_FONT_COLOR, 2)
             
            if not self.q.full():
                self.q.put(frame)
            previous_frame = frame  # Update previous frame
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage

    def display_frames(self):
        cv2.namedWindow('Tanka Poem Generator', cv2.WINDOW_NORMAL)
        while not self.stop_event.is_set():
            if not self.q.empty():
                frame = self.q.get()
                try:
                    cv2.imshow('Tanka Poem Generator', frame)
                except cv2.error:
                    print("Error displaying frame")
                    continue
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop_event.set()
                elif key == ord(' '):
                    with self.text_lock:
                        self.text_to_display = "Generating a new tanka poem..."
                    self.last_clock = time.time()
                    self.generate_a_new_poem = True
                    frame_height, frame_width = frame.shape[:2]
                    self.x = random.randint(50, frame_width // 2)  # Adjust based on expected text width
                    self.y = random.randint(50, frame_height // 2)  # Adjust based on expected text height
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage

    def update_text(self):
        while not self.stop_event.is_set():
            if self.generate_a_new_poem:
                poem = self.generate_tanka()
                time_spent = round(time.time() - self.last_clock, 2)
                print(f"Time spent: {time_spent} seconds")
                print(poem)
                print()
                with self.text_lock:
                    self.text_to_display = poem
                self.generate_a_new_poem = False

    def generate_tanka(self):
        return chat_with_gpt("Generate a tanka poem that's in response to the previous ones, with following words: " + ", ".join(self.words_for_poem))

    def setup_chatgpt_initial_prompt(self):
        chat_with_gpt(INITIAL_PROMPT)

    def start(self):
        threads = [
            threading.Thread(target=self.setup_chatgpt_initial_prompt),
            threading.Thread(target=self.capture_frames),
            threading.Thread(target=self.update_text)
        ]
        for thread in threads:
            thread.start()

        # Run display_frames in the main thread
        self.display_frames()

        for thread in threads:
            thread.join()
        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    stream = CameraStream()
    stream.start()
