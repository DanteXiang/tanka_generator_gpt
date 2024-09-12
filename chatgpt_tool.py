from openai import OpenAI
import os
from dotenv import load_dotenv
import queue
import cv2
import random
import time
import threading

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
TEXT_TO_FRAME_SCALE = 900
def chat_with_gpt(prompt, model="gpt-4o-mini"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                # {"role": "system", "content": "You are a tanka poem generator."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "Sorry, I'm not able to generate a tanka poem right now."

class CameraStream:
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.q = queue.Queue(maxsize=2)  # Limit queue size
        self.stop_event = threading.Event()
        self.text_to_display = "Press SPACE to generate a tanka poem..."
        self.text_lock = threading.Lock()
        self.generate_a_new_poem = False
        self.x, self.y = 50, 50
        self.last_clock = 0

    @staticmethod
    def draw_multiline_text(frame, text, position, font_scale, color, thickness):
        x, y = position
        for line in text.split('\n'):
            # Calculate font scale based on frame size
            frame_height, frame_width = frame.shape[:2]
            font_scale = min(frame_width, frame_height) / TEXT_TO_FRAME_SCALE  # Adjust this factor as needed

            # Use a more visible font and color
            font = cv2.FONT_HERSHEY_DUPLEX
            color = (0, 255, 255)  # Yellow color
            thickness = max(2, int(font_scale * 2))  # Thicker lines for better visibility

            # Add a dark background behind the text for better contrast
            (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
            # cv2.rectangle(frame, (x, y - text_height - 5), (x + text_width, y + 5), (0, 0, 0), -1)

            cv2.putText(frame, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
            y += int(text_height * 1.5)  # Adjust line spacing based on text height

    def capture_frames(self):
        while not self.stop_event.is_set():
            ret, frame = self.capture.read()
            if not ret:
                break
            with self.text_lock:
                text = self.text_to_display

            self.draw_multiline_text(frame, text, (self.x, self.y), 1, (255, 255, 255), 2)
            
            if not self.q.full():
                self.q.put(frame)
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

    def start(self):
        threads = [
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

    def generate_tanka(self):
        return chat_with_gpt("Generate a tanka poem that's completely different from the previous ones.")

if __name__ == '__main__':
    stream = CameraStream()
    stream.start()
