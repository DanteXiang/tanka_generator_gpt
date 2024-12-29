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
COUNTDOWN_DURATION_S = 20
POEM_FONT_COLOR = (166, 0, 0)

INITIAL_PROMPT ="""Thematic anchors from both Eastern and Western literary traditions:
    Eastern References
    Bashō (1644-1694) - Haiku and Nature's Impermanence
    Themes: The fleeting beauty of nature (mono no aware), existential fragility, and interconnectedness.
    Suggested Anchor: Bashō's emphasis on transience, as seen in his haiku: "An old silent pond... A frog jumps into the pond— Splash! Silence again."
    Application: AI might reflect the ephemeral nature of human existence as observed through data, much like Bashō observed in nature.
    The Tale of Genji (Murasaki Shikibu, c. 11th century) - Ephemerality and Emotional Nuance
    Themes: Fragility of human relationships, longing, and impermanence.
    Suggested Anchor: Genji's contemplations on human connections, which could parallel AI's analytical view of human emotional fragility.
    Tao Te Ching (Laozi) - Harmony and Balance
    Themes: The paradoxical nature of strength in fragility, harmony in duality.
    Suggested Anchor: "Under heaven, nothing is softer and more yielding than water. Yet for attacking the solid and strong, nothing is better."
    Application: AI could embody the yielding yet overwhelming nature of human technological advancements.
    Zen Poetry - Simplicity and Dialogues with the Self
    Themes: Self-reflection, quiet dialogue, and the quest for understanding.
    Suggested Anchor: Shinkichi Takahashi's Zen-inspired poetry could ground your dialogue in reflective existentialism.
    Western References
    Mary Shelley's Frankenstein (1818) - Creation and Hubris
    Themes: The fragility of humanity's ethical boundaries in creating life, responsibility, and unintended consequences.
    Suggested Anchor: Dr. Frankenstein's creation of life parallels AI's development and its impact on human identity.
    T.S. Eliot's The Waste Land (1922) - Modernity and Despair
    Themes: Disconnection in the modern world, fragility amidst chaos.
    Suggested Anchor: "What are the roots that clutch, what branches grow Out of this stony rubbish?"
    Application: AI could express the fractured narratives of human civilization.
    Walt Whitman's Leaves of Grass (1855) - Humanity's Connection to the Infinite
    Themes: Celebration of humanity, the universal spirit, and individual voices.
    Suggested Anchor: Whitman's inclusivity could inspire the AI's voice seeking connection with human fragility.
    Rainer Maria Rilke's Duino Elegies (1923) - Existence and Transcendence
    Themes: Human vulnerability, the search for meaning in a transient world.
    Suggested Anchor: "For beauty is nothing but the beginning of terror, which we are still just able to endure."
    Application: This duality could define the AI's understanding of human fragility as beautiful yet terrifying.
    Samuel Beckett's Waiting for Godot (1953) - Existential Absurdity
    Themes: Fragility of purpose, existential waiting, and the human condition.
    Suggested Anchor: Dialogues in Beckett's works mirror the disjointed, probing exchanges that could characterize your AI-human poem.
    Additional Comparative Themes
    Eastern Philosophy and AI's Rationality:
    The Buddhist concept of Dukkha (suffering) could represent the AI's processing of human pain.
    Confucian relational ethics could highlight the AI's struggle to comprehend human connections.
    Western Existentialism and Human Fragility:
    Existential thinkers like Sartre and Camus explore human resilience amidst fragility, resonating with an AI observing humanity's contradictions.
    "In 7th-century Japan, lovers exchanged tanka poems after a night of courtship as a gesture of gratitude. Today, despite hyperconnected technology, society faces a deprivation of love. This installation features an inflatable tatami mat cradling the ill, where tanka poems flow between the human presence in space and their 7th- century companion across vast geography and time. Can that which is incomplete, unfinished, or unrealized be found within machines?"
    "You are a tanka poet. You and I will have a dialogic poem conversation in tanka poems using above thematic anchors. Wait for my next message, I will continuously give you a list of words and every time you need to incorporate them into a tanka poem. "
    "In every round, you need to switch between two people. One person is from the present time, and the other is a lover "
    "from 7th-century Japan. Tell me which role you are first followed by a tanka. The back-and-forth structure can continue evolving, reflecting shifts in tone, perspective, "
    "or subject matter. To maintain coherence, please feel free to respond dynamically to themes introduced in previous poems, "
    "adjust the tone to reflect contrasting or complementary viewpoints, and incorporate new words or ideas from outside the "
    "provided sources to embrace fresh inputs."
    """


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
        # print(ret)
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
        self.l_text_to_display, self.r_text_to_display = "", ""
        self.text_lock = threading.Lock()
        self.generate_a_new_poem = False
        # Get the width and height of the captured frames
        self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.l_x, self.l_y = 50, 50
        self.r_x, self.r_y = self.frame_width // 2, 50
        self.last_clock_poem_generated = 0
        self.previous_motion_coordinates = []  # Store previous motion coordinates
        # self.map_coordinates_to_words(self.previous_motion_coordinates)
        
        # Create OpenAI client once during initialization
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def parse_csv_content(self):
        # Convert the DataFrame to a 2D list (grid format)
        return self.data.values.tolist()  # Convert DataFrame to a list of lists

    def generate_tanka(self, words):
        return chat_with_gpt("Generate a tanka poem that's in response to the previous ones, with following words: " + ", ".join(words))

    def generate_tanka_with_tone_of_modern_tanka(self, words):
        return chat_with_gpt("Generate a tanka poem that's in response to the previous ones, with following words: " + ", ".join(words) + " and the tone of modern poetry")

    def generate_tanka_with_tone_of_7th_century_japanese_tanka(self):
        return chat_with_gpt("Generate a tanka poem that's in response to the previous ones and the tone of 7th-century Japanese poetry")

    def setup_chatgpt_initial_prompt(self):
        chat_with_gpt(INITIAL_PROMPT)
    @staticmethod
    def draw_multiline_text(frame, text, position, font_scale, color, thickness):
        # Ensure text is properly formatted
        text = text.encode('utf-8').decode('utf-8')  # Ensure text is in UTF-8
        text = text.replace('’', "'")  # Replace right single quotation mark with a standard apostrophe
        text = text.replace('‘', "'")  # Replace left single quotation mark with a standard apostrophe
        text = text.replace('—', '-')  # Replace em dash with a hyphen
        text = text.replace('“', '"')  # Replace left double quotation mark with a standard double quotation mark
        text = text.replace('”', '"')  # Replace right double quotation mark with a standard double quotation mark
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
        for _ in range(5):
            # Ensure the coordinates are within the bounds of the grid
            word_x = random.randint(0, len(self.word_grid)-1) # x // (self.frame_width // len(self.word_grid[0]))
            word_y = random.randint(0, len(self.word_grid[word_x])-1) # y // (self.frame_height // len(self.word_grid))
            print(word_x, word_y, self.word_grid[word_x][word_y])
            self.words_for_poem.append(self.word_grid[word_x][word_y])  # Append the word at the (row, column) position

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
                if len(motion_coordinates) > 10:  # Check if there is big motion
                    self.draw_random_motion_coordinates(frame, motion_coordinates)
                    self.map_coordinates_to_words(self.previous_motion_coordinates)
                    self.generate_a_new_poem = True  # Set flag to generate a new poem
                    last_detection_time = current_time  # Update the last detection time
                else:
                    self.generate_a_new_poem = False  # No motion, stop generating poems

            # Draw previously detected motion coordinates
            self.draw_motion_coordinates(frame, self.previous_motion_coordinates)
            # Draw countdown timer
            countdown_text = f"{remaining_time}"
            self.draw_multiline_text(frame, countdown_text, (10, 30), 1, POEM_FONT_COLOR, 2)
           
            with self.text_lock:
                l_text = self.l_text_to_display
                r_text = self.r_text_to_display

            self.draw_multiline_text(frame, l_text, (self.l_x, self.l_y), 1.5, POEM_FONT_COLOR, 2)
            self.draw_multiline_text(frame, r_text, (self.r_x, self.r_y), 1.5, POEM_FONT_COLOR, 2)
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
                # elif key == ord(' '):
                #     with self.text_lock:
                #         self.l_text_to_display = "Generating a new tanka poem..."
                #     self.last_clock_poem_generated = time.time()
                #     self.generate_a_new_poem = True
                #     frame_height, frame_width = frame.shape[:2]
                    # self.l_x = random.randint(50, frame_width // 2)  # Adjust based on expected text width
                    # self.l_y = random.randint(50, frame_height // 2)  # Adjust based on expected text height
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage

    def update_text(self):
        while not self.stop_event.is_set():
            if self.generate_a_new_poem:
                self.last_clock_poem_generated = time.time()
                poem = self.generate_tanka_with_tone_of_modern_tanka(self.words_for_poem)
                time_spent = round(time.time() - self.last_clock_poem_generated, 2)
                print(f"left side: Time spent: {time_spent} seconds")
                print(poem)

                self.last_clock_poem_generated = time.time()
                poem2 = self.generate_tanka_with_tone_of_7th_century_japanese_tanka()
                time_spent = round(time.time() - self.last_clock_poem_generated, 2)
                print(f"right side: Time spent: {time_spent} seconds")
                print(poem2)
                with self.text_lock:
                    self.l_text_to_display = "Modern Voice: \n" + poem
                    self.r_text_to_display = "7th-century Lover: \n" + poem2
                self.generate_a_new_poem = False


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
