import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# 0 = wrist, 5 = index MCP, 9 = middle MCP, 13 = ring MCP, 17 = pinky MCP
PALM_POINTS = [0, 5, 9, 13, 17]

FINGERTIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}


class Hand:
    """Hand object :3"""

    def __init__(self, landmarks, frame_shape):
        self.landmarks = landmarks
        h, w = frame_shape[:2]
        self.points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]
        self.palm_center = self._compute_palm_center()

    # grab the midpoint between all finger ends for x
    # then midpoint between wrist and middle finger for y
    def _compute_palm_center(self):
        xs = [self.points[i][0] for i in PALM_POINTS]
        y = (self.points[0][1] + self.points[9][1]) // 2
        return sum(xs) // len(xs), y

    # draw the points on the hand 
    def draw(self, frame):
        mp_draw.draw_landmarks(frame, self.landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.circle(frame, self.palm_center, 8, (0, 0, 255), cv2.FILLED)

    # get fingertips from hand
    def fingertips(self):
        return {name: self.points[i] for name, i in FINGERTIPS.items()}

    # print the finger tips and the palm coordinates
    def print_coords(self, label):
        parts = [f"palm={self.palm_center}"]
        parts += [f"{name}={pt}" for name, pt in self.fingertips().items()]
        print(f"{label}: " + " ".join(parts))


class HandTracker:
    """Wraps MediaPipe Hands and returns Hand objects for each frame."""

    def __init__(self, max_hands=2, detection_conf=0.5, tracking_conf=0.5):
        self.hands = mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if not results.multi_hand_landmarks:
            return []
        return [Hand(lm, frame.shape) for lm in results.multi_hand_landmarks]

    def close(self):
        self.hands.close()


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    tracker = HandTracker()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            for i, hand in enumerate(tracker.process(frame)):
                hand.draw(frame)
                hand.print_coords(f"hand {i}")

            cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
