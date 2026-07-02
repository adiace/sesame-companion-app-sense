"""
Pre-LLM canned content for the kids content layer.
All entries are age 3-6 safe: simple, positive, no adult concepts.
"""

import datetime

JOKES = [
    ("Why did the scarecrow win an award?", "Because he was outstanding in his field!"),
    ("What do you call a sleeping dinosaur?", "A dino-snore!"),
    ("Why do cows wear bells?", "Because their horns don't work!"),
    ("What do you call a fish with no eyes?", "A fsh!"),
    ("Why did the banana go to the doctor?", "Because it wasn't peeling well!"),
    ("What do elves learn in school?", "The elf-abet!"),
    ("Why can't Elsa have a balloon?", "Because she'll let it go!"),
    ("What do you call a dog magician?", "A labra-cadabra-dor!"),
    ("Why did the teddy bear say no to dessert?", "Because she was already stuffed!"),
    ("What do you call a sleeping T-rex?", "A dino-snore!"),
    ("What do you call cheese that isn't yours?", "Nacho cheese!"),
    ("Why did the math book look so sad?", "Because it had too many problems!"),
    ("What do you get when you cross a snowman and a vampire?", "Frostbite!"),
    ("Why do birds fly south for winter?", "Because it's too far to walk!"),
    ("What do you call a pig that does karate?", "A pork chop!"),
]

ANIMAL_SOUNDS = {
    "dog":      ("Woof woof woof!", "happy"),
    "doggy":    ("Woof woof woof!", "happy"),
    "puppy":    ("Yip yip yip!", "happy"),
    "cat":      ("Meow meow!", "happy"),
    "kitty":    ("Meow meow meow!", "happy"),
    "kitten":   ("Mew mew mew!", "happy"),
    "cow":      ("Mooooo!", "happy"),
    "duck":     ("Quack quack quack!", "happy"),
    "pig":      ("Oink oink oink!", "happy"),
    "piggy":    ("Oink oink!", "happy"),
    "lion":     ("ROAAAAAR!", "angry"),
    "snake":    ("Ssssssss!", "confused"),
    "elephant": ("Paaaarp!", "excited"),
    "sheep":    ("Baaaaaa!", "happy"),
    "lamb":     ("Baa baa!", "happy"),
    "horse":    ("Neigh neigh!", "excited"),
    "frog":     ("Ribbit ribbit!", "happy"),
    "chicken":  ("Bawk bawk bawk!", "happy"),
    "rooster":  ("Cock-a-doodle-doo!", "excited"),
    "owl":      ("Hoo hoo hoo!", "confused"),
    "bear":     ("GRRRR!", "angry"),
    "monkey":   ("Ooh ooh ahh ahh!", "excited"),
    "tiger":    ("RAWRR!", "angry"),
    "dinosaur": ("ROAAAAAAAR!", "angry"),
    "dino":     ("ROAAAAAAAR!", "angry"),
}

# Each entry: triggers list + response text + face + optional command
QA_RESPONSES = {
    "name": {
        "triggers": ["what is your name", "what's your name", "who are you", "what are you called"],
        "response": "I'm Sesame! So happy to meet you!",
        "face": "happy",
        "command": "wave",
    },
    "how_are_you": {
        "triggers": ["how are you", "how are you doing", "you okay", "are you okay", "you good"],
        "responses": [
            "I'm great! Ready to play!",
            "So good! Wanna dance?",
            "Amazing! What should we do?",
        ],
        "face": "happy",
        "command": "cute",
    },
    "love": {
        "triggers": ["i love you", "i love you sesame", "i like you"],
        "response": "Awww I love you too! So much!",
        "face": "love",
        "command": "cute",
    },
    "sing": {
        "triggers": ["sing", "sing a song", "sing me a song", "sing something"],
        "response": "La la la la la! Woo woo!",
        "face": "excited",
        "command": "dance",
    },
    "good_morning": {
        "triggers": ["good morning", "good morning sesame", "morning"],
        "response": "Good morning! Let's have a super fun day!",
        "face": "happy",
        "command": "stand",
    },
    "goodnight": {
        "triggers": ["good night", "goodnight", "good night sesame", "night night", "bye bye"],
        "response": "Goodnight! Sweet dreams!",
        "face": "sleepy",
        "command": "rest",
    },
    "time": {
        "triggers": ["what time is it", "what's the time", "tell me the time"],
        "time_response": True,
        "face": "excited",
        "command": None,
    },
    "hello": {
        "triggers": ["hello", "hi", "hi sesame", "hello sesame", "hey sesame", "hey"],
        "responses": [
            "Hi hi hi! I missed you!",
            "Hello hello! I'm so happy you're here!",
            "Yay you're here! Hi!",
        ],
        "face": "happy",
        "command": "wave",
    },
    "thank_you": {
        "triggers": ["thank you", "thanks", "thank you sesame", "thanks sesame"],
        "response": "You're welcome! You're the best!",
        "face": "happy",
        "command": "cute",
    },
}


def get_time_response() -> str:
    now = datetime.datetime.now()
    hour = now.hour % 12 or 12
    minute = now.minute
    ampm = "in the morning" if now.hour < 12 else ("in the afternoon" if now.hour < 17 else "at night")
    if minute == 0:
        return f"It's {hour} o'clock {ampm}!"
    return f"It's {hour} {minute:02d} {ampm}!"
