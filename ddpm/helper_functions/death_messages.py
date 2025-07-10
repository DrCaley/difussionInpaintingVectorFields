import random

def get_death_message():
    messages = [
        "💀 model got taken out back",
        "🐟 model is sleeping with the fishies",
        "🌊 model is now experiencing real ocean currents at the bottom of the ocean"
        "🖼️ model got buried in the noise",
        "📉 model dieded",
        "👮 model smoothed its last operator",
        "🖼️ model could not denoise fast enough, survival of the fittest I suppose!",
        "🧑‍🌾 model went to live with other models on a farm",
        "👁️ model got ████████ ██ █████, and went ████████",
        "⚰️ model got unalived"
        ]
    return random.choice(messages)
