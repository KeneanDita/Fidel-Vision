import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# --------------------------
# Load Model and Classes
# --------------------------
MODEL_PATH = "Models/amharic_cnn.keras"
CLASS_PATH = "Models/class_names.npy"
DATA_DIR = "data"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
class_names = np.load(CLASS_PATH, allow_pickle=True)

# --------------------------
# Amharic Fidel Table (34 roots × 7 orders)
# --------------------------
amharic_table = [
    ["ሀ", "ሁ", "ሂ", "ሃ", "ሄ", "ህ", "ሆ"],
    ["ለ", "ሉ", "ሊ", "ላ", "ሌ", "ል", "ሎ"],
    ["መ", "ሙ", "ሚ", "ማ", "ሜ", "ም", "ሞ"],
    ["ሠ", "ሡ", "ሢ", "ሣ", "ሤ", "ሥ", "ሦ"],
    ["ረ", "ሩ", "ሪ", "ራ", "ሬ", "ር", "ሮ"],
    ["ሰ", "ሱ", "ሲ", "ሳ", "ሴ", "ስ", "ሶ"],
    ["ሸ", "ሹ", "ሺ", "ሻ", "ሼ", "ሽ", "ሾ"],
    ["ቀ", "ቁ", "ቂ", "ቃ", "ቄ", "ቅ", "ቆ"],
    ["ቐ", "ቑ", "ቒ", "ቓ", "ቔ", "ቕ", "ቖ"],
    ["በ", "ቡ", "ቢ", "ባ", "ቤ", "ብ", "ቦ"],
    ["ቨ", "ቩ", "ቪ", "ቫ", "ቬ", "ቭ", "ቮ"],
    ["ተ", "ቱ", "ቲ", "ታ", "ቴ", "ት", "ቶ"],
    ["ቸ", "ቹ", "ቺ", "ቻ", "ቼ", "ች", "ቾ"],
    ["ኀ", "ኁ", "ኂ", "ኃ", "ኄ", "ኅ", "ኆ"],
    ["ነ", "ኑ", "ኒ", "ና", "ኔ", "ን", "ኖ"],
    ["ኘ", "ኙ", "ኚ", "ኛ", "ኜ", "ኝ", "ኞ"],
    ["አ", "ኡ", "ኢ", "ኣ", "ኤ", "እ", "ኦ"],
    ["ከ", "ኩ", "ኪ", "ካ", "ኬ", "ክ", "ኮ"],
    ["ኸ", "ኹ", "ኺ", "ኻ", "ኼ", "ኽ", "ኾ"],
    ["ወ", "ዉ", "ዊ", "ዋ", "ዌ", "ው", "ዎ"],
    ["ዐ", "ዑ", "ዒ", "ዓ", "ዔ", "ዕ", "ዖ"],
    ["ዘ", "ዙ", "ዚ", "ዛ", "ዜ", "ዝ", "ዞ"],
    ["ዠ", "ዡ", "ዢ", "ዣ", "ዤ", "ዥ", "ዦ"],
    ["የ", "ዩ", "ዪ", "ያ", "ዬ", "ይ", "ዮ"],
    ["ደ", "ዱ", "ዲ", "ዳ", "ዴ", "ድ", "ዶ"],
    ["ገ", "ጉ", "ጊ", "ጋ", "ጌ", "ግ", "ጎ"],
    ["ጘ", "ጙ", "ጚ", "ጛ", "ጜ", "ጝ", "ጞ"],
    ["ጠ", "ጡ", "ጢ", "ጣ", "ጤ", "ጥ", "ጦ"],
    ["ጨ", "ጩ", "ጪ", "ጫ", "ጬ", "ጭ", "ጮ"],
    ["ጰ", "ጱ", "ጲ", "ጳ", "ጴ", "ጵ", "ጶ"],
    ["ጸ", "ጹ", "ጺ", "ጻ", "ጼ", "ጽ", "ጾ"],
    ["ፀ", "ፁ", "ፂ", "ፃ", "ፄ", "ፅ", "ፆ"],
    ["ፈ", "ፉ", "ፊ", "ፋ", "ፌ", "ፍ", "ፎ"],
    ["ፐ", "ፑ", "ፒ", "ፓ", "ፔ", "ፕ", "ፖ"],
]

# Flatten mapping
fidel_map = {}
i = 0
for row in amharic_table:
    for letter in row:
        if i < len(class_names):
            fidel_map[class_names[i]] = letter
            i += 1

# Detect model input size dynamically
input_shape = model.input_shape
img_height, img_width = input_shape[1:3]

# --------------------------
# Streamlit UI
# --------------------------
st.title("📖 Amharic Handwritten Character Recognition")

tab1, tab2 = st.tabs(["📂 Upload Image", "✍️ Draw Character"])

with tab1:
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")  # grayscale
        st.image(image, caption="Uploaded Image", width=200)  # smaller preview

        # Resize and preprocess
        img_resized = image.resize((img_width, img_height))
        img_array = np.array(img_resized) / 255.0
        img_array = img_array.reshape(1, img_height, img_width, 1)

        preds = model.predict(img_array)
        pred_idx = np.argmax(preds)
        pred_class = class_names[pred_idx]
        pred_letter = fidel_map.get(pred_class, "Unknown")

        st.success(f"**Predicted Class:** {pred_class}")
        st.info(f"**Mapped Fidel Character:** {pred_letter}")

with tab2:
    st.write("Draw inside the box below 👇")

    canvas = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",  # transparent background for strokes
        stroke_width=10,
        stroke_color="black",
        background_color="#EEE",  # light gray background so it's visible
        width=200,
        height=200,
        drawing_mode="freedraw",
        key="canvas_draw",  # unique key
    )

    if canvas.image_data is not None:
        # Convert to grayscale & invert (black ink on white)
        img = Image.fromarray((255 - canvas.image_data[:, :, 0]).astype(np.uint8))
        img_resized = img.resize((img_width, img_height))

        st.image(img_resized, caption="🖌️ Drawn Character", width=150)

        img_array = np.array(img_resized) / 255.0
        img_array = img_array.reshape(1, img_height, img_width, 1)

        preds = model.predict(img_array)
        pred_idx = np.argmax(preds)
        pred_class = class_names[pred_idx]
        pred_letter = fidel_map.get(pred_class, "Unknown")

        st.success(f"**Predicted Class:** {pred_class}")
        st.info(f"**Mapped Fidel Character:** {pred_letter}")
