import os
import requests
import io
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

# 1. Load the API key securely from the .env file
client = OpenAI(api_key="YOU_OPENAI_API_KEY")

# Ensure the key was loaded correctly
if not client.api_key:
    raise ValueError("OpenAI API Key not found. Please check your .env file.")

def generate_transformer_capabilities_image(prompt, filename_prefix="dalle_output"):
    print(f"\n--- Original Prompt ---\n'{prompt}'\n")
    print("Generating image (this may take 15-30 seconds)...")
    
    try:
        # 2. Call the DALL-E 3 Model
        # The model uses Transformer-based text encoding and latent diffusion
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard", # Use "hd" for finer detail if needed
            n=1,
        )

        # 3. Analyze the Multimodal Result
        # DALL-E 3 automatically expands your prompt to add detail and style
        revised_prompt = response.data[0].revised_prompt
        print(f"--- Transformer Revised Prompt ---\n'{revised_prompt}'\n")

        # Extract the temporary image URL
        image_url = response.data[0].url
        print(f"Image created successfully! Downloading...")

        # 4. Download and Save the Image Locally
        img_data = requests.get(image_url).content
        filename = f"{filename_prefix}.png"
        
        # Open the image using PIL to display it
        image = Image.open(io.BytesIO(img_data))
        image.show() # Opens the default image viewer
        image.save(filename)
            
        print(f"Image saved as {filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage for exploring capabilities
if __name__ == "__main__":
    # Challenge prompt 1: Complex Scene Composition and Lighting
    # Tests how well the model understands spatial relationships and textures.
    challenge_prompt_1 = (
        "A hyper-realistic close-up of a futuristic mechanical hand assembling a tiny clockwork hummingbird "
        "on a mahogany desk, with soft morning light filtering through a window."
    )
    
    # Challenge prompt 2: Stylistic Synthesis and Absurd Concepts
    # Tests how the model handles stylistic adjectives (e.g., Fauvist, 8k).
    challenge_prompt_2 = (
        "An 8k digital painting of a Fauvist-style library where the books are actually made of cascading "
        "stained glass that glows with internal light."
    )

    # Run the generation
    generate_transformer_capabilities_image(challenge_prompt_1, "clockwork_hummingbird")
    generate_transformer_capabilities_image(challenge_prompt_2, "stained_glass_library")